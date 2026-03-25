import re
import time
import threading
from collections import OrderedDict
from typing import Dict, Any, List, Optional
from rag_runtime_config import (
    PRODUCT_ALIASES, PROJECT_ALIASES, QUESTION_ROUTES,
    BUDDHIST_TERMS, CONCEPT_TERMS,
    USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE,
    LLM_REWRITE_ENABLED,
)
from search_utils import detect_terms, uniq, split_multi_question
try:
    from clarification_engine import should_clarify, generate_clarification
except ImportError:
    should_clarify = None
    generate_clarification = None

# 路由专属检索扩展词：当检测到某路由时，补充高区分度关键词帮助 BM25 命中正确 chunk
# 注意：扩展词应是该路由的核心分类词，不是具体经典名（避免引入噪音）
_ROUTE_EXPANSION = {
    "doctrine":   ["教义", "佛理"],
    "practice":   ["修行", "修法"],
    "scripture":  ["经典", "论典"],
    "sect":       ["宗派", "传承"],
    "concept":    ["概念", "法义"],
    "history":    ["佛教历史", "传播"],
    "ritual":     ["礼仪", "法会"],
    "basic":      ["佛教", "基础"],
    "life":       ["生活", "应用", "佛法与生活"],
}

# ===== LLM 查询改写 =====
# 当静态同义词/别名无法识别用户术语时，用 LLM 将其映射到知识库已有概念

def _is_llm_rewrite_enabled():
    """动态检查 LLM 改写是否启用（响应热更新）"""
    import rag_runtime_config as _cfg
    return getattr(_cfg, 'USE_OPENAI', False) and getattr(_cfg, 'LLM_REWRITE_ENABLED', False)

_LLM_REWRITE_ENABLED = USE_OPENAI and LLM_REWRITE_ENABLED  # 保留作快速路径
_LLM_REWRITE_CACHE_SIZE = 256

# 构建知识库已知术语列表（告知 LLM 可以映射到哪些词）
_KNOWN_VOCAB_PARTS: List[str] = []
for _aliases in PRODUCT_ALIASES.values():
    _KNOWN_VOCAB_PARTS.extend(_aliases[:2])
for _aliases in PROJECT_ALIASES.values():
    _KNOWN_VOCAB_PARTS.extend(_aliases[:3])
_KNOWN_VOCAB_PARTS.extend(BUDDHIST_TERMS[:20])
_KNOWN_VOCAB_PARTS.extend(CONCEPT_TERMS[:10])
_KNOWN_VOCAB = "、".join(sorted(set(_KNOWN_VOCAB_PARTS)))


class _LRUCache:
    """线程安全 LRU 缓存，支持 TTL 过期"""
    def __init__(self, maxsize: int = 256, ttl: float = 3600.0):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl  # 秒，默认 1 小时
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                ts, value = entry
                # TTL 过期检查
                if time.monotonic() - ts > self._ttl:
                    self._cache.pop(key, None)
                    self._misses += 1
                    return None
                self._cache.move_to_end(key)
                self._hits += 1
                return value
            self._misses += 1
        return None

    def put(self, key: str, value: str) -> None:
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                self._cache[key] = (now, value)
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = (now, value)

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"size": len(self._cache), "hits": self._hits,
                    "misses": self._misses, "maxsize": self._maxsize}


_llm_rewrite_cache = _LRUCache(_LLM_REWRITE_CACHE_SIZE)

# 预编译：用于判断 LLM 改写结果是否有效
_RE_NO_REWRITE = re.compile(r"(无法|不能|不确定|抱歉|sorry|NO_REWRITE)", re.IGNORECASE)


def _should_trigger_llm_rewrite(question: str, products: list, projects: list,
                                 detected_routes: list, is_chitchat: bool,
                                 is_offtopic: bool) -> bool:
    """判断是否需要触发 LLM 查询改写。

    改进策略：不仅在静态手段完全失效时触发，还在以下「边界情况」触发：
    1. 未识别到任何产品/项目且路由为 basic（最弱命中）
    2. 问题含有模糊/口语化表达，但只命中了一个通用路由关键词
    这样可以让更多「用词不精确」的查询被 LLM 纠正到规范术语。
    """
    if not _is_llm_rewrite_enabled():
        return False
    if is_chitchat or is_offtopic:
        return False
    # 问题太短（≤2字）或太长（>50字）不触发
    q = question.strip()
    if len(q) <= 2 or len(q) > 50:
        return False
    # 已识别到产品 + 明确路由 → 静态手段工作良好，不需要 LLM
    if products and detected_routes and not (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return False
    # 已识别到项目 + 明确路由 → 不需要 LLM
    if projects and detected_routes and not (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return False
    # 如果只命中了 basic 路由或没命中任何路由 → 触发 LLM 改写
    # 即使识别到了产品/项目，路由不明确也值得让 LLM 帮助理解用户意图
    if not detected_routes or (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return True
    # 检查是否包含足够的已知路由关键词（命中数 >= 2 才认为静态手段可靠）
    q_lower = q.lower()
    route_kw_hits = sum(1 for kw in _ALL_ROUTE_KEYWORDS if kw in q_lower)
    if route_kw_hits >= 2:
        return False
    # 只命中一个路由关键词且没有产品/项目 → 边界情况，触发 LLM
    if not products and not projects:
        return True
    return False


def _llm_rewrite_query(question: str) -> str:
    """调用 LLM 将用户查询中的未知术语映射到知识库已有概念。

    返回改写后的查询字符串。如果 LLM 认为无需改写或改写失败，返回空字符串。
    """
    cached = _llm_rewrite_cache.get(question)
    if cached is not None:
        return cached

    _chat_model = OPENAI_MODEL
    try:
        from llm_client import get_client as _get_multi_client, get_model as _get_multi_model, is_enabled as _is_enabled
        if _is_enabled("chat"):
            client = _get_multi_client("chat")
            if client:
                _chat_model = _get_multi_model("chat") or OPENAI_MODEL
        else:
            client = None
    except ImportError:
        client = None
    if client is None:
        try:
            from rag_answer import _get_openai_client
            client = _get_openai_client()
        except Exception:
            client = None
    if client is None:
        # 不缓存：client 可能稍后变为可用（如用户配置了 API key）
        return ""

    system_prompt = (
        "你是佛教知识库的查询改写助手。用户的问题可能包含俗称、缩写或口语化表达，"
        "请将其改写为知识库能理解的规范术语。\n"
        "规则：\n"
        "1. 只改写术语，保留用户的原始意图和问题结构\n"
        "2. 如果用户用语已经规范或你不确定对应关系，原样返回用户问题\n"
        "3. 只输出改写后的问题，不要解释\n"
        f"\n知识库已有的术语包括：{_KNOWN_VOCAB}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        if not resp.choices:
            _llm_rewrite_cache.put(question, "")
            return ""
        result = (resp.choices[0].message.content or "").strip()
        if not result or _RE_NO_REWRITE.search(result) or result == question:
            _llm_rewrite_cache.put(question, "")
            return ""
        _llm_rewrite_cache.put(question, result)
        return result
    except Exception as e:
        try:
            from rag_logger import log_error
            log_error("llm_rewrite_query", f"LLM 查询改写失败: {e}",
                      meta={"question": question[:100]})
        except Exception:
            pass
        # 不缓存 API 异常结果，允许后续重试（仅缓存 LLM 明确返回"无需改写"的情况）
        return ""


# 指代词模式：命中时用产品名/主题名替换指代词
_PRONOUN_PATTERNS = re.compile(
    r"(它|这个|那个|该产品|这款|那款)"
    r"|(那|那么).{0,2}(呢|吗|怎么样)"
)

# 追问/延续模式：命中时在问题前补充主题名
_FOLLOWUP_PATTERNS = re.compile(
    r"^还有(别的|其他|什么|吗)"
    r"|(呢|吗|怎么样|有哪些|是什么|是多少|怎么办)$"
)

# 非提问：问候、致谢、确认等不需要检索的输入
_CHITCHAT_PATTERNS = re.compile(
    r"^(你好|嗨|hi|hello|hey|您好|在吗|在不在)[啊呀哇~！!？?。.]*$"
    r"|^(谢谢|感谢|多谢|辛苦了|好的|明白了|了解了|知道了|收到|OK|ok|嗯|嗯嗯|哦|噢)[啊呀~！!。.]*$"
    r"|^(再见|拜拜|bye)[~！!。.]*$"
    r"|^[？?！!。\.…·\s~]+$",
    re.IGNORECASE,
)

# 明确的非领域词：命中这些词的问题大概率与佛教无关，不应补全
_OFFTOPIC_PATTERNS = re.compile(
    r"(天气|新闻|股票|电影|音乐|美食|旅游|游戏|足球|篮球|奥运)"
)

# 完全脱离佛教领域的问题检测：命中时直接标记为 offtopic
# 注意：只拦截明显与佛教无关的问题，佛教相关（教义、修行、经典、宗派、
# 历史、概念、礼仪等）的问题都应放行
_OFFTOPIC_FULL_PATTERNS = re.compile(
    r"(做菜|菜谱|食谱|烹饪|炒菜|炖|煮|蒸|烤|红烧|清蒸|糖醋|锅包|麻辣|酸辣|宫保|鱼香"
    r"|天气预报|气温|下雨|下雪|台风|暴雨"
    r"|股票|基金|理财|炒股|A股|港股|美股|比特币|加密货币|区块链"
    r"|电影|电视剧|综艺|动漫|追剧|票房|演员|导演|明星八卦"
    r"|歌曲|歌手|演唱会|专辑|歌词"
    r"|旅游攻略|景点|机票|酒店预订|签证|出境"
    r"|足球|篮球|排球|乒乓球|羽毛球|网球|世界杯|NBA|奥运|体育"
    r"|编程|代码|Python|Java|JavaScript|数据库|算法|软件开发"
    r"|数学题|物理|化学方程|地理"
    r"|汽车|发动机|轮胎|驾照|交通违章|加油"
    r"|手机|电脑|笔记本|平板|耳机|显卡|CPU|内存"
    r"|外卖|快递|物流|淘宝|京东|拼多多|网购"
    r"|装修|家具|水电|房贷|租房|房价|买房"
    r"|宠物|养狗|养猫|猫粮|狗粮"
    r"|减肥食谱|健身计划"
    r"|政治|选举|法律咨询|打官司|离婚|遗产"
    r"|种菜|种花|园艺|农业|养殖"
    r"|整形|美容|玻尿酸|肉毒|医美|注射填充|激光美容|热玛吉)",
    re.IGNORECASE,
)

# 佛教领域保护词：即使命中了非领域词，如果同时包含这些词则仍认为是佛教问题
_BUDDHISM_GUARD_PATTERNS = re.compile(
    r"(佛|佛教|佛法|佛学|佛陀|释迦牟尼|菩萨|阿罗汉|罗汉"
    r"|四圣谛|八正道|十二因缘|涅槃|轮回|因果|业力|业报"
    r"|空性|般若|五蕴|三宝|六度|三法印|缘起|中道|菩提"
    r"|禅|禅修|禅定|打坐|念佛|净土|持咒|诵经|修行|止观"
    r"|心经|金刚经|法华经|华严经|楞严经|地藏经|坛经|三藏"
    r"|禅宗|净土宗|天台宗|华严宗|密宗|南传|上座部|大乘|小乘|藏传"
    r"|戒律|五戒|戒定慧|三学|皈依|受戒|法会|礼仪"
    r"|寺|寺院|寺庙|法师|僧|比丘|沙弥|居士"
    r"|经|律|论|入行论|寂天|龙树|玄奘|达摩|慧能|宗喀巴"
    r"|烦恼|无明|解脱|执著|无常|无我|苦|法身|佛性|如来藏"
    r"|贪|嗔|痴|贪心|嗔心|痴心|贪嗔痴|三毒|五毒|烦恼障|所知障"
    r"|断除|对治|调伏|降伏|出离|厌离|忏悔|发愿|回向|发心"
    r"|暇满|人身难得|寿命无常|业因果|轮回过患"
    r"|布施|持戒|忍辱|安忍|精进|静虑|智慧|波罗蜜"
    r"|学处|菩萨戒|别解脱戒|律仪|三聚净戒"
    r"|正知|正念|不放逸|自他交换|自他相换|慈悲|大悲|利他"
    r"|闻思修|闻|思|修|加行|前行|正行|资粮|道次第)"
)

# 产品/主题切换意图：用户想问另一个主题，不应继承历史
_SWITCH_PATTERNS = re.compile(
    r"(换一个|另一个|另一种|其他的|别的|不同的|有没有其他|还有什么)"
)

# 隐含关联模式：问题本身关于某个主题但缺少主语
_IMPLICIT_TOPIC_PATTERNS = re.compile(
    r"(区别|对比|优势|好处|原理|作用|含义|意义"
    r"|怎么理解|怎么修|怎么做|怎么念|怎么持|如何修|如何理解"
    r"|有什么用|有什么好处|有什么意义|什么意思|什么区别"
    r"|是什么|是怎样|是多少|注意什么|注意事项)"
    r"|(效果|功效|作用|意义).{0,4}(吗|呢|怎么样|如何|好不好|怎样)"
)

# 纠正前缀：用户纠正上一个回答时的引导语，检索时应去除
_CORRECTION_PREFIX = re.compile(
    r"^(不对[，,]?|不是[，,]?(这个[，,]?)?|我(想|要)?问的是|我说的是)"
)

# 所有路由关键词汇集，用于判断问题是否包含领域词
_ALL_ROUTE_KEYWORDS: set = set()
for _kws in QUESTION_ROUTES.values():
    _ALL_ROUTE_KEYWORDS.update(kw.lower() for kw in _kws)

# 所有产品/项目别名汇集（小写），用于领域相关性检测
_ALL_PRODUCT_TERMS: set = set()
for _aliases in PRODUCT_ALIASES.values():
    _ALL_PRODUCT_TERMS.update(a.lower() for a in _aliases)
for _aliases in PROJECT_ALIASES.values():
    _ALL_PRODUCT_TERMS.update(a.lower() for a in _aliases)


def _has_domain_relevance(text: str) -> bool:
    """检查查询是否包含任何佛教领域相关词汇。

    用于拦截与佛教完全无关的短查询，
    避免系统对无关查询强行返回知识库内容。
    """
    if not text:
        return False
    low = text.lower()

    # 1. 包含佛教保护词
    if _BUDDHISM_GUARD_PATTERNS.search(text):
        return True

    # 2. 包含路由关键词
    for kw in _ALL_ROUTE_KEYWORDS:
        if kw in low:
            return True

    # 3. 包含产品/项目名
    for term in _ALL_PRODUCT_TERMS:
        if term in low:
            return True

    return False

# 预计算小写路由关键词
_QUESTION_ROUTES_LOWER = {
    route: [kw.lower() for kw in keywords]
    for route, keywords in QUESTION_ROUTES.items()
}


_MAX_HISTORY_SCAN = 12  # 最多回溯的用户消息数


def _extract_history_context(history: List[Dict]) -> Dict[str, Any]:
    """从对话历史中提取结构化上下文信息。

    最多回溯 _MAX_HISTORY_SCAN 条用户消息。当 product + route + projects
    全部找到时提前退出。

    返回:
        product: 最近提到的佛教主题标准名
        product_id: 主题 ID
        projects: 最近提到的修行法门/经典列表
        route: 最近的路由主题
        last_user_q: 最近的用户问题原文
        last_routed_q: 最近一条含路由关键词的用户问题
    """
    ctx: Dict[str, Any] = {
        "product": "", "product_id": "", "projects": [],
        "route": "", "last_user_q": "",
        "last_routed_q": "",
    }
    user_count = 0
    scan_slice = history[-(2 * _MAX_HISTORY_SCAN):] if len(history) > 2 * _MAX_HISTORY_SCAN else history
    for item in reversed(scan_slice):
        raw_content = item.get("content")
        content = (str(raw_content) if raw_content is not None else "")[:500]
        if item.get("role") != "user":
            continue
        user_count += 1

        if not ctx["last_user_q"]:
            ctx["last_user_q"] = content

        if not ctx["product"]:
            products = detect_terms(content, PRODUCT_ALIASES)
            if products:
                pid = products[0]
                aliases = PRODUCT_ALIASES.get(pid, [])
                if aliases:
                    ctx["product"] = aliases[0]
                    ctx["product_id"] = pid

        if not ctx["projects"]:
            projects = detect_terms(content, PROJECT_ALIASES)
            if projects:
                proj_names = []
                for pj in projects:
                    aliases = PROJECT_ALIASES.get(pj, [])
                    if aliases:
                        proj_names.append(aliases[0])
                ctx["projects"] = proj_names

        if not ctx["route"]:
            content_lower = content.lower()
            for route, keywords in _QUESTION_ROUTES_LOWER.items():
                if any(kw in content_lower for kw in keywords):
                    ctx["route"] = route
                    if not ctx["last_routed_q"]:
                        ctx["last_routed_q"] = content
                    break

        # 全部找到 -> 提前退出
        if ctx["product"] and ctx["route"] and ctx["projects"]:
            break
        if ctx["product"] and ctx["route"] and user_count >= 3:
            break
        if user_count >= _MAX_HISTORY_SCAN:
            break
    return ctx


def _resolve_context(question: str, history_ctx: Dict[str, Any]) -> str:
    """基于已提取的历史上下文解析指代和省略，补全当前问题。

    策略（从高优先级到低）：
    1. 当前问题已包含主题名 -> 不需要补全
    2. 指代词（它/这个/那个）-> 从历史替换为具体主题名
    3. 追问句式（"...呢/吗/怎么样"）-> 补充主题名
    4. 隐含关联（含领域词但无主语）-> 补充主题名
    5. 含路由关键词但无主题名 -> 补充主题名
    """
    q = question.strip()

    history_product = history_ctx.get("product", "")
    if not history_product:
        return q

    # 当前问题已有明确主题名，不需要补全
    if detect_terms(q, PRODUCT_ALIASES):
        return q

    # 排除明显的非领域问题
    if _OFFTOPIC_PATTERNS.search(q):
        return q

    # 排除主题切换意图
    if _SWITCH_PATTERNS.search(q):
        return q

    # 构建补全前缀
    prefix_parts = [history_product]
    if not detect_terms(q, PROJECT_ALIASES):
        for pj in history_ctx.get("projects", []):
            prefix_parts.append(pj)
    prefix = " ".join(prefix_parts)

    # 模式1: 指代词替换
    m = _PRONOUN_PATTERNS.search(q)
    if m:
        pos = m.start()
        if pos == 0:
            resolved = prefix + q[m.end():]
            return resolved
        prev = q[pos - 1]
        if prev in "，。？！；、,;!? " and (pos < 2 or q[pos - 2] not in "和跟比与"):
            resolved = q[:pos] + prefix + q[m.end():]
            return resolved

    # 模式2: 追问/延续补全
    if _FOLLOWUP_PATTERNS.search(q):
        return f"{prefix} {q}"

    # 模式3: 隐含关联
    if _IMPLICIT_TOPIC_PATTERNS.search(q) and len(q) <= 30:
        return f"{prefix} {q}"

    # 模式4: 含路由关键词但无主题名
    q_lower = q.lower()
    has_route_keyword = any(kw in q_lower for kw in _ALL_ROUTE_KEYWORDS)
    if has_route_keyword and len(q) <= 40:
        return f"{prefix} {q}"

    return q


def _detect_route_for_expansion(q: str) -> List[str]:
    """检测问题命中的路由，返回匹配到的路由列表"""
    q_lower = q.lower()
    matched = []
    for route, keywords in _QUESTION_ROUTES_LOWER.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(route)
    return matched


def _build_history_summary_and_pairs(
    history: List[Dict], max_turns: int = 3, max_pairs: int = 3
) -> tuple:
    """单次反向扫描同时构建多轮摘要和 Q&A 对。

    返回 (summary: str, pairs: List[Dict])。
    summary: "四圣谛是什么 → 八正道呢 → 禅修方法" 格式。
    pairs: [{"user": "...", "assistant": "..."}, ...] 格式。
    """
    recent_questions: List[str] = []
    pairs: List[Dict] = []
    current_assistant = ""
    summary_done = False
    pairs_done = False

    for item in reversed(history):
        if summary_done and pairs_done:
            break
        role = item.get("role", "")
        raw = item.get("content")
        content = (str(raw) if raw is not None else "").strip()

        if role == "assistant":
            if not pairs_done and not current_assistant:
                current_assistant = content[:200]
        elif role == "user":
            if not summary_done and content:
                recent_questions.append(content)
                if len(recent_questions) >= max_turns:
                    summary_done = True
            if not pairs_done and current_assistant:
                pairs.append({"user": content, "assistant": current_assistant})
                current_assistant = ""
                if len(pairs) >= max_pairs:
                    pairs_done = True

    recent_questions.reverse()
    pairs.reverse()
    return " → ".join(recent_questions), pairs


def _build_history_summary(history: List[Dict], max_turns: int = 3) -> str:
    """向后兼容包装：委托给合并后的单次扫描函数。"""
    summary, _ = _build_history_summary_and_pairs(history, max_turns=max_turns, max_pairs=0)
    return summary


def _build_history_pairs(history: List[Dict], max_pairs: int = 3) -> List[Dict]:
    """向后兼容包装：委托给合并后的单次扫描函数。"""
    _, pairs = _build_history_summary_and_pairs(history, max_turns=0, max_pairs=max_pairs)
    return pairs


def is_chitchat(text: str) -> bool:
    """判断输入是否为闲聊/问候/致谢等非提问"""
    return bool(_CHITCHAT_PATTERNS.match((text or "").strip()))


def rewrite_query(question: str, history: Optional[List[Dict]] = None,
                   _cached_ctx: Optional[Dict] = None) -> Dict[str, Any]:
    raw = (question or "").strip()

    # 快速判断：非提问（问候/致谢/确认）直接返回，跳过后续处理
    chitchat = bool(_CHITCHAT_PATTERNS.match(raw))

    # 非佛教领域检测：命中非领域关键词且不含佛教保护词 -> 标记为 offtopic
    is_offtopic = bool(
        _OFFTOPIC_FULL_PATTERNS.search(raw)
        and not _BUDDHISM_GUARD_PATTERNS.search(raw)
    )

    # 领域无关兜底检测：短查询（≤4字）且不含任何佛教领域词汇时也标记为 offtopic
    # 防止完全无关查询被默认路由到 basic 并返回知识库内容
    # 注意：≤6字时误杀率太高（如"如何断除贪心"6字），降低到4字
    if not is_offtopic and not chitchat and len(raw) <= 4:
        if not _has_domain_relevance(raw):
            is_offtopic = True

    # 清理纠正前缀
    cleaned = _CORRECTION_PREFIX.sub("", raw).strip() if _CORRECTION_PREFIX.search(raw) else raw

    # 提取历史上下文（支持缓存复用）
    history_ctx: Dict[str, Any] = {}
    if _cached_ctx is not None:
        history_ctx = _cached_ctx
    elif history:
        history_ctx = _extract_history_context(history)

    # 上下文补全：解析指代词和省略
    q = _resolve_context(cleaned, history_ctx) if (history_ctx and not chitchat) else cleaned
    context_resolved = (q != cleaned)

    search_q = q

    products = detect_terms(q, PRODUCT_ALIASES)
    projects = detect_terms(q, PROJECT_ALIASES)

    buddhist = [x for x in BUDDHIST_TERMS if x in q]
    concepts = [x for x in CONCEPT_TERMS if x in q]

    expanded_terms: List[str] = []
    for pid in products:
        expanded_terms.extend(PRODUCT_ALIASES.get(pid, [])[:4])
    for pj in projects:
        expanded_terms.extend(PROJECT_ALIASES.get(pj, [])[:3])
    expanded_terms.extend(buddhist)
    expanded_terms.extend(concepts)

    # 路由感知扩展
    detected_routes = _detect_route_for_expansion(q)
    for rt in detected_routes:
        expanded_terms.extend(_ROUTE_EXPANSION.get(rt, []))

    # 历史路由继承
    if not detected_routes and history_ctx.get("route"):
        inherited_rt = history_ctx["route"]
        expanded_terms.extend(_ROUTE_EXPANSION.get(inherited_rt, []))

    sub_questions = split_multi_question(q)
    expanded_query = " ".join(uniq([search_q] + expanded_terms))

    # ---- LLM 查询改写 ----
    llm_rewritten = ""
    if _should_trigger_llm_rewrite(q, products, projects, detected_routes,
                                    chitchat, is_offtopic):
        llm_rewritten = _llm_rewrite_query(q)
        if llm_rewritten:
            try:
                from synonym_store import save_learned
                save_learned(q, llm_rewritten)
            except Exception:
                pass
            search_q = llm_rewritten
            products = detect_terms(llm_rewritten, PRODUCT_ALIASES) or products
            projects = detect_terms(llm_rewritten, PROJECT_ALIASES) or projects
            detected_routes = _detect_route_for_expansion(llm_rewritten) or detected_routes
            for rt in detected_routes:
                expanded_terms.extend(_ROUTE_EXPANSION.get(rt, []))
            expanded_query = " ".join(uniq([q, llm_rewritten] + expanded_terms))
            sub_questions = split_multi_question(llm_rewritten)

    # 构建多轮历史摘要
    history_summary = ""
    history_pairs: List[Dict] = []
    last_user_q = ""
    if history:
        history_summary, history_pairs = _build_history_summary_and_pairs(history)
        last_user_q = history_ctx.get("last_user_q", "")

    # ---- 消歧引导：查询模糊且缺乏上下文时生成候选选项 ----
    clarification = None
    needs_clarification = False
    if should_clarify is not None and not chitchat and not is_offtopic:
        history_product = history_ctx.get("product", "")
        history_route = history_ctx.get("route", "")
        if should_clarify(
            raw, products, projects, detected_routes,
            history_product=history_product,
            history_route=history_route,
            is_chitchat=chitchat,
            is_offtopic=is_offtopic,
        ):
            clarification = generate_clarification(
                raw,
                products=products,
                projects=projects,
                history_product=history_product,
            )
            if clarification:
                needs_clarification = True

    return {
        "original": q,
        "raw_input": raw,
        "search_query": search_q,
        "is_chitchat": chitchat,
        "is_offtopic": is_offtopic,
        "context_resolved": context_resolved,
        "expanded": expanded_query,
        "products": products,
        "projects": projects,
        "buddhist_terms": uniq(buddhist),
        "concepts": uniq(concepts),
        "sub_questions": sub_questions,
        "detected_routes": detected_routes,
        "llm_rewritten": llm_rewritten,
        "history_summary": history_summary,
        "history_pairs": history_pairs,
        "last_user_q": last_user_q,
        "last_routed_q": history_ctx.get("last_routed_q", ""),
        "needs_clarification": needs_clarification,
        "clarification": clarification,
    }
