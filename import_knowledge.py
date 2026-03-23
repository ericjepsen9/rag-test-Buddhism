#!/usr/bin/env python3
"""知识库导入工具：提供原始文档，LLM 自动整理生成结构化佛教知识库文件。

用法：
    # 导入佛教经典知识
    python import_knowledge.py --type scripture --id heart_sutra --input 心经原文.txt

    # 导入教义知识
    python import_knowledge.py --type doctrine --id four_noble_truths --input 四圣谛.txt

    # 导入单文件知识（追加到现有 main.txt）
    python import_knowledge.py --type general --input 佛教常识.txt

    # 从多个文件导入
    python import_knowledge.py --type scripture --id diamond_sutra --input doc1.txt doc2.txt

    # 导入后自动构建索引
    python import_knowledge.py --type scripture --id heart_sutra --input raw.txt --build

    # 预览 LLM 整理结果（不写文件）
    python import_knowledge.py --type scripture --id heart_sutra --input raw.txt --dry-run

支持的文档格式：纯文本(.txt)、Markdown(.md)、PDF(.pdf)
"""

import os
import sys
import argparse
import json
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tempfile

from rag_runtime_config import (
    KNOWLEDGE_DIR, OPENAI_MODEL, OPENAI_API_BASE,
    SHARED_ENTITY_DIRS,
)

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """原子写入文件：先写临时文件再 rename，避免写入中途崩溃导致文件损坏。"""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# 知识类型 → (目录名, 是否单文件追加模式)
_ENTITY_TYPES = {
    # 多实例知识（每个实体独立目录）
    "scripture":   ("scripture",   False),   # 经典
    "doctrine":    ("doctrine",    False),   # 教义
    "practice":    ("practice",    False),   # 修行方法
    "sect":        ("sect",        False),   # 宗派
    "master":      ("master",      False),   # 高僧大德
    # 单文件知识（追加到同一文件）
    "general":     ("general",     True),    # 通用佛教知识
    "history":     ("history",     True),    # 佛教历史
    "ritual":      ("ritual",      True),    # 仪轨
    "glossary":    ("glossary",    True),    # 术语解释
}


def _get_openai_client():
    """获取知识库整理用 LLM client（优先 llm_client 多提供商，回退旧版）"""
    try:
        from llm_client import get_client as _get_multi_client, is_enabled as _is_enabled
        if _is_enabled("knowledge"):
            client = _get_multi_client("knowledge")
            if client is not None:
                return client
    except ImportError:
        pass
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未设置 OPENAI_API_KEY 环境变量，且未配置知识库整理用 LLM")
    try:
        from openai import OpenAI
        kwargs = {"api_key": key}
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        return OpenAI(**kwargs)
    except ImportError:
        raise RuntimeError("未安装 openai 库，请运行: pip install openai")


def _get_knowledge_model() -> str:
    """获取知识库整理用的模型名称"""
    try:
        from llm_client import get_model as _get_multi_model, is_enabled as _is_enabled
        if _is_enabled("knowledge"):
            m = _get_multi_model("knowledge")
            if m:
                return m
    except ImportError:
        pass
    return OPENAI_MODEL


def _read_input_file(path: str) -> str:
    """读取输入文件，支持 txt/md/pdf"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(p) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            return "\n\n".join(text_parts)
        except ImportError:
            raise RuntimeError("读取 PDF 需要安装 pdfplumber: pip install pdfplumber")
    else:
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
            try:
                return p.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return p.read_text(errors="replace")


def _llm_call(client, system_prompt: str, user_prompt: str,
              max_tokens: int = 4000) -> str:
    """调用 LLM API"""
    resp = client.chat.completions.create(
        model=_get_knowledge_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    if not resp.choices:
        return ""
    return (resp.choices[0].message.content or "").strip()


# ============================================================
# LLM Prompt 模板（佛教知识整理）
# ============================================================

_SYSTEM_SCRIPTURE = """你是佛教经典知识整理专家。用户会提供一份关于某部佛经或论典的原始文档。
请将其整理为结构化的知识库文档。

输出要求（JSON 格式）：
{
  "main_txt": "整理后的主文档内容",
  "faq_txt": "FAQ 问答对",
  "alias_txt": "别名和关键词",
  "scripture_name": "经典中文名",
  "scripture_aliases": ["别名1", "别名2", ...]
}

main_txt 整理规则：
1. 按以下章节标题组织（用"一、二、三..."中文编号），内容不存在的章节可以跳过：
   一、经典概述（全称、简称、梵文名、译者、所属部类）
   二、核心思想与教义（主要教理、宗旨、核心概念）
   三、经文结构与内容（品目、科判、主要段落）
   四、重要偈颂与名句
   五、修行指导（本经推荐的修行方法）
   六、相关宗派（与哪些宗派关系密切）
   七、历代注疏（重要的注解和注疏）
   八、在佛教中的地位与影响
2. 内容要具体，保留原文重要引用
3. 不要编造原文中没有的信息
4. 用"- "做要点列表

faq_txt 整理规则：
1. 从原文中提取可能的常见问题，生成 10-20 个 FAQ
2. 格式：【Q】问题\n【A】回答\n\n（每对之间空一行）
3. 覆盖主要话题：经典介绍、核心教义、修行方法、历史背景等
4. 回答简洁但完整，50-200 字

alias_txt 整理规则：
1. 每行一个别名或关键词
2. 包含：经典全称、简称、梵文名、巴利文名、英文名、核心关键词
3. 不超过 25 行"""

_SYSTEM_MULTI = """你是佛教知识整理专家。用户会提供一份关于{entity_label}的原始文档。
请将其整理为结构化的知识库文档。

输出要求（JSON 格式）：
{{
  "main_txt": "整理后的主文档内容",
  "alias_txt": "别名和关键词（如适用，否则留空）",
  "entity_name": "实体中文名",
  "entity_aliases": ["别名1", "别名2", ...]
}}

main_txt 整理规则：
1. 用"一、二、三..."中文编号组织章节
2. 内容结构根据{entity_label}类型自行组织，通常包含：
   - 概述/定义
   - 核心教义/思想
   - 修行方法/实践
   - 历史渊源/传承
   - 相关经典
   - 注意事项
3. 内容要具体，保留原文重要引用
4. 不要编造原文中没有的信息
5. 用"- "做要点列表

alias_txt 整理规则：
1. 每行一个别名或关键词
2. 包含：正式名、常见别称、梵文名、英文名
3. 不超过 15 行"""

_SYSTEM_SINGLE_FILE = """你是佛教知识整理专家。用户会提供一份关于{entity_label}的原始文档。
请将其整理为结构化的知识库内容，会追加到已有的知识库文件中。

输出要求（JSON 格式）：
{{
  "main_txt": "整理后的内容"
}}

main_txt 整理规则：
1. 用"一、二、三..."中文编号或适当的子标题组织
2. 内容要具体，保留原文重要引用
3. 不要编造原文中没有的信息
4. 用"- "做要点列表"""

_SYSTEM_REFINE = """你是佛教知识整理专家。用户对上一次的整理结果不满意，
请根据用户的修改意见进行修订。

规则：
1. 仅修改用户指出的问题部分，保留其他已满意的内容
2. 不要丢失原有的引用、术语、具体数据
3. 输出完整的修订后 JSON（不是只输出修改部分）
4. 遵循与首次整理相同的格式规范
5. JSON 格式与首次整理一致（包含 main_txt, faq_txt, alias_txt 等字段）"""


_ENTITY_LABELS = {
    "scripture":   "佛教经典",
    "doctrine":    "佛教教义",
    "practice":    "修行方法",
    "sect":        "佛教宗派",
    "master":      "高僧大德",
    "general":     "佛教通识",
    "history":     "佛教历史",
    "ritual":      "佛教仪轨",
    "glossary":    "佛教术语",
}


def refine_knowledge(client, current: dict, feedback: str,
                     raw_text: str = "", entity_type: str = "scripture") -> dict:
    """根据用户反馈修订知识库内容。"""
    user_parts = ["【当前整理结果】"]
    if current.get("main_txt"):
        user_parts.append(f"main_txt:\n{current['main_txt']}")
    if current.get("faq_txt"):
        user_parts.append(f"faq_txt:\n{current['faq_txt']}")
    if current.get("alias_txt"):
        user_parts.append(f"alias_txt:\n{current['alias_txt']}")

    user_parts.append(f"\n【用户修改意见】\n{feedback}")

    if raw_text:
        max_raw = 6000
        if len(raw_text) > max_raw:
            raw_text = raw_text[:max_raw] + f"\n...(原始文档省略 {len(raw_text) - max_raw} 字)"
        user_parts.append(f"\n【原始文档参考】\n{raw_text}")

    user_prompt = "\n\n".join(user_parts)

    if entity_type == "scripture":
        field_hint = '输出 JSON 需包含字段: main_txt, faq_txt, alias_txt'
    else:
        _, is_single = _ENTITY_TYPES.get(entity_type, ("", False))
        if is_single:
            field_hint = '输出 JSON 需包含字段: main_txt'
        else:
            field_hint = '输出 JSON 需包含字段: main_txt, alias_txt'

    system_prompt = _SYSTEM_REFINE + f"\n\n{field_hint}"
    result_text = _llm_call(client, system_prompt, user_prompt, max_tokens=4000)
    return _parse_json_result(result_text)


def _parse_json_result(text: str) -> dict:
    """解析 LLM 返回的 JSON 结果"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        # 查找第一个平衡的 JSON 对象（处理嵌套大括号）
        start = cleaned.find('{')
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            # Fallback: greedy match (may fail on multiple JSON objects)
            m = re.search(r'\{[\s\S]*\}', cleaned)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    raise ValueError(f"LLM 返回内容无法解析为 JSON：{cleaned[:500]}")
        raise ValueError(f"LLM 返回内容中未找到 JSON：{cleaned[:500]}")


def _generate_knowledge(client, raw_text: str, entity_type: str,
                         entity_id: str = "") -> dict:
    """调用 LLM 将原始文档整理为结构化知识库内容"""
    label = _ENTITY_LABELS.get(entity_type, entity_type)
    _, is_single = _ENTITY_TYPES[entity_type]

    if entity_type == "scripture":
        system = _SYSTEM_SCRIPTURE
    elif is_single:
        system = _SYSTEM_SINGLE_FILE.format(entity_label=label)
    else:
        system = _SYSTEM_MULTI.format(entity_label=label)

    user_prompt = f"以下是关于「{entity_id or label}」的原始文档，请整理为结构化知识库内容：\n\n{raw_text}"

    max_chars = 12000
    if len(raw_text) > max_chars:
        print(f"[INFO] 原始文档较长（{len(raw_text)} 字），将分段处理")
        part1 = raw_text[:max_chars]
        part2 = raw_text[max_chars:]

        user_prompt_1 = (
            f"以下是关于「{entity_id or label}」的原始文档（第1部分，共2部分）。"
            f"请先整理这部分内容：\n\n{part1}"
        )
        result_text = _llm_call(client, system, user_prompt_1, max_tokens=4000)

        if len(part2) > max_chars:
            print(f"[WARN] 文档第2部分仍超长（{len(part2)} 字），截断至 {max_chars} 字")
        user_prompt_2 = (
            f"以下是文档的第2部分，请整理并补充到之前的结果中。"
            f"输出完整的最终 JSON（合并两部分内容）：\n\n"
            f"第1部分整理结果：\n{result_text}\n\n"
            f"第2部分原文：\n{part2[:max_chars]}"
        )
        result_text = _llm_call(client, system, user_prompt_2, max_tokens=4000)
    else:
        result_text = _llm_call(client, system, user_prompt, max_tokens=4000)

    return _parse_json_result(result_text)


def _write_knowledge_files(result: dict, entity_type: str, entity_id: str,
                            dry_run: bool = False) -> Path:
    """将 LLM 整理结果写入知识库文件"""
    _, is_single = _ENTITY_TYPES[entity_type]

    if is_single:
        dir_name = _ENTITY_TYPES[entity_type][0]
        out_dir = KNOWLEDGE_DIR / "buddhism" / dir_name
    else:
        dir_name = _ENTITY_TYPES[entity_type][0]
        out_dir = KNOWLEDGE_DIR / "buddhism" / dir_name / entity_id

    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] 目标目录: {out_dir}")
        print(f"{'='*60}")
        if result.get("main_txt"):
            print(f"\n--- main.txt ({len(result['main_txt'])} 字) ---")
            print(result["main_txt"][:2000])
            if len(result["main_txt"]) > 2000:
                print(f"\n... (省略 {len(result['main_txt']) - 2000} 字)")
        if result.get("faq_txt"):
            print(f"\n--- faq.txt ({len(result['faq_txt'])} 字) ---")
            print(result["faq_txt"][:1000])
        if result.get("alias_txt"):
            print(f"\n--- alias.txt ---")
            print(result["alias_txt"])
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # main.txt
    main_txt = result.get("main_txt", "")
    if main_txt:
        main_path = out_dir / "main.txt"
        if is_single and main_path.exists():
            # 单文件模式：追加内容（使用文件锁防止并发覆盖）
            if _HAS_FCNTL:
                lock_path = main_path.with_suffix(".lock")
                with open(lock_path, "w") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    try:
                        existing = main_path.read_text(encoding="utf-8")
                        main_txt = existing.rstrip() + "\n\n" + main_txt
                        print(f"[INFO] 追加内容到已有文件: {main_path}")
                        _atomic_write(main_path, main_txt)
                    finally:
                        fcntl.flock(lf, fcntl.LOCK_UN)
            else:
                existing = main_path.read_text(encoding="utf-8")
                main_txt = existing.rstrip() + "\n\n" + main_txt
                print(f"[INFO] 追加内容到已有文件: {main_path}")
                _atomic_write(main_path, main_txt)
        else:
            _atomic_write(main_path, main_txt)
        print(f"[OK] 写入 {main_path} ({len(main_txt)} 字)")

    # faq.txt（仅经典类型）
    faq_txt = result.get("faq_txt", "")
    if faq_txt and entity_type == "scripture":
        faq_path = out_dir / "faq.txt"
        _atomic_write(faq_path, faq_txt)
        print(f"[OK] 写入 {faq_path} ({len(faq_txt)} 字)")

    # alias.txt
    alias_txt = result.get("alias_txt", "")
    if alias_txt:
        alias_path = out_dir / "alias.txt"
        _atomic_write(alias_path, alias_txt)
        print(f"[OK] 写入 {alias_path}")

    return out_dir


def _print_registration_hint(result: dict, entity_type: str, entity_id: str):
    """打印别名注册提示"""
    if entity_type == "scripture":
        aliases = result.get("scripture_aliases", [])
        if aliases:
            alias_str = json.dumps(aliases, ensure_ascii=False)
            print(f"\n[提示] 可在 knowledge/buddhism/relations.json 中添加经典关联信息")
            print(f"  经典名: {result.get('scripture_name', entity_id)}")
            print(f"  别名: {alias_str}")
    else:
        aliases = result.get("entity_aliases", [])
        if aliases:
            alias_str = json.dumps(aliases, ensure_ascii=False)
            print(f"\n[提示] 可在 rag_runtime_config.py 的相关配置中注册别名：")
            print(f'    "{entity_id}": {alias_str}')


def _build_index(entity_type: str, entity_id: str):
    """构建 FAISS 索引"""
    from build_faiss import build_for_product, build_shared

    _, is_single = _ENTITY_TYPES[entity_type]

    print(f"\n[INFO] 构建索引: buddhism")
    build_for_product("buddhism")

    # 共享知识类型也构建共享索引
    if entity_type != "scripture" and entity_type != "doctrine":
        print(f"\n[INFO] 构建共享知识索引")
        build_shared()

    print("[DONE] 索引构建完成")


def main():
    ap = argparse.ArgumentParser(
        description="佛教知识库导入工具：原始文档 → LLM 整理 → 结构化知识库文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 导入经典
  python import_knowledge.py --type scripture --id heart_sutra --input 心经.txt --build

  # 导入教义
  python import_knowledge.py --type doctrine --id four_noble_truths --input 四圣谛.txt

  # 预览（不写文件）
  python import_knowledge.py --type scripture --id test --input doc.txt --dry-run
        """,
    )
    ap.add_argument("--type", required=True, choices=list(_ENTITY_TYPES.keys()),
                    help="知识类型: scripture/doctrine/practice/sect/master/general/history/ritual/glossary")
    ap.add_argument("--id", type=str, default="",
                    help="实体ID（目录名），scripture/doctrine/practice/sect/master 必填")
    ap.add_argument("--input", nargs="+", required=True,
                    help="输入文件路径（支持 .txt .md .pdf，可指定多个）")
    ap.add_argument("--build", action="store_true",
                    help="导入后自动构建 FAISS 索引")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅预览 LLM 整理结果，不写入文件")
    ap.add_argument("--no-keywords", action="store_true",
                    help="跳过 LLM 关键词提取（同义词/分词/路由关键词）")
    args = ap.parse_args()

    entity_type = args.type
    entity_id = args.id.strip()
    _, is_single = _ENTITY_TYPES[entity_type]

    # 校验：entity_id 不允许路径遍历字符
    if entity_id and (".." in entity_id or "/" in entity_id or "\\" in entity_id):
        ap.error(f"--id 不允许包含路径分隔符或 '..'：{entity_id}")

    # 校验：非单文件类型必须提供 --id
    if not is_single and not entity_id:
        ap.error(f"--type {entity_type} 需要提供 --id 参数（作为目录名）")

    raw_parts = []
    for input_path in args.input:
        print(f"[INFO] 读取文件: {input_path}")
        text = _read_input_file(input_path)
        if text.strip():
            raw_parts.append(text.strip())
            print(f"  → {len(text)} 字")
        else:
            print(f"  → 文件为空，跳过")

    if not raw_parts:
        print("[ERROR] 所有输入文件均为空")
        sys.exit(1)

    raw_text = "\n\n---\n\n".join(raw_parts)
    print(f"[INFO] 总计 {len(raw_text)} 字原始内容")

    print(f"[INFO] 调用 LLM 整理内容（模型: {_get_knowledge_model()}）...")
    client = _get_openai_client()
    result = _generate_knowledge(client, raw_text, entity_type, entity_id)
    print(f"[OK] LLM 整理完成")

    out_dir = _write_knowledge_files(result, entity_type, entity_id,
                                      dry_run=args.dry_run)

    if not args.dry_run:
        _print_registration_hint(result, entity_type, entity_id)

    # ============================================================
    # 关键词提取：在导入时自动从原始文档中提取同义词、分词词典、路由关键词
    # ============================================================
    if not args.dry_run and not args.no_keywords:
        print(f"\n[INFO] 正在提取关键词（同义词/分词词典/路由关键词）...")
        try:
            from keyword_extractor import extract_keywords_from_document, save_extraction_result
            # 获取已有同义词用于去重
            existing_synonyms = {}
            try:
                from search_utils import _SYNONYM_MAP
                existing_synonyms = dict(_SYNONYM_MAP)
            except ImportError:
                pass
            try:
                from synonym_store import get_all_learned
                for item in get_all_learned():
                    existing_synonyms[item["original"]] = item["mapped_to"]
            except ImportError:
                pass

            kw_result = extract_keywords_from_document(
                client, _get_knowledge_model(), raw_text,
                entity_type, entity_id, existing_synonyms,
            )

            # 保存提取结果
            stats = save_extraction_result(kw_result)
            print(f"[OK] 关键词提取完成:")
            print(f"  - 同义词新增: {stats['synonyms_added']} 条（待审核）")
            print(f"  - jieba 自定义词新增: {stats['jieba_words_added']} 条")
            print(f"  - 路由关键词新增: {stats['route_keywords_added']} 条")

            if stats["synonyms_added"] > 0:
                print(f"[提示] 新增同义词需要审核，请在管理后台查看或运行：")
                print(f"  python -c \"from keyword_extractor import get_pending_review; "
                      f"import json; print(json.dumps(get_pending_review(), ensure_ascii=False, indent=2))\"")
        except Exception as e:
            print(f"[WARN] 关键词提取失败: {e}")

    elif args.no_keywords:
        print(f"[INFO] 跳过关键词提取（--no-keywords）")

    if args.build and not args.dry_run:
        _build_index(entity_type, entity_id)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("[完成] 预览模式，未写入任何文件")
    else:
        print(f"[完成] 知识已导入到: {out_dir}")
        if not args.build:
            print(f"[提示] 运行以下命令构建索引：")
            print(f"  python build_faiss.py --product buddhism")


if __name__ == "__main__":
    main()
