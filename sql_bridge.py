"""SQL Bridge: 让 RAG 管线实时查询 PostgreSQL 数据库。

流程：
1. 路由层检测到「数据查询类」问题
2. LLM 将自然语言转为 SQL（仅 SELECT）
3. 安全校验 → 执行查询 → 格式化结果
4. 将结果交给 LLM 生成自然语言回答

配置文件：data/sql_config.json
"""

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sql_bridge")

BASE_DIR = Path(__file__).resolve().parent
SQL_CONFIG_FILE = BASE_DIR / "data" / "sql_config.json"

_lock = threading.Lock()
_pool = None  # psycopg2 connection pool

# ============================================================
# 问题路由：检测是否需要查 SQL
# ============================================================

_SQL_PATTERNS = re.compile(
    r"(多少人|多少位|多少个用户|多少学员|有几个人|有几位|"
    r"学习进度|修行记录|修了多少|学了多少|完成了|"
    r"谁在学|谁修了|哪些人|哪些用户|"
    r"统计|排行|排名|总数|数量|人数|"
    r"注册|登录|签到|打卡|"
    r"最近|最新|最活跃|最多|最少|"
    r"平均|总计|合计|累计|"
    r"用户.*信息|学员.*信息|个人.*记录|我的.*记录)",
    re.IGNORECASE,
)

_SQL_NEGATIVE = re.compile(
    r"(佛说|经中说|论中说|菩萨有多少|佛有多少|几种功德|几种过患|"
    r"多少品|多少偈颂|多少章|佛教有多少宗派)",
    re.IGNORECASE,
)


def needs_sql(question: str) -> bool:
    """判断用户问题是否需要查询 SQL 数据库。"""
    if not is_configured():
        return False
    if _SQL_NEGATIVE.search(question):
        return False
    return bool(_SQL_PATTERNS.search(question))


# ============================================================
# 配置管理
# ============================================================

_DEFAULT_CONFIG = {
    "enabled": False,
    "host": "localhost",
    "port": 5432,
    "database": "",
    "user": "",
    "password": "",
    "schema": "public",
    "max_connections": 3,
    "query_timeout_sec": 10,
    "max_rows": 100,
    "tables": {},
    "table_descriptions": {},
}


def _load_config() -> Dict[str, Any]:
    try:
        with SQL_CONFIG_FILE.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        merged = {**_DEFAULT_CONFIG, **cfg}
        return merged
    except FileNotFoundError:
        return dict(_DEFAULT_CONFIG)
    except Exception as e:
        logger.error("加载 SQL 配置失败: %s", e)
        return dict(_DEFAULT_CONFIG)


def save_config(cfg: Dict[str, Any]) -> None:
    SQL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SQL_CONFIG_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        safe = {k: v for k, v in cfg.items() if k != "password"}
        safe["password"] = "***" if cfg.get("password") else ""
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    tmp.replace(SQL_CONFIG_FILE)


def is_configured() -> bool:
    cfg = _load_config()
    return bool(cfg.get("enabled") and cfg.get("database"))


def get_config_safe() -> Dict[str, Any]:
    cfg = _load_config()
    safe = dict(cfg)
    if safe.get("password"):
        safe["password"] = "***"
    return safe


# ============================================================
# 数据库连接
# ============================================================

def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    with _lock:
        if _pool is not None:
            return _pool
        try:
            import psycopg2
            from psycopg2 import pool as pg_pool
        except ImportError:
            raise RuntimeError(
                "需要安装 psycopg2: pip install psycopg2-binary"
            )
        cfg = _load_config()
        _pool = pg_pool.SimpleConnectionPool(
            minconn=1,
            maxconn=cfg.get("max_connections", 3),
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 5432),
            database=cfg["database"],
            user=cfg.get("user", ""),
            password=cfg.get("password", ""),
            options=f"-c search_path={cfg.get('schema', 'public')}",
            connect_timeout=5,
        )
        logger.info("PostgreSQL 连接池已创建: %s:%s/%s",
                     cfg["host"], cfg["port"], cfg["database"])
        return _pool


def close_pool():
    global _pool
    with _lock:
        if _pool:
            _pool.closeall()
            _pool = None


def reset_pool():
    close_pool()


def test_connection() -> Dict[str, Any]:
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                ver = cur.fetchone()[0]
            return {"ok": True, "version": ver}
        finally:
            pool.putconn(conn)
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================
# Schema 自省：自动读取表结构
# ============================================================

def introspect_schema() -> Dict[str, Any]:
    """从 PostgreSQL 读取所有表的列信息，用于 Text-to-SQL prompt。"""
    cfg = _load_config()
    schema_name = cfg.get("schema", "public")
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type, is_nullable,
                       column_default
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position
            """, (schema_name,))
            rows = cur.fetchall()
        tables = {}
        for table, col, dtype, nullable, default in rows:
            if table not in tables:
                tables[table] = []
            tables[table].append({
                "column": col,
                "type": dtype,
                "nullable": nullable == "YES",
                "default": default,
            })
        return {"ok": True, "tables": tables}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        pool.putconn(conn)


def get_schema_prompt() -> str:
    """生成给 LLM 的数据库 schema 描述。"""
    cfg = _load_config()
    table_descs = cfg.get("table_descriptions", {})

    configured_tables = cfg.get("tables", {})
    if configured_tables:
        lines = ["以下是可查询的 PostgreSQL 数据库表结构：\n"]
        for tname, cols in configured_tables.items():
            desc = table_descs.get(tname, "")
            lines.append(f"表 `{tname}`" + (f" — {desc}" if desc else "") + ":")
            for c in cols:
                cname = c if isinstance(c, str) else c.get("column", c.get("name", ""))
                ctype = "" if isinstance(c, str) else c.get("type", "")
                cdesc = "" if isinstance(c, str) else c.get("description", "")
                parts = [f"  - `{cname}`"]
                if ctype:
                    parts.append(f"({ctype})")
                if cdesc:
                    parts.append(f"-- {cdesc}")
                lines.append(" ".join(parts))
            lines.append("")
        return "\n".join(lines)

    result = introspect_schema()
    if not result.get("ok"):
        return ""
    lines = ["以下是可查询的 PostgreSQL 数据库表结构：\n"]
    for tname, cols in result["tables"].items():
        desc = table_descs.get(tname, "")
        lines.append(f"表 `{tname}`" + (f" — {desc}" if desc else "") + ":")
        for c in cols:
            parts = [f"  - `{c['column']}` ({c['type']})"]
            lines.append(" ".join(parts))
        lines.append("")
    return "\n".join(lines)


# ============================================================
# SQL 安全校验
# ============================================================

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"EXECUTE|EXEC|CALL|COPY|pg_sleep|pg_terminate|pg_cancel)\b",
    re.IGNORECASE,
)

_FORBIDDEN_COMMENT = re.compile(r"(--|/\*|\*/)")


def _validate_sql(sql: str) -> Tuple[bool, str]:
    """校验 SQL 安全性：只允许 SELECT。"""
    sql_stripped = sql.strip().rstrip(";").strip()

    if _FORBIDDEN_COMMENT.search(sql_stripped):
        return False, "SQL 中不允许注释"

    if _FORBIDDEN_SQL.search(sql_stripped):
        return False, "仅允许 SELECT 查询"

    if not sql_stripped.upper().startswith("SELECT"):
        return False, "仅允许 SELECT 查询"

    lower = sql_stripped.lower()
    if "into " in lower and "select" in lower:
        if lower.index("into ") > lower.index("select"):
            return False, "不允许 SELECT INTO"

    return True, ""


# ============================================================
# Text-to-SQL：LLM 将自然语言转为 SQL
# ============================================================

_TEXT2SQL_SYSTEM = """你是一个 PostgreSQL 查询生成器。根据用户的自然语言问题和给定的数据库表结构，生成一条安全的 SELECT 查询。

规则：
1. 只生成 SELECT 查询，不能修改数据
2. 使用 LIMIT 限制结果行数（最多 {max_rows} 行）
3. 对敏感字段（密码、token 等）不要查询
4. 直接输出 SQL 语句，不要任何解释或 markdown 格式
5. 如果问题无法用给定表结构回答，输出：CANNOT_ANSWER
6. 字符串匹配使用 ILIKE 而非精确匹配
7. 日期/时间相关查询使用 PostgreSQL 函数（NOW(), INTERVAL 等）"""


def text_to_sql(question: str) -> Tuple[Optional[str], str]:
    """用 LLM 将自然语言转为 SQL。返回 (sql, error)。"""
    cfg = _load_config()
    schema_prompt = get_schema_prompt()
    if not schema_prompt:
        return None, "无法获取数据库表结构"

    max_rows = cfg.get("max_rows", 100)
    system_msg = _TEXT2SQL_SYSTEM.format(max_rows=max_rows)

    user_msg = f"{schema_prompt}\n用户问题：{question}"

    try:
        from llm_client import get_client, get_model, is_enabled
        if not is_enabled("chat"):
            return None, "LLM 未启用"
        client = get_client("chat")
        model = get_model("chat")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=500,
        )
        sql = resp.choices[0].message.content.strip()
        sql = sql.strip("`").strip()
        if sql.startswith("sql\n"):
            sql = sql[4:]
        if sql.startswith("sql "):
            sql = sql[4:]
        sql = sql.strip()

        if sql == "CANNOT_ANSWER":
            return None, "该问题无法通过数据库查询回答"

        ok, err = _validate_sql(sql)
        if not ok:
            logger.warning("LLM 生成了不安全的 SQL: %s — %s", sql, err)
            return None, f"生成的查询不安全: {err}"

        return sql, ""
    except Exception as e:
        logger.error("Text-to-SQL 失败: %s", e)
        return None, str(e)


# ============================================================
# 执行查询
# ============================================================

def execute_query(sql: str) -> Dict[str, Any]:
    """执行 SELECT 查询并返回结果。"""
    ok, err = _validate_sql(sql)
    if not ok:
        return {"ok": False, "error": err}

    cfg = _load_config()
    timeout_ms = cfg.get("query_timeout_sec", 10) * 1000

    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {timeout_ms}")
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            return {
                "ok": True,
                "columns": columns,
                "rows": [list(r) for r in rows],
                "row_count": len(rows),
            }
    except Exception as e:
        logger.error("SQL 执行失败: %s — %s", sql, e)
        return {"ok": False, "error": str(e)}
    finally:
        pool.putconn(conn)


# ============================================================
# 结果 → 自然语言回答
# ============================================================

def format_query_result(question: str, sql: str, result: Dict[str, Any]) -> str:
    """将 SQL 查询结果格式化为用户友好的自然语言回答。"""
    if not result.get("ok"):
        return f"数据查询出错：{result.get('error', '未知错误')}"

    columns = result.get("columns", [])
    rows = result.get("rows", [])

    if not rows:
        return "查询完成，暂无相关数据记录。"

    result_text = _tabulate(columns, rows)

    prompt = (
        f"用户问题：{question}\n\n"
        f"执行的 SQL 查询：\n{sql}\n\n"
        f"查询结果：\n{result_text}\n\n"
        f"请根据查询结果，用简洁的中文回答用户的问题。"
        f"如果结果是数字统计，直接说出数字；如果是列表，整理成易读的格式。"
        f"不要暴露 SQL 语句或技术细节。"
    )

    try:
        from llm_client import get_client, get_model, is_enabled
        if not is_enabled("chat"):
            return f"查询到 {len(rows)} 条记录：\n{result_text}"
        client = get_client("chat")
        model = get_model("chat")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是佛教知识问答助手，正在回答关于用户数据的问题。用中文回答，语气友善。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("结果格式化失败: %s", e)
        return f"查询到 {len(rows)} 条记录：\n{result_text}"


def _tabulate(columns: List[str], rows: List[list]) -> str:
    """简单表格化。"""
    if not columns or not rows:
        return "（无数据）"
    lines = [" | ".join(str(c) for c in columns)]
    lines.append("-" * len(lines[0]))
    for row in rows[:50]:
        lines.append(" | ".join(str(v) if v is not None else "" for v in row))
    if len(rows) > 50:
        lines.append(f"... 共 {len(rows)} 行，仅显示前 50 行")
    return "\n".join(lines)


# ============================================================
# 完整查询流程（供 rag_answer 调用）
# ============================================================

def answer_from_sql(question: str) -> Optional[str]:
    """完整流程：问题 → SQL → 执行 → 自然语言回答。

    返回 None 表示无法通过 SQL 回答。
    """
    if not is_configured():
        return None

    sql, err = text_to_sql(question)
    if not sql:
        logger.info("Text-to-SQL 无法处理: %s — %s", question, err)
        return None

    result = execute_query(sql)
    if not result.get("ok"):
        logger.warning("SQL 执行失败: %s", result.get("error"))
        return None

    return format_query_result(question, sql, result)
