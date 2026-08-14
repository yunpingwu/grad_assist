"""
对话历史工具

为 query 图的多轮对话提供 MongoDB 持久化：
- load_chat_history: 读取最近 N 轮历史，格式化为纯文本，供 rewrite_query 消歧。
- append_turn: 追加一轮 user/assistant 消息（会话不存在则创建）。

存储是旁路：任何读写失败只记 warning 降级，不阻断问答主链路。
底层连接见 clients/mongo_client.py。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from pymongo.errors import PyMongoError

from app.clients import mongo_client
from app.config import mongo_config
from app.core import logger

# 中国时区（UTC+8）：消息时间戳按本地时间存储与展示
# 注：pymongo 对带 tzinfo 的 datetime 会转成 UTC 存储、读回 naive UTC，导致库里看到的是 UTC
# 而非北京时间；故写入时先取中国时刻再去掉 tzinfo，使 BSON 中直接保存北京时间字面值。
CHINA_TZ = ZoneInfo("Asia/Shanghai")

# 进 prompt 的历史轮数上限（存储保留全部）
MAX_HISTORY_TURNS = 5
# 会话列表单次返回上限
MAX_SESSIONS = 50
# 会话历史单次返回轮数上限（1 轮 = user + assistant 两条；防前端一次渲染过多消息）
MAX_HISTORY_TURNS_LIMIT = 50
# 会话列表 preview 截断长度
PREVIEW_LENGTH = 60


def _format_history(messages: list[dict]) -> str:
    """将消息列表格式化为 "用户: ...\n助手: ..." 纯文本。"""
    lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def load_chat_history(session_id: str, limit: int = MAX_HISTORY_TURNS) -> str:
    """加载最近 limit 轮历史对话，格式化为纯文本。

    Args:
        session_id: 会话 ID（文档 _id）。
        limit: 取最近多少轮（每轮含 user + assistant 两条消息）。

    Returns:
        纯文本历史；无会话或读取失败返回空字符串（降级为单轮问答）。
    """
    if not session_id:
        return ""
    try:
        doc = mongo_client.get_collection(mongo_config.chat_collection).find_one({"_id": session_id})
    except PyMongoError as exc:
        logger.warning(f"读取会话历史失败(session_id={session_id}): {exc}")
        return ""
    if not doc:
        return ""
    messages = doc.get("messages", [])
    recent = messages[-limit:]
    return _format_history(recent)


def append_turn(
    session_id: str,
    textbook_name: str,
    user_msg: str,
    assistant_msg: str,
    *,
    rewritten_query: str | None = None,
    images: list[dict] | None = None,
) -> None:
    """追加一轮对话（user + assistant）到会话文档；会话不存在则创建。

    幂等设计：$push 原子追加两条消息；$setOnInsert 保证会话级元数据只写一次。
    失败仅记 warning——答案已生成，落库失败不应让整轮报错。

    Args:
        session_id: 会话 ID（文档 _id）。
        textbook_name: 教材名，首次创建会话时写入（一个会话绑定一本教材）。
        user_msg: 用户原始问题。
        assistant_msg: 生成的答案（可能含 markdown 图片引用）。
        rewritten_query: 可选的检索改写问题，随 assistant 消息存档。
        images: 可选的图片引用列表，随 assistant 消息存档。
    """
    if not session_id:
        logger.warning("append_turn 跳过: session_id 为空")
        return

    # 中国本地时间（naive）：直接存北京时间字面值，见文件头 _CHINA_TZ 注释
    now = datetime.now(CHINA_TZ).replace(tzinfo=None)
    assistant_doc: dict = {"role": "assistant", "content": assistant_msg, "created_at": now}
    if rewritten_query:
        assistant_doc["rewritten_query"] = rewritten_query
    if images:
        assistant_doc["images"] = images

    try:
        mongo_client.get_collection(mongo_config.chat_collection).update_one(
            {"_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_msg, "created_at": now},
                            assistant_doc,
                        ]
                    }
                },
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "textbook_name": textbook_name,
                    "created_at": now,
                },
            },
            upsert=True,
        )
    except PyMongoError as exc:
        logger.warning(f"保存对话失败(session_id={session_id}): {exc}")


def list_sessions(textbook_name: str, limit: int = MAX_SESSIONS) -> list[dict]:
    """列出某教材的会话摘要，按最近更新倒序。

    Args:
        textbook_name: 教材名（会话文档按此过滤）。
        limit: 单次返回条数上限。

    Returns:
        形如 [{session_id, updated_at, message_count, preview}] 的列表；
        无会话或读取失败返回空列表（旁路降级）。
    """
    if not textbook_name:
        return []
    try:
        pipeline = [
            {"$match": {"textbook_name": textbook_name}},
            {"$sort": {"updated_at": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "updated_at": 1,
                    "message_count": {"$size": "$messages"},
                    # 倒数第二条 = 用户提问（消息成对追加 [user, assistant]）
                    "last_message": {"$arrayElemAt": ["$messages", -2]},
                }
            },
        ]
        sessions = []
        for d in mongo_client.get_collection(mongo_config.chat_collection).aggregate(pipeline):
            last = d.get("last_message") or {}
            role = "我" if last.get("role") == "user" else ""
            preview = (last.get("content") or "")[:PREVIEW_LENGTH]
            # 库里是 naive 北京时间，补 +08:00 偏移返回，前端 new Date() 可正确本地化显示
            updated_at = d.get("updated_at")
            if isinstance(updated_at, datetime):
                updated_at = updated_at.replace(tzinfo=CHINA_TZ).isoformat()
            sessions.append(
                {
                    "session_id": d["_id"],
                    "updated_at": updated_at,
                    "message_count": d.get("message_count", 0),
                    "preview": f"{role}: {preview}",
                }
            )
        return sessions
    except PyMongoError as exc:
        logger.warning(f"读取会话列表失败(textbook_name={textbook_name}): {exc}")
        return []


def get_session_messages(session_id: str, max_turns: int = MAX_HISTORY_TURNS_LIMIT) -> list[dict]:
    """获取某会话最近 max_turns 轮的结构化消息（历史回显用）。

    Args:
        session_id: 会话 ID（文档 _id）。
        max_turns: 返回的轮数上限（1 轮 = user + assistant 两条），只取尾部，
                   防止超长会话一次性拉全量导致前端卡顿；存储保留全部。

    Returns:
        消息列表 [{role, content, created_at, rewritten_query?, images?}]；
        无会话或读取失败返回空列表（旁路降级）。
    """
    if not session_id:
        return []
    try:
        doc = mongo_client.get_collection(mongo_config.chat_collection).find_one({"_id": session_id})
    except PyMongoError as exc:
        logger.warning(f"读取会话历史失败(session_id={session_id}): {exc}")
        return []
    if not doc:
        return []
    messages = doc.get("messages", [])
    return messages[-(max_turns * 2) :]


def delete_session(session_id: str) -> bool:
    """删除单个会话文档（含其全部消息）。

    Args:
        session_id: 会话 ID（文档 _id）。

    Returns:
        True 表示确实删除了一个会话；False 表示会话不存在或删除失败（旁路降级，
        与其余对话读写一致：失败仅记 warning，不向上抛）。
    """
    if not session_id:
        return False
    try:
        result = mongo_client.get_collection(mongo_config.chat_collection).delete_one({"_id": session_id})
        return result.deleted_count > 0
    except PyMongoError as exc:
        logger.warning(f"删除会话失败(session_id={session_id}): {exc}")
        return False
