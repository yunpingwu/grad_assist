"""FastAPI 公共依赖：匿名设备身份。"""
from fastapi import Header


def get_user_id(x_user_id: str = Header(default="anonymous")) -> str:
    """从 X-User-Id 取匿名身份，缺省 anonymous。

    软隔离：可伪造、不是安全边界；同一 user_id 同时充当跨设备恢复码。
    """
    return (x_user_id or "").strip() or "anonymous"
