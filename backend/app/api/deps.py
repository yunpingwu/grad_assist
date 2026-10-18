"""FastAPI 公共依赖：匿名设备身份。"""
from fastapi import Header


def get_user_id(x_user_id: str = Header(default="anonymous")) -> str:
    """从 X-User-Id 请求头取匿名设备身份（前端 localStorage 生成），缺省回退 anonymous。

    注意：这是「软隔离」——客户端可自行伪造 user_id，仅用于区分用户、防数据串扰，
    不是安全边界；将来接入认证时改为从 token 解析，隔离逻辑不变。

    user_id 同时充当「恢复码」：用户在任意设备输入同一 user_id 即可找回其教材与会话历史。
    """
    return (x_user_id or "").strip() or "anonymous"
