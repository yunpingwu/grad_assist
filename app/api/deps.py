"""FastAPI 公共依赖：匿名设备身份。"""
import os

from dotenv import load_dotenv
from fastapi import Header

load_dotenv()
def get_user_id(x_user_id: str = Header(default="anonymous")) -> str:
    """从 X-User-Id 请求头取匿名设备身份（前端 localStorage 生成），缺省回退 anonymous。

    注意：这是「软隔离」——客户端可自行伪造 user_id，仅用于区分用户、防数据串扰，
    不是安全边界；将来接入认证时改为从 token 解析，隔离逻辑不变。

    临时使用一个写死的user_id用于调试
    """
    #return x_user_id.strip() or "anonymous"
    return os.getenv("USER_ID") or "anonymous"
