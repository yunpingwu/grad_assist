"""
MongoDB 客户端

负责连接管理（惰性单例）与默认库/集合访问。
对话等业务数据访问见 utils/chat_util.py。
"""

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import mongo_config
from app.core import logger

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """延迟初始化全局 MongoClient 单例。

    MongoClient 内部自带连接池且线程安全，进程内复用一个即可；
    serverSelectionTimeoutMS 让连接失败快速报错，避免节点挂死等待。
    """
    global _client
    if _client is None:
        _client = MongoClient(
            mongo_config.uri,
            username=mongo_config.access_key or None,
            password=mongo_config.secret_key or None,
            serverSelectionTimeoutMS=3000,
        )
        logger.info(f"MongoDB 已连接: {mongo_config.uri}")
    return _client


# ── 连接管理 ──────────────────────────────────────────────


def connect_mongo() -> None:
    """连接 MongoDB（幂等，重复调用复用单例）。"""
    get_client()


def disconnect_mongo() -> None:
    """断开 MongoDB。"""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB 已断开")


# ── 数据库/集合访问 ──────────────────────────────────────────────


def get_database() -> Database:
    """获取配置的默认数据库（惰性，连接不存在时自动建立）。"""
    return get_client()[mongo_config.db]


def get_collection(collection_name: str) -> Collection:
    """获取默认库中的集合（Mongo 懒创建，无需先建）。"""
    return get_database()[collection_name]
