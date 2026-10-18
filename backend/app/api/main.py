"""API 路由汇总：只装配 APIRouter，不创建应用实例。

应用实例、CORS 与生命周期在 ``app/main.py``；本模块被其导入，避免让
``app.api`` 同时承担运行入口与路由汇总两种职责。
"""
from fastapi import APIRouter

from app.api.routes.study import router as study_router
from app.api.routes.textbook import router as textbook_router

api_router = APIRouter()
api_router.include_router(textbook_router)
api_router.include_router(study_router)
