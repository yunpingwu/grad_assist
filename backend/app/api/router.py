"""API 路由汇总：只装配 APIRouter；应用实例与生命周期在 ``app/main.py``。"""
from fastapi import APIRouter

from app.api.routes.study import router as study_router
from app.api.routes.textbook import router as textbook_router

api_router = APIRouter()
api_router.include_router(textbook_router)
api_router.include_router(study_router)
