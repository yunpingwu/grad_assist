"""FastAPI 应用入口：负责应用实例、生命周期与路由装配。

教材摄入（textbook_service）与检索问答（query_service）均以 APIRouter 挂载于此，
共享同一端口与 CORS 配置，统一由本模块作为 uvicorn 启动入口。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.query_service import router as query_router
from app.api.textbook_service import router as textbook_router
from app.core import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：退出时优雅释放外部连接（均幂等，未初始化时 no-op）。"""
    yield
    from app.clients import milvus_client, minio_client, mongo_client

    milvus_client.disconnect_milvus()
    minio_client.disconnect_minio()
    mongo_client.disconnect_mongo()
    logger.info("外部连接已优雅释放")


app = FastAPI(
    title="Textbook Agent",
    description="一个将教材向量化后存储入向量数据库的langgraph流程",
    version="0.1.0",
    lifespan=lifespan,
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 教材摄入与检索问答路由：共享同一 app、端口与 CORS
app.include_router(textbook_router)
app.include_router(query_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
