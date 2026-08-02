from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# 创建 FastAPI 实例
app = FastAPI (
    title="Textbook Agent",
    description="一个将教材向量化后存储入向量数据库的langgraph流程",
    version="0.1.0",
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


