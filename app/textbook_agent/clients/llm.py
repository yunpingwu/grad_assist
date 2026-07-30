import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


def get_llm_client():
    # 加载环境变量文件
    load_dotenv()

    # 创建一个LLM模型
    model = init_chat_model(
        model="",
        model_provider="openai",
        temperature=0.7,
        api_key=os.getenv("ALIBABA_API_KEY"),
        base_url=os.getenv("ALIBABA_BASE_URL"),
    )

    # 创建一个Agent实例
    llm = create_agent(
        model=model,
        tools=[],
        system_prompt="You are an AI assistant.",
    )

    return llm

# 单元测试
if __name__ == '__main__':
    # 获取LLM调用结果
    agent = get_llm_client()
    # 标准入参：字典包裹 messages 列表
    resp = agent.invoke({
        "messages": [("user", "你好，你是谁?")]
    })
    print(resp)