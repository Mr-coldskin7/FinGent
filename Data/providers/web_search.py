import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

bocha_api_key = os.getenv("BO_CHA_API_KEY")


async def search(
    query: str, freshness: str = "noLimit", summary: bool = True, count: int = 5
) -> dict:
    """带完整参数的请求示例"""
    headers = {
        "Authorization": f"Bearer {bocha_api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "query": query,
        "freshness": freshness,
        "summary": summary,
        "count": count,
    }

    async with httpx.AsyncClient(
        headers=headers,  # 全局 headers，每个请求自动带
        timeout=30.0,  # 总超时 30 秒
        follow_redirects=True,  # 自动跟随重定向
    ) as client:
        response = await client.post(
            "https://api.bocha.cn/v1/web-search",
            json=params,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":

    async def main():
        result = await search("深圳")
        print(result)

    asyncio.run(main())
