import asyncio
import asyncpg

async def test():
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5433,              # 🔥 FIXED
        user="postgres",
        password="YOUR_PASSWORD",
        database="enterprise_rag",
        ssl=False,
    )
    print("CONNECTED")
    await conn.close()

asyncio.run(test())
