import asyncio
import json
from app.core.topology import TOPO_MAP

async def llm_stream_think(prompt: str):
    """纯逻辑：根据 prompt 吐出思考过程和 JSON"""
    text = f"1. 收到指令：「{prompt}」\n2. 正在匹配拓扑...\n"
    target = "N_3" if "厨房" in prompt else "N_4"
    text += f"3. 匹配成功：起点 N_0，目标 {target}\n4. 指令封装完毕。"
    
    for c in text:
        yield c
        await asyncio.sleep(0.02)
    yield f"\n\n[DATA_JSON]:{{\"start\": \"N_0\", \"end\": \"{target}\"}}"
