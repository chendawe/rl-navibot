import asyncio
from fastapi import WebSocket, WebSocketDisconnect

import logging
logger = logging.getLogger(__name__)

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    # 从 app.state 获取注入的实例，如果没有，说明 main.py 兜底逻辑失效了
    telemetry_svc = websocket.app.state.services.telemetry_service
    
    try:
        while True:
            try:
                data = telemetry_svc.get_telemetry()
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"打包遥测数据失败: {e}")
                # 可以选择发个错误状态给前端，或者直接跳过这帧
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass

