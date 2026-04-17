import asyncio
from fastapi import WebSocket, WebSocketDisconnect

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    # 从 app.state 获取注入的实例，如果没有，说明 main.py 兜底逻辑失效了
    telemetry_svc = websocket.app.state.services.telemetry_service
    
    try:
        while True:
            data = telemetry_svc.get_telemetry()
            await websocket.send_json(data)
            await asyncio.sleep(0.1)  # 严格 10Hz
    except WebSocketDisconnect:
        pass
