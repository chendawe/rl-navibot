import asyncio
from fastapi import WebSocket, WebSocketDisconnect

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    # 🌟 正确获取 depth_service
    depth_svc = websocket.app.state.services.depth_service
    try:
        while True:
            frame = depth_svc.get_frame()
            if frame is not None:
                await websocket.send_bytes(frame)
            await asyncio.sleep(0.03)  # 30fps 足够
    except WebSocketDisconnect:
        pass