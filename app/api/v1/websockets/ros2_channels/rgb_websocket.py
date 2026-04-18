import asyncio
from fastapi import WebSocket, WebSocketDisconnect
import base64

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    rgb_svc = websocket.app.state.services.rgb_service
    try:
        while True:
            frame = rgb_svc.get_frame()
            if frame is not None:
                # 直接发送二进制 JPEG 数据
                await websocket.send_bytes(frame)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass

# async def ws_handler(websocket: WebSocket):
#     await websocket.accept()
#     rgb_svc = websocket.app.state.services.rgb_service
    
#     try:
#         while True:
#             frame = rgb_svc.get_frame()
#             # 只有当 Service 返回真实新帧或黑屏兜底时才发送，避免死循环空转占 CPU
#             if frame is not None:
#                 frame_base64 = base64.b64encode(frame).decode('utf-8')
#                 await websocket.send_json({"frame": frame_base64})
#                 # await websocket.send_bytes(frame_base64)
            
#             # 约 20Hz 轮询。配合 RGBStreamer 内部的 15fps 限流，
#             # 实际推送频率会被自动钳制在 15fps 左右，且绝不重复推相同帧。
#             await asyncio.sleep(0.05) 
            
#     except WebSocketDisconnect:
#         pass
