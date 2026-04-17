import asyncio
from fastapi import WebSocket, WebSocketDisconnect

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    robot_svc = websocket.app.state.services.robot_service
    
    try:
        while True:
            await websocket.send_json({
                "robot": robot_svc.get_odom_data(),
                "sensor": robot_svc.get_sensor_data()
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
