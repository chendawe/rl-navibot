import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    rl_svc = websocket.app.state.services.rl_service
    
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")
            
            if action is not None:
                state = rl_svc.step_action(action)
                await websocket.send_json(state)
    except WebSocketDisconnect:
        pass
