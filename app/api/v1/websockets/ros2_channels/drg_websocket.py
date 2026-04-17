import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    drg_svc = websocket.app.state.services.drg_service
    
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            
            if msg.get("type") in ["plan", "execute"]:
                payload = msg.get("payload", msg)
                start_id = payload.get("start")
                end_id = payload.get("end")
                
                # 如果前端没传起点，默认取第一个节点
                if not start_id and drg_svc.nodes:
                    start_id = drg_svc.nodes[0]['id']
                if not end_id and drg_svc.nodes:
                    end_id = drg_svc.nodes[-1]['id']
                    
                if not start_id or not end_id:
                    await websocket.send_json({"type": "done", "msg": "❌ 缺少起点或终点"})
                    continue

                result = drg_svc.plan_path(start_id, end_id)
                if not result["success"]:
                    await websocket.send_json({"type": "done", "msg": result["msg"]})
                    continue
                    
                plan_path = result["path"]
                await websocket.send_json({"type": "path", "nodes": plan_path})
                await asyncio.sleep(0.5)
                
                for frame in drg_svc.generate_move_frames(plan_path):
                    await websocket.send_json({"type": "pose", "x": frame["x"], "y": frame["y"]})
                    await asyncio.sleep(frame["sleep"])
                    
                await websocket.send_json({"type": "done", "msg": f"✅ 到达 {end_id}"})
    except WebSocketDisconnect:
        pass
