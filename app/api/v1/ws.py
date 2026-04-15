import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

import math

from perception.slam.drg import PathNotFoundError

async def handle_websocket(websocket: WebSocket, drg):
    await websocket.accept()
    
    try:
        nodes = drg.nodes
        res = drg.res  # ★ 获取分辨率

        # ★ 修改：必须除以 res 转换成像素坐标，和前端 /api/topo 保持绝对一致！
        minX = min(n['x'] / res for n in nodes)
        maxX = max(n['x'] / res for n in nodes)
        minY = min(n['y'] / res for n in nodes)
        maxY = max(n['y'] / res for n in nodes)
        
        rangeX = maxX - minX if maxX != minX else 1
        rangeY = maxY - minY if maxY != minY else 1
        targetSize = 8 

        def get_canvas_coord(n_id):
            node = next(n for n in nodes if n['id'] == n_id)
            # ★ 修改：同样除以 res
            px_x = node['x'] / res
            px_y = node['y'] / res
            
            norm_x = ((px_x - minX) / rangeX) * targetSize + 1
            norm_y = ((px_y - minY) / rangeY) * targetSize + 1
            return (norm_x, norm_y)

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            
            if msg.get("type") == "execute":
                payload = msg.get("payload")
                end_id = payload.get("end", nodes[0]['id'])
                start_id = payload.get("start", nodes[0]['id']) 
                
                try:
                    plan_path, total_dist = drg.plan(start_id, end_id)
                except PathNotFoundError as e:
                    await websocket.send_json({"type": "done", "msg": f"❌ 寻路失败: {str(e)}"})
                    continue
                
                await websocket.send_json({"type": "path", "nodes": plan_path})
                await asyncio.sleep(0.5)
                
                # 3. 模拟小车平滑移动 (★ 根据物理距离实现匀速)
                TARGET_SPEED = 10.0  # ★ 设定小车的期望速度，单位：米/秒 (改成 5.0 就飞快，改成 0.5 就很慢)

                for i in range(len(plan_path) - 1):
                    n1 = get_canvas_coord(plan_path[i])
                    n2 = get_canvas_coord(plan_path[i+1])
                    
                    # 算这一段的真实物理距离 (米)
                    node1 = next(n for n in nodes if n['id'] == plan_path[i])
                    node2 = next(n for n in nodes if n['id'] == plan_path[i+1])
                    real_dist = math.hypot(node2['x'] - node1['x'], node2['y'] - node1['y'])
                    
                    # 根据距离算需要多少步 (保底10步，防卡顿)
                    steps = max(10, int(real_dist / TARGET_SPEED * 50))
                    # 根据距离算每步睡多久
                    sleep_time = (real_dist / TARGET_SPEED) / steps
                    
                    for s in range(steps):
                        ratio = s / steps
                        curr_x = n1[0] + (n2[0] - n1[0]) * ratio
                        curr_y = n1[1] + (n2[1] - n1[1]) * ratio
                        await websocket.send_json({"type": "pose", "x": curr_x, "y": curr_y})
                        await asyncio.sleep(sleep_time)

                        
                done_msg = f"✅ 任务执行完毕。\n已通过 A* 规划路径 {plan_path}，行驶距离 {total_dist:.2f} 米，到达 {end_id}。"
                await websocket.send_json({"type": "done", "msg": done_msg})
                
    except WebSocketDisconnect:
        pass
