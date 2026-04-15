import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi import WebSocket
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent / "frontend"

from app.api.v1.chat import router as chat_router
from app.api.v1.ws import handle_websocket
from app.core.ros2_bridge import Ros2Bridge       # ★ 导入桥接器
from perception.slam.drg import DRG
from perception.slam.baselines.maps import make_baseline_grid
from perception.slam.utils import ros_occ_to_numpy # ★ 之前的转换函数

app = FastAPI(title="Turtlebo3 Navi CMD")
app.include_router(chat_router)

# ==========================================
# 全局状态
# ==========================================
drg = None
topo_view_data = {"map_w": 0, "map_h": 0, "nodes": [], "edges": []}
needs_rebuild = False  # ★ 标记位：ROS2 来了新地图吗？

def build_drg_from_grid(grid, W, H, res):
    global drg, topo_view_data
    drg = DRG(grid, resolution=res)
    drg.extract()
    topo_view_data = {"map_w": W, "map_h": H, "nodes": [], "edges": drg.edges}
    for n in drg.nodes:
        topo_view_data["nodes"].append({
            "id": n["id"], "x": n["x"] / res, "y": n["y"] / res,
            "w": max(n["span_x"] / res, 3), "h": max(n["span_y"] / res, 3)
        })
    print("✅ DRG 拓扑构建完成！")

# Baseline 兜底
try:
    grid, W, H, res = make_baseline_grid()
    build_drg_from_grid(grid, W, H, res)
except Exception as e:
    print(f"Baseline 失败: {e}")

# ==========================================
# 启动事件：使用 Ros2Bridge
# ==========================================
@app.on_event("startup")
async def startup_event():
    from nav_msgs.msg import OccupancyGrid
    
    bridge = Ros2Bridge(node_name="web_topo_backend")
    
    # ★ 一行代码搞定订阅，极度清爽
    bridge.create_subscriber(
        topic='/map', 
        msg_type=OccupancyGrid,
        preprocess_cb=lambda msg: setattr(startup_event, '_map_flag', True) # 收到地图打个标记
    )
    
    # 把 bridge 存到 app.state 里，以后别的路由要用直接拿
    app.state.ros_bridge = bridge

# ==========================================
# 路由
# ==========================================
@app.get("/")
async def get_index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api/topo")
async def get_topo():
    global needs_rebuild
    bridge = app.state.ros_bridge
    
    # 检查是否有新地图
    if needs_rebuild or hasattr(startup_event, '_map_flag'):
        map_msg = bridge.get_data('/map')
        if map_msg is not None:
            print("🔄 收到 ROS2 /map，重建拓扑...")
            grid, W, H, res = ros_occ_to_numpy(map_msg)
            build_drg_from_grid(grid, W, H, res)
            needs_rebuild = False
            if hasattr(startup_event, '_map_flag'):
                delattr(startup_event, '_map_flag')
                
    return JSONResponse(content=topo_view_data)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, drg)
