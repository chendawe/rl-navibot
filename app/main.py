import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from app.api.v1.routers.chat import router as chat_router
from app.api.v1.websockets.ros2_channels import (
    drg_ws, telemetry_ws, robot_ws, rl_ws, rgb_ws
)
from perception.slam.drg import DRG
from perception.slam.baselines.maps import make_baseline_grid

from app.core.services.telemetry.robot_service import RobotService
from app.core.services.telemetry.rl_service import RLService
from app.core.services.telemetry.drg_service import DRGService
from app.core.services.telemetry.telemetry_service import TelemetryService

from app.core.services.telemetry.rgb_service import RGBService
# from app.api.v1.websockets.ros2_channels import rgb_websocket

FRONTEND_DIR = Path(__file__).parent / "frontend"

class AppState:
    drg = None
    res = 0.05
    topo_view_data = {"map_w": 0, "map_h": 0, "nodes": [], "edges": []}
    needs_rebuild = False
    services = None

state = AppState()

def build_drg_from_grid(grid, W, H, res):
    state.drg = DRG(grid, resolution=res)
    state.drg.extract()
    state.res = res
    state.topo_view_data = {"map_w": W, "map_h": H, "nodes": [], "edges": state.drg.edges}
    for n in state.drg.nodes:
        state.topo_view_data["nodes"].append({
            "id": n["id"], "x": n["x"] / res, "y": n["y"] / res,
            "w": max(n["span_x"] / res, 3), "h": max(n["span_y"] / res, 3)
        })

try:
    grid, W, H, res = make_baseline_grid()
    build_drg_from_grid(grid, W, H, res)
except Exception as e:
    print(f"⚠️ Baseline 失败: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = None
    env = None
    robot_bridge = None
    rgb_streamer = None  # 🌟 新增：RGB Streamer 句柄

    # 1. 尝试拿 Runtime
    try:
        from core.ros2.master import Ros2Runtime
        
        if not Ros2Runtime._instance:
            print("🌐 检测到 Web 独立启动，主动初始化 Ros2Runtime...")
            # 🌟 这里可能需要根据你实际的 Ros2Runtime 构造函数微调
            # 如果你的 Ros2Runtime 是单例模式且通过 init() 初始化，可能是这样：
            runtime = Ros2Runtime()
            print("✅ Ros2Runtime 自主初始化完成")
        else:
            runtime = Ros2Runtime._instance
            print("✅ 成功挂载已有的 Ros2Runtime")
            
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"⚠️ 获取/初始化 Ros2Runtime 失败: {e}")

    # 2. 尝试实例化 RobotBridge
    if runtime:
        try:
            from core.ros2.channels.bridges.robot import RobotBridge
            robot_bridge = RobotBridge(runtime, node_name='web_robot_bridge')
            robot_bridge.setup(
                laser_topic='/scan', imu_topic='/imu', odom_topic='/odom',
                cmd_vel_topic='/cmd_vel', goal_topic='/goal_pose'
            )
            runtime.register_node(robot_bridge)
            print("✅ 成功实例化并注册 RobotBridge")
        except Exception as e:
            print(f"⚠️ 实例化 RobotBridge 失败: {e}")
            robot_bridge = None

        # 🌟 新增：尝试实例化 RGBStreamer
        try:
            from core.ros2.channels.streamers.video import RGBStreamer
            # rgb_streamer = RGBStreamer(runtime, topic='/camera/image_raw/compressed')
            rgb_streamer = RGBStreamer(runtime, topic='/camera/image_raw')
            runtime.register_node(rgb_streamer) # 别忘了入驻 Runtime 线程池！
            print("✅ 成功实例化并注册 RGBStreamer")
        except Exception as e:
            print(f"⚠️ 实例化 RGBStreamer 失败，视频流将显示黑屏: {e}")
            rgb_streamer = None

    # 3. 尝试获取 RL Env
    if runtime and hasattr(runtime, 'env') and runtime.env:
        env = runtime.env
        print("✅ 成功获取 RL Env")
    else:
        print("⚠️ 未检测到 RL Env")

    # ==========================================
    # 组装所有 Service (传入真实实体或 None)
    # ==========================================
    robot_svc = RobotService(robot_bridge=robot_bridge)
    rl_svc = RLService(env=env)
    rgb_svc = RGBService(streamer=rgb_streamer) # 🌟 新增
    drg_svc = DRGService(
        drg=state.drg, 
        nodes=state.topo_view_data.get("nodes"), 
        edges=state.topo_view_data.get("edges"),
        res=state.res
    )
    telemetry_svc = TelemetryService(robot_service=robot_svc, rl_service=rl_svc)

    class Services: pass
    app.state.services = Services()
    app.state.services.robot_service = robot_svc
    app.state.services.rl_service = rl_svc
    app.state.services.rgb_service = rgb_svc       # 🌟 新增
    app.state.services.drg_service = drg_svc
    app.state.services.telemetry_service = telemetry_svc

    print("🚀 所有 Service 组装完毕")
    yield

app = FastAPI(title="Turtlebo3 Navi CMD", lifespan=lifespan)

app.include_router(chat_router)
app.add_api_websocket_route("/ws/telemetry", telemetry_ws)
app.add_api_websocket_route("/ws/drg", drg_ws)
app.add_api_websocket_route("/ws/robot", robot_ws)
app.add_api_websocket_route("/ws/rl", rl_ws)
app.add_api_websocket_route("/ws/rgb", rgb_ws)

@app.get("/")
async def get_index():
    return FileResponse(FRONTEND_DIR / "public/index.html")

@app.get("/api/topo")
async def get_topo():
    return JSONResponse(content=state.topo_view_data)
