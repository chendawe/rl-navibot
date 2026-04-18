# import asyncio
# import time
# import logging
# from fastapi import WebSocket, WebSocketDisconnect
# from starlette.websockets import WebSocketState

# logger = logging.getLogger(__name__)

# async def ws_handler(websocket: WebSocket):
#     await websocket.accept()
#     path = websocket.scope['path']
#     logger.info(f"[{path}] 连接已建立")

#     # 获取对应的 Service（根据频道调整变量名）
#     service = websocket.app.state.services.map_service   # 替换为具体 service

#     # 独立限流变量（避免全局状态污染）
#     last_send = 0.0
#     send_interval = 0.1  # 根据频道调整：telemetry 0.1s, rgb 0.033s, map 0.1s, depth 0.033s

#     try:
#         while True:
#             # 🌟 关键：只有在连接仍处于 OPEN 状态时才发送
#             if websocket.client_state != WebSocketState.CONNECTED:
#                 logger.warning(f"[{websocket.path}] 连接已非 OPEN 状态，退出循环")
#                 break

#             now = time.time()
#             if now - last_send >= send_interval:
#                 try:
#                     # 获取数据（各频道不同：get_telemetry() / get_frame() 等）
#                     data = service.get_frame()
#                     if data:
#                         await websocket.send_bytes(data)
#                     send_interval = 0.1  # 10 Hz
#                 except Exception as e:
#                     logger.error(f"[{websocket.path}] 获取数据失败: {e}", exc_info=True)
#                     data = None

#                 if data is not None:
#                     try:
#                         # 根据数据类型发送：send_json(data) 或 send_bytes(data)
#                         await websocket.send_json(data)   # 或 send_bytes
#                         last_send = now
#                     except Exception as e:
#                         logger.error(f"[{websocket.path}] 发送数据失败: {e}")
#                         break   # 发送失败则退出循环

#             await asyncio.sleep(0.01)  # 避免 CPU 空转

#     except WebSocketDisconnect:
#         logger.info(f"[{websocket.path}] 客户端正常断开")
#     except Exception as e:
#         logger.error(f"[{websocket.path}] 未知异常: {e}", exc_info=True)
#     finally:
#         logger.info(f"[{websocket.path}] 连接清理完毕")



import asyncio
import time
import logging
import traceback
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    path = websocket.scope['path']
    logger.info(f"[MAP] {path} 连接已建立")

    try:
        map_svc = websocket.app.state.services.map_service
        logger.info(f"[MAP] map_service 获取成功: {map_svc}")
    except Exception as e:
        logger.error(f"[MAP] 获取 map_service 失败: {e}")
        await websocket.close()
        return

    # 强制推送首帧
    try:
        provider = getattr(map_svc, 'provider', None)
        logger.info(f"[MAP] provider 存在: {provider is not None}")
        if provider:
            latest = getattr(provider, '_latest_frame', None)
            logger.info(f"[MAP] _latest_frame 是否为 None: {latest is None}")
            if latest is not None:
                if isinstance(latest, bytes):
                    await websocket.send_bytes(latest)
                    logger.info(f"[MAP] 首帧发送成功，大小: {len(latest)}")
                else:
                    logger.error(f"[MAP] _latest_frame 不是 bytes，类型: {type(latest)}")
            else:
                logger.warning("[MAP] _latest_frame 为 None，跳过首帧发送")
    except Exception as e:
        logger.error(f"[MAP] 首帧发送失败: {e}\n{traceback.format_exc()}")

    last_send = 0.0
    send_interval = 0.1

    try:
        while True:
            # 状态检查
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning(f"[MAP] 连接状态异常 ({websocket.client_state})，退出循环")
                break

            now = time.time()
            if now - last_send >= send_interval:
                try:
                    frame = map_svc.get_frame()
                except Exception as e:
                    logger.error(f"[MAP] get_frame() 抛出异常: {e}\n{traceback.format_exc()}")
                    break  # 发生异常退出循环

                if frame is not None:
                    if not isinstance(frame, bytes):
                        logger.error(f"[MAP] get_frame() 返回非 bytes 类型: {type(frame)}")
                        break
                    try:
                        await websocket.send_bytes(frame)
                        last_send = now
                        logger.debug(f"[MAP] 帧发送成功，大小: {len(frame)}")
                    except Exception as e:
                        logger.error(f"[MAP] send_bytes 失败: {e}")
                        break
                else:
                    logger.debug("[MAP] get_frame() 返回 None")

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"[MAP] {path} 客户端正常断开")
    except Exception as e:
        logger.error(f"[MAP] 主循环未知异常: {e}\n{traceback.format_exc()}")
    finally:
        logger.info(f"[MAP] {path} 连接清理完毕")
        
        
        
        
# 核心机制在于 FastAPI/Uvicorn 的事件循环是共享的，所有 WebSocket 协程都运行在同一个事件循环线程池中。当 Telemetry 的旧连接循环因未检查状态而疯狂抛出异常并立即重试时，会产生以下效应：

# CPU 密集型错误循环：while True 中 send_json 失败后，except 捕获后继续 sleep(0.1) 然后再次尝试发送，形成高频错误日志输出。大量 Cannot call "send" 错误处理占用了事件循环的执行时间。

# 新连接被延迟：刷新页面时，浏览器会同时发起多个 WebSocket 连接请求（Telemetry、RGB、Map、Depth 等）。事件循环需要处理这些新连接的握手和 ws_handler 协程的启动。然而，因为 Telemetry 的错误循环在持续占用 CPU，事件循环调度新协程的时机被推迟，甚至可能因为超时导致连接建立失败。

# 为什么只有 Map 受影响？ 这通常与 连接建立的顺序 和 协程调度公平性 有关。假设前端连接顺序为：Telemetry → RGB → Depth → Map。当刷新时，旧 Telemetry 协程仍在错误循环中，事件循环在尝试启动新 Map 协程之前可能已经花费了大量时间处理其他频道的重连和错误日志。Map 协程的 ws_handler 在 accept() 之后可能执行缓慢，或者由于资源竞争在初始化阶段（如获取 app.state.services）被挂起，最终导致超时或连接被重置。

# 其他频道为何看似正常？ RGB 和 Depth 可能因为连接建立更早，或者它们的 ws_handler 内部逻辑更简单（比如直接发送 bytes，没有 JSON 序列化等额外开销），从而在事件循环被抢占的情况下仍能勉强完成握手和首次数据发送。此外，它们的错误处理可能更早地退出了循环（尽管您之前未修改），减少了错误风暴的持续时间。