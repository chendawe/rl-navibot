import asyncio
import time
import logging
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    path = websocket.scope.get("path", "/ws/telemetry")
    logger.info(f"[TEL] {path} 连接已建立")

    # 获取 Service 实例（带防御）
    try:
        telemetry_svc = websocket.app.state.services.telemetry_service
    except AttributeError:
        logger.error("[TEL] 无法获取 telemetry_service，关闭连接")
        await websocket.close()
        return

    last_send = 0.0
    send_interval = 0.1  # 10 Hz

    try:
        while True:
            # 🌟 关键：状态检查，避免向已关闭连接发送数据
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning(f"[TEL] 连接状态异常 ({websocket.client_state})，退出循环")
                break

            now = time.time()
            if now - last_send >= send_interval:
                try:
                    data = telemetry_svc.get_telemetry()
                except Exception as e:
                    logger.error(f"[TEL] get_telemetry() 异常: {e}", exc_info=True)
                    data = None

                if data is not None:
                    try:
                        await websocket.send_json(data)
                        last_send = now
                    except Exception as e:
                        logger.error(f"[TEL] send_json 失败: {e}")
                        break  # 发送失败说明连接已不可用，退出循环
                else:
                    logger.debug("[TEL] get_telemetry() 返回 None")

            await asyncio.sleep(0.01)  # 小睡片刻，避免空转

    except WebSocketDisconnect:
        logger.info(f"[TEL] {path} 客户端正常断开")
    except Exception as e:
        logger.error(f"[TEL] 主循环未知异常: {e}", exc_info=True)
    finally:
        logger.info(f"[TEL] {path} 连接清理完毕")