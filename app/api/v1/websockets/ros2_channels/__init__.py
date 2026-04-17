# 显式导入每个模块的 ws_handler，并用 as 起个别名
# 这样外面拿到的就全是纯函数，彻底消除模块与函数的歧义
from .drg_websocket import ws_handler as drg_ws
from .telemetry_websocket import ws_handler as telemetry_ws
from .robot_websocket import ws_handler as robot_ws
from .rl_websocket import ws_handler as rl_ws
from .rgb_websocket import ws_handler as rgb_ws
