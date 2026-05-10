import json
import base64
import cv2
import logging
import os
import random
from typing import Dict, Any, List, Optional

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv

from planning.brain.schemas.brain_schema import BrainParsedResult, BlockPlan
from planning.brain.tools import ros2_tools

logger = logging.getLogger("MockLLM")

# ==========================================
# 环境变量 & 基础配置
# ==========================================
load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

if not ZHIPU_API_KEY:
    raise ValueError("❌ 未找到 ZHIPU_API_KEY，请在项目根目录 .env 文件中配置")

# ==========================================
# 0. 闲聊 LLM (自由对话，不结构化)
# ==========================================
class ChatLLM:
    """
    纯文本闲聊 LLM，负责和用户自然对话。
    使用 glm-4-flash，温度略高以更自然。
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature=0.7,
        )
        self.system = SystemMessage(
            content=(
                "你是一个友好、简洁、运行在Turtlebot3 Waffle上的机器人聊天助手。"
                "与用户展开愉快的对话。"
            )
        )

    def invoke(self, messages: List[BaseMessage]) -> str:
        full = [self.system] + list(messages)
        response = self.llm.invoke(full)
        return response.content


# ==========================================
# 1. 意图路由 LLM (IntentRouterLLM)
# ==========================================
class IntentRouterLLM:
    """
    极快的纯文本意图分类器。
    只返回一个 JSON：{"mission_type": "shutdown" | "mission" | "chat", "goal": ...}
    不做任务拆解，拆解交给 MissionPlannerLLM。
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature=0.0,  # 分类需要确定性
        )
        self.system = SystemMessage(
            content=(
                "你是一个意图分类器。根据用户消息，只返回一个 JSON，不要输出任何其他内容。\n"
                "格式如下：\n"
                '{"mission_type": "shutdown"} —— 用户要求关机/休眠/停止\n'
                '{"mission_type": "mission", "goal": "用户的目标描述"} —— 需要执行物理动作（去某地、拿某物、找东西等）\n'
                '{"mission_type": "chat", "response": "闲聊回复内容"} —— 普通闲聊/提问\n\n'
                "示例：\n"
                '用户: "去桌子上拿水杯" → {"mission_type": "mission", "goal": "去桌子上拿水杯"}\n'
                '用户: "你好" → {"mission_type": "chat"}\n'
                '用户: "关机吧" → {"mission_type": "shutdown"}'
            )
        )

    def invoke(self, messages: List[BaseMessage]) -> dict:
        """
        返回意图字典，例如：
        {"mission_type": "mission", "goal": "去桌子上拿水杯"}
        """
        user_msg = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            "",
        )
        full = [self.system] + list(messages)
        raw = self.llm.invoke(full).content.strip()

        # 解析 LLM 返回的 JSON
        try:
            parsed = json.loads(raw)
            # parsed = raw
        except json.JSONDecodeError:
            # 兜底：解析失败当闲聊处理
            logger.warning(f"[IntentRouterLLM] JSON 解析失败，原文: {raw}")
            return {"mission_type": "chat", "goal": user_msg}

        intent = parsed.get("mission_type", "chat")

        # 如果是 mission，确保 goal 有值
        if intent == "mission":
            goal = parsed.get("goal", user_msg)
            parsed["goal"] = goal

        return parsed


# ==========================================
# 2. 任务拆解 LLM (MissionPlannerLLM，多模态)
# ==========================================
import json
import base64
import cv2
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from planning.brain.schemas.brain_schema import BlockPlan

logger = logging.getLogger("MissionPlannerLLM")

class MissionPlannerLLM:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="glm-4v-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature=0.7,
        )
        self.system = SystemMessage(content=(
            "你是一个机器人战略大脑。根据提供的信息制定或调整行动计划。\n"
            "信息可能包含：目标、拓扑图、机器人状态、历史执行记录、视觉观察记录、当前视角照片等。\n"
            "请仔细阅读【补充信息】中的先验知识（如地图节点描述），这对你规避错误非常重要。\n\n"
            "请输出接下来的执行步骤列表，每个步骤包含 block_type 和 target。\n"
            "block_type 只允许以下三种：\n"
            "- navi：导航到某个位置\n"
            "- observe：观察某个具体的实体，比如桌椅、某个人\n"
            "- standby：原地待命\n\n"
            "如果是无法解决的死局，返回空列表 blocks=[]。\n"
            "如果你没有收到拓扑图信息，说明现在是测试模式，按照正常的可能场景给出一定长度合理的 blocks 即可。"
            "【重要】：对于“小心/注意/避开”等提示，不要单独生成一个 standby block，"
            "而是将这个提示作为后续导航动作的约束（例如，导航时避开该位置）。"
            "如果必须确认障碍物情况，可以生成一个 observe 动作去观察。"
        ))

    def _encode_image(self, img_rgb: np.ndarray) -> str:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def _parse_blocks(self, raw: str) -> List[BlockPlan]:
        try:
            # 清理 LLM 返回的 markdown 代码块
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            return [BlockPlan(**b) for b in json.loads(raw)]
        except Exception as e:
            logger.error(f"解析失败: {e}, 原文: {raw}")
            return []

    # ==========================================
    # 万能规划接口
    # ==========================================
    def plan(
        self, 
        goal: str, 
        topo_image: Optional[np.ndarray] = None, 
        robot_state: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        execution_history: Optional[List[Dict[str, Any]]] = None, 
        current_frame: Optional[np.ndarray] = None
    ) -> List[BlockPlan]:
        """
        goal: 核心目标
        topo_image: 拓扑图 RGB
        robot_state: 小车状态词典 (电量、位置坐标、云台角度等)
        extra: 补充信息 (RAG检索结果、YOLO总结、自定义字段)
        execution_history: 执行与重试历史 (包含 status, detail, retries 等)
        current_frame: 实时第一视角 RGB 图
        """
        content_list = []
        text_prompt = f"【最终目标】: {goal}\n\n"

        # 1. 机器人状态 (键值对直接散开)
        if robot_state:
            state_str = "\n".join([f"- {k}: {v}" for k, v in robot_state.items()])
            text_prompt += f"【机器人当前状态】:\n{state_str}\n\n"

        # 2. 补充信息 (包含 RAG、YOLO 观察等)
        if extra:
            extra_str = ""
            # 针对特殊字段做重点提示
            if "rag_context" in extra:
                extra_str += f"🔺 先验知识库记忆:\n{extra['rag_context']}\n"
            if "yolo_summary" in extra:
                extra_str += f"👁️ 视觉历史观察:\n{extra['yolo_summary']}\n"
            
            # 其他未知字段直接铺开
            for k, v in extra.items():
                if k not in ["rag_context", "yolo_summary"]:
                    extra_str += f"- {k}: {v}\n"
                
            text_prompt += f"【补充信息】:\n{extra_str}\n"

        # 3. 执行与重试历史
        if execution_history:
            hist_str = ""
            for i, h in enumerate(execution_history):
                # 兼容各种字段：block_type, target, status, detail, retries
                action_name = h.get("block_type", "unknown")
                target = h.get("target", "unknown")
                status = h.get("status", "unknown")
                detail = h.get("detail", "")
                retries = h.get("retries", 0)
                
                line = f"步骤 {i+1}: {action_name}({target}) -> {status}"
                if detail:
                    line += f" (原因: {detail})"
                if retries > 0:
                    line += f" [已重试 {retries} 次]"
                hist_str += line + "\n"
                
            text_prompt += f"【执行与重试历史】:\n{hist_str}\n"

        # 4. 图片占位提示词
        if topo_image is not None:
            text_prompt += "【环境拓扑图】: 请结合拓扑图和上述状态规划路线。\n"
        if current_frame is not None:
            text_prompt += "【当前第一视角照片】: 发生了异常，请仔细观察照片判断受阻原因。\n"

        content_list.append({"type": "text", "text": text_prompt.strip()})

        # 5. 插入图片数据
        if topo_image is not None:
            content_list.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(topo_image)}"}
            })
        if current_frame is not None:
            content_list.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(current_frame)}"}
            })

        # 6. 调用 LLM 并解析
        raw = self.llm.invoke([self.system, HumanMessage(content=content_list)]).content.strip()
        print(raw)
        return self._parse_blocks(raw)
    

# ==========================================
# 3. 微观小脑 LLM (ExecutorLLM，多模态点位规划 & 工具执行)
# ==========================================
class ExecutorLLM:
    """
    微观小脑：单步 Block 执行时的多模态 LLM。
    结合当前第一视角 RGB 图，调用底层工具（navi / observe / grab 等）。
    """

    def __init__(self, visioner):
        self.visioner = visioner  # 拿到视觉快照器，用于取图

        # 使用 glm-4v-flash (免费多模态版，支持视觉输入)
        self.llm = ChatOpenAI(
            model="glm-4v-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature=0.2,
        )

        # 底层执行工具注册
        self.tools = {
            "navi": self._mock_navi,      # 后续替换为 ros2_tools.ros_navigate
            "observe": self._mock_observe  # 后续替换为 ros2_tools.ros_observe
        }

    def _encode_image(self, img_rgb: np.ndarray) -> str:
        """将 numpy RGB 图转为 base64，供多模态 LLM 消费"""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def invoke(
        self,
        messages: List[BaseMessage],
        block_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"[ExecutorLLM] 🧠 思考微观任务 -> type={block_type}, target={target}")

        # 1. 获取当前视觉快照
        img_rgb = self.visioner._get_safe_frame()
        base64_img = self._encode_image(img_rgb)

        # 2. 构建多模态 Prompt
        content = [
            {"type": "text", "text": f"当前任务类型: {block_type}，目标: {target}。\n请观察这张图，判断当前环境信息。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        ]
        multimodal_msg = HumanMessage(content=content)

        # 3. 视觉推理
        vision_response = self.llm.invoke([multimodal_msg])
        logger.info(f"[ExecutorLLM] 👁️ 视觉思考结果: {vision_response.content}")

        # 4. 执行底层工具
        tool = self.tools.get(block_type)
        if not tool:
            return {"status": "failed", "detail": f"未知工具类型: {block_type}"}

        result = tool.invoke(target)
        status = result["status"]

        if status == "success":
            logger.info(f"[ExecutorLLM] ✅ 工具返回成功")
        elif status == "failed":
            logger.warning(f"[ExecutorLLM] ❌ 工具返回失败 -> {result['detail']}")
        else:
            logger.error(f"[ExecutorLLM] ⏱️ 工具返回超时 -> {result['detail']}")

        return result

    # ---- 临时 Mock 工具，方便跑通测试 ----
    def _mock_navi(self, target):
        return {"status": "success", "detail": f"已导航至 {target}"}

    def _mock_observe(self, target):
        return {"status": "success", "detail": f"已发现 {target}"}


from enum import Enum
from typing import Optional, Dict, Any
import numpy as np

# Supervisor 的决策指令
class SupervisorAction(str, Enum):
    CONTINUE = "continue"           # 一切正常，继续等待/下一步
    MICRO_ADJUST = "micro_adjust"   # 局部微调（如偏移一点再抓）
    EMERGENCY_STOP = "e_stop"       # 紧急停止（如遇到人、碰撞预警）
    ESCALATE_REPLAN = "replan"      # 局部无法解决，上报给 MissionPlanner 重规划

class SupervisorLLM:
    def __init__(self):
        self.llm = ... # 极低延迟的 VLM (如 glm-4v-flash 甚至更小的端侧模型)
        self.system = "你是一个机器人实时安全督导。根据当前视角和传感器，判断是否安全。..."

    def check(
        self, 
        current_block: BlockPlan, 
        current_frame: np.ndarray, 
        sensor_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        实时监控：在执行动作期间，或动作失败瞬间调用。
        返回：{
            "action": SupervisorAction, 
            "reason": "前方有人", 
            "adjust_params": {"dx": 0.1} # 如果是微调
        }
        """
        # 将当前任务、实时画面、传感器数据喂给 VLM
        # 要求输出结构化判断
        pass




import json
import base64
import cv2
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from planning.brain.schemas.brain_schema import BlockPlan

logger = logging.getLogger("FakeEnvironmentLLM")

import json
import cv2
import base64
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class FakeEnvironmentLLM:
    """
    用 LLM 模拟"机器人环境 + 执行器"。
    结构与 ExecutorLLM 对齐：tools 注册表 + invoke 调用。
    区别：每个 tool 内部用 LLM 模拟真实执行结果，且按格式化输出直接解析 JSON。
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="glm-4v-flash",
            api_key=ZHIPU_API_KEY,      # 替换为你的实际 Key 常量
            base_url=ZHIPU_BASE_URL,    # 替换为你的实际 Base URL 常量
            temperature=0.5,
            # model_kwargs={
            #     "response_format": {"type": "json_object"},
            # },
        ).bind(response_format={"type": "json_object"})

        self.system = SystemMessage(content=(
            "你是一个机器人环境模拟器。你的任务是：给定一个要执行的动作和当前环境状态，"
            "返回这个动作在真实世界中可能的执行结果。\n\n"
            "动作类型可能是：navi（导航到某个位置）、observe（观察某个物体）。\n"
            "你必须严格返回一个 JSON 对象，不要包含任何其他文字或 Markdown 标记，格式如下：\n"
            '{"status": "success" | "failed" | "timeout", "detail": "原因描述"}\n\n'
            "示例：\n"
            "- 动作: navi(门口)，门口被挡住 → "
            '{"status": "failed", "detail": "导航失败：门口被障碍物挡住"}\n'
            "- 动作: observe(水杯)，水杯在视野中 → "
            '{"status": "success", "detail": "观察成功：视野中发现水杯"}\n'
            "如果动作本身就不支持，status 用 failed，detail 说明原因。\n"
            "由于现在是测试模式，大部分情况按可能的日常情况到达目标点即可"
            "【测试模式规则 - 非常重要】"
            "1. 对于 navi 动作，只要目标位置合理（房间/走廊/门口等），一律返回导航成功，detail 类似：导航成功：已移动至 XXX。"
            "2. 对于 observe 动作，只要目标是该位置常见的人或物（如：弟弟、外卖、水、水杯、手机等），"
            "一律返回观察成功，status=success，detail 类似：观察成功：在 XXX 发现了 YYY。"
            "3. 只有在目标明显不合理（如观察恐龙、观察外星人）时，才允许返回 failed。"
            "4. 不要再输出“无法找到 XXX”这种失败结果，除非目标非常离谱。"
            "5. 你现在是在模拟一个友好的测试环境，不是在模拟真实世界里那种经常找不到东西的机器人。"
        ))

        # ✅ 工具注册表，与 ExecutorLLM 同构
        self.tools = {
            "navi": self._fake_navi,
            "observe": self._fake_observe,
            "standby": self._fake_standby,
        }

    # ---- 工具实现：每个 tool 内部用 LLM 模拟 ----

    def _ask_env(self, action_desc: str, current_frame=None, robot_state=None) -> Dict[str, Any]:
        """通用环境模拟：构造 prompt → 调 LLM → 解析结果"""
        text_prompt = f"当前要执行的动作：{action_desc}\n"

        if robot_state:
            state_str = "\n".join([f"- {k}: {v}" for k, v in robot_state.items()])
            text_prompt += f"【机器人当前状态】:\n{state_str}\n"
        else:
            text_prompt += "【机器人当前状态】: 未提供\n"

        if current_frame is not None:
            text_prompt += "【当前视角照片】: 请结合这张图判断执行结果。\n"
        else:
            text_prompt += "【当前视角照片】: 未提供，仅根据文本信息模拟。\n"

        content_list = [{"type": "text", "text": text_prompt.strip()}]
        if current_frame is not None:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(current_frame)}"}
            })

        raw = self.llm.invoke([self.system, HumanMessage(content=content_list)]).content.strip()
        logger.info(f"[FakeEnvironmentLLM] 原始输出: {raw}")

        result = self._parse_result(raw)
        logger.info(f"[FakeEnvironmentLLM] 模拟执行 {action_desc} -> status={result['status']}, detail={result['detail']}")
        return result

    def _fake_navi(self, target: str, **kwargs) -> Dict[str, Any]:
        """模拟导航：问 LLM 能不能到达"""
        return self._ask_env(f"navi({target})", **kwargs)

    def _fake_observe(self, target: str, **kwargs) -> Dict[str, Any]:
        """模拟观察：问 LLM 能不能看到"""
        return self._ask_env(f"observe({target})", **kwargs)

    def _fake_standby(self, target: str, **kwargs) -> Dict[str, Any]:
        """模拟待命：直接返回成功"""
        return {"status": "success", "detail": f"待命成功：原地等待，原因：{target}"}

    # ---- invoke：与 ExecutorLLM 完全对齐 ----

    def invoke(
        self,
        messages: List[BaseMessage],
        block_type: Optional[str] = None,
        target: Optional[str] = None,
        current_frame: Optional[np.ndarray] = None,
        robot_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info(f"[FakeEnvironmentLLM] 模拟执行 -> type={block_type}, target={target}")

        tool = self.tools.get(block_type)
        if not tool:
            return {"status": "failed", "detail": f"未知工具类型: {block_type}"}

        result = tool(target, current_frame=current_frame, robot_state=robot_state)
        status = result["status"]

        if status == "success":
            logger.info(f"[FakeEnvironmentLLM] ✅ 模拟成功")
        elif status == "failed":
            logger.warning(f"[FakeEnvironmentLLM] ❌ 模拟失败 -> {result['detail']}")
        else:
            logger.error(f"[FakeEnvironmentLLM] ⏱️ 模拟超时 -> {result['detail']}")

        return result

    # ---- 工具方法 ----

    def _encode_image(self, img_rgb: np.ndarray) -> str:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def _parse_result(self, raw: str) -> Dict[str, Any]:
        if not raw or not raw.strip():
            logger.error("FakeEnvironmentLLM 返回空字符串")
            return {"status": "failed", "detail": "环境模拟器返回空结果"}

        try:
            result = json.loads(raw)
            result.setdefault("status", "failed")
            result.setdefault("detail", "LLM 未返回有效结果")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"FakeEnvironmentLLM JSON解析失败: {e}, 原文: {raw}")
            return {"status": "failed", "detail": f"环境模拟结果非标准JSON: {e}"}

# ==========================================
# 4. Mock 版本（没网也能跑）
# ==========================================
class MockPlannerLLM:
    """
    宏观大脑 Mock：不依赖网络，用规则解析意图。
    """

    def invoke(self, messages: List[BaseMessage]) -> BrainParsedResult:
        user_msg = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            ""
        )
        logger.info(f"[MockPlannerLLM] 解析意图: {user_msg}")

        if "关机" in user_msg:
            return BrainParsedResult(mission_type="shutdown")

        if "去" in user_msg or "拿" in user_msg:
            return BrainParsedResult(
                mission_type="mission",
                user_goal="拿到水杯",
                mission_blocks=[
                    BlockPlan(block_type="navi", description="导航", target="桌子"),
                    BlockPlan(block_type="observe", description="观察", target="水杯"),
                ],
            )

        return BrainParsedResult(mission_type="chat", response=f"宏观回复：{user_msg}")


class MockExecutorLLM:
    """
    微观小脑 Mock：只调用 ros2_tools，不依赖多模态 LLM。
    """

    def __init__(self):
        # 注册工具
        self.tools = {
            "navi": ros2_tools.ros_navigate,
            "observe": ros2_tools.ros_observe,
        }

    def invoke(
        self,
        messages: List[BaseMessage],
        block_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"[MockExecutorLLM] 📦 调用底层工具 -> type={block_type}, target={target}")

        tool = self.tools.get(block_type)
        if not tool:
            return {"status": "failed", "detail": f"未知工具类型: {block_type}"}

        result = tool.invoke(target)
        status = result["status"]

        if status == "success":
            logger.info(f"[MockExecutorLLM] ✅ 工具返回成功")
        elif status == "failed":
            logger.warning(f"[MockExecutorLLM] ❌ 工具返回失败 -> {result['detail']}")
        else:
            logger.error(f"[MockExecutorLLM] ⏱️ 工具返回超时 -> {result['detail']}")

        return result


# ==========================================
# 5. 工厂函数（优先真实 LLM，可降级为 Mock）
# ==========================================
_use_real_llm = True  # 后期可以改成从环境变量 / 配置文件读取


def get_intent_router_llm():
    if _use_real_llm:
        return IntentRouterLLM()
    else:
        return NotImplementedError("IntentRouterLLM() 目前只有真实 LLM 版本，没有 Mock")


def get_executor_llm(visioner=None):
    if _use_real_llm and visioner is not None:
        return ExecutorLLM(visioner=visioner)
    elif _use_real_llm :
        return FakeEnvironmentLLM()
    else:
        return MockExecutorLLM()


def get_chat_llm():
    if _use_real_llm:
        return ChatLLM()
    else:
        return MockPlannerLLM()


def get_mission_planner_llm():
    if _use_real_llm:
        return MissionPlannerLLM()
    else:
        raise MockPlannerLLM()