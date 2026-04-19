import math
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class DRGService:
    def __init__(self, drg=None, nodes=None, edges=None, res=0.05):
        self.drg = drg
        self.nodes = nodes if nodes is not None else (drg.nodes if drg else [])
        self.edges = edges if edges is not None else (drg.edges if drg else [])
        self.res = res
        self.canvas_coords = {}
        
        if self.nodes:
            self._init_canvas_mapping()

    def _init_canvas_mapping(self):
        if not self.nodes: return
        minX = min(n['x'] / self.res for n in self.nodes)
        maxX = max(n['x'] / self.res for n in self.nodes)
        minY = min(n['y'] / self.res for n in self.nodes)
        maxY = max(n['y'] / self.res for n in self.nodes)
        
        self.rangeX = maxX - minX if maxX != minX else 1
        self.rangeY = maxY - minY if maxY != minY else 1
        self.minX, self.minY = minX, minY

        for n in self.nodes:
            px_x, px_y = n['x'] / self.res, n['y'] / self.res
            self.canvas_coords[n['id']] = (
                ((px_x - self.minX) / self.rangeX) * 8 + 1,
                ((px_y - self.minY) / self.rangeY) * 8 + 1
            )

    def plan_path(self, start_id: str, end_id: str) -> Dict[str, Any]:
        if not self.drg:
            return {"success": False, "msg": "❌ DRG 实体未注入 (当前为 Mock 状态)"}
        try:
            result = self.drg.plan(start_id, end_id)
            if isinstance(result, tuple): plan_path, total_dist = result
            else: plan_path, total_dist = result, 0.0
            return {"success": True, "path": plan_path, "total_dist": total_dist}
        except Exception as e:
            return {"success": False, "msg": f"❌ 寻路失败: {str(e)}"}

    def generate_move_frames(self, path: List[str], target_speed=10.0):
        if not self.nodes or not self.canvas_coords: return
        for i in range(len(path) - 1):
            n1 = self.canvas_coords.get(path[i])
            n2 = self.canvas_coords.get(path[i+1])
            if not n1 or not n2: continue
            
            node1 = next((n for n in self.nodes if n['id'] == path[i]), None)
            node2 = next((n for n in self.nodes if n['id'] == path[i+1]), None)
            if not node1 or not node2: continue

            real_dist = math.hypot(node2['x'] - node1['x'], node2['y'] - node1['y'])
            steps = max(10, int(real_dist / target_speed * 50))
            sleep_time = (real_dist / target_speed) / steps
            
            for s in range(steps):
                ratio = s / steps
                yield {
                    "x": n1[0] + (n2[0] - n1[0]) * ratio,
                    "y": n1[1] + (n2[1] - n1[1]) * ratio,
                    "sleep": sleep_time
                }


