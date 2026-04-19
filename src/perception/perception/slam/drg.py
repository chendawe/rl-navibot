import numpy as np
import matplotlib
matplotlib.use('Agg') # ROS2 节点无头模式必须加
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.graph import pixel_graph
import math
import heapq

# ================= 自定义异常 =================
class TopologyError(Exception):
    """拓扑构建基础异常"""
    pass

class PathNotFoundError(TopologyError):
    """寻路失败异常"""
    pass

class DRG:
    ROBOT_DIAMETERS = {
        'burger': 0.28,
        'waffle': 0.34,
        'waffle_pi': 0.34
    }

    def __init__(self, grid: np.ndarray, resolution: float = 0.05,
                 robot_name: str = None, merge_threshold: int = None,
                 target_ratio: float = 0.75):
        """
        :param grid: 二值化栅格图
        :param resolution: 米/像素
        :param robot_name: 机器人名称，用于自动计算合并阈值（如 'burger'）
        :param merge_threshold: 手动指定合并阈值（像素），优先级高于 robot_name
        :param target_ratio: 自动计算时，目标节点间距占机器人直径的比例（0~1）
        """
        self.grid = grid
        self.res = resolution
        self.H, self.W = grid.shape

        # 计算合并阈值
        if merge_threshold is not None:
            self.merge_threshold = merge_threshold
        elif robot_name is not None:
            diameter = self.ROBOT_DIAMETERS.get(robot_name.lower())
            if diameter is None:
                raise ValueError(f"未知机器人名称 '{robot_name}'，支持: {list(self.ROBOT_DIAMETERS.keys())}")
            target_spacing = diameter * target_ratio
            self.merge_threshold = max(1, int(target_spacing / self.res))
        else:
            self.merge_threshold = 5  # 默认值

        # 内部状态初始化（不变）
        self._skel_set = set()
        self._px_graph = None
        self._degrees = None
        self._coord_to_idx = {}
        self._node_xy_coords = []
        self.nodes = []
        self.edges = []
        self._nav_graph = {}
        self._node_pos = {}

    def extract(self, merge_threshold: int = 5):
        """执行完整的 DRG 提取流水线"""
        self._preprocess()
        self._extract_key_points()
        self._trace_and_merge_edges(merge_threshold)
        self._build_nav_graph()
        return self.nodes, self.edges

    # ==========================================
    # 私有方法：流水线步骤
    # ==========================================
    def _preprocess(self):
        free = (self.grid == 255).astype(np.uint8)
        skeleton = skeletonize(free).astype(np.uint8)
        skel_bool = skeleton.astype(bool)
        
        skel_y, skel_x = np.where(skeleton > 0)
        self._skel_set = set(zip(skel_x.tolist(), skel_y.tolist()))
        
        self._px_graph, flat_indices = pixel_graph(skel_bool, connectivity=2, edge_function=lambda x, y, d: d)
        self._degrees = np.diff(self._px_graph.indptr)
        
        for i, flat_idx in enumerate(flat_indices):
            y, x = np.unravel_index(flat_idx, skel_bool.shape)
            self._coord_to_idx[(x, y)] = i
            self._node_xy_coords.append((y, x))

    def _extract_key_points(self):
        endpoint_indices = np.where(self._degrees == 1)[0]
        junction_indices = np.where(self._degrees >= 3)[0]
        
        raw_key_points = set()
        for i in np.concatenate([endpoint_indices, junction_indices]):
            y, x = self._node_xy_coords[i]
            raw_key_points.add((x, y))
            
        # 提取拐弯点
        raw_key_points.update(self._get_bend_points())
        
        self._raw_key_indices = set()
        for x, y in raw_key_points:
            if (x, y) in self._coord_to_idx:
                self._raw_key_indices.add(self._coord_to_idx[(x, y)])

    def _get_bend_points(self):
        """通过向量夹角提取拐弯点"""
        def get_neighbors(x, y, s_set):
            return [(x+dx, y+dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                    if not (dx == 0 and dy == 0) and (x+dx, y+dy) in s_set]

        def walk_straight(sx, sy, prev_x, prev_y, steps, s_set):
            path, cx, cy = [], sx, sy
            for _ in range(steps):
                neis = [n for n in get_neighbors(cx, cy, s_set) if n != (prev_x, prev_y)]
                if len(neis) != 1: break 
                path.append(neis[0])
                prev_x, prev_y = cx, cy
                cx, cy = neis[0]
            return path

        bend_points = set()
        ANGLE_THRESHOLD = 20
        MAX_STEPS = 10

        for x, y in self._skel_set:
            neis = get_neighbors(x, y, self._skel_set)
            if len(neis) != 2: continue 
            
            n1, n2 = neis
            path1 = walk_straight(n1[0], n1[1], x, y, MAX_STEPS, self._skel_set)
            path2 = walk_straight(n2[0], n2[1], x, y, MAX_STEPS, self._skel_set)
            
            if path1 and path2:
                v1 = np.array(path1[-1]) - np.array([x, y])
                v2 = np.array(path2[-1]) - np.array([x, y])
                n1_norm, n2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
                
                if n1_norm > 0 and n2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (n1_norm * n2_norm)
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    if angle < (180 - ANGLE_THRESHOLD):
                        bend_points.add((x, y))
        return bend_points

    def _trace_and_merge_edges(self, merge_threshold):
        # 1. 追踪边
        edges_set = set()
        for k_idx in self._raw_key_indices:
            for t_idx in self._trace_path(k_idx):
                edges_set.add(tuple(sorted([k_idx, t_idx])))

        # 2. 合并节点
        current_indices, final_edges = self._merge_nodes(self._raw_key_indices, edges_set, merge_threshold)

        # 3. 格式化输出
        idx = 0
        node_id_map = {}
        for k_idx in current_indices:
            y, x = self._node_xy_coords[k_idx]
            sx, sy = self._calc_local_span(x, y)
            node_id = f"N_{idx}"
            self.nodes.append({"id": node_id, "x": round(x * self.res, 2), "y": round(y * self.res, 2), "span_x": sx, "span_y": sy})
            node_id_map[k_idx] = node_id
            idx += 1

        for edge in final_edges:
            if edge[0] in node_id_map and edge[1] in node_id_map:
                self.edges.append({"from": node_id_map[edge[0]], "to": node_id_map[edge[1]]})

    def _trace_path(self, start_idx):
        neighbors = self._px_graph[start_idx].indices
        connected_nodes = []
        for n_idx in neighbors:
            if n_idx in self._raw_key_indices:
                connected_nodes.append(n_idx)
            else:
                prev_idx, curr_idx, steps = start_idx, n_idx, 0
                while self._degrees[curr_idx] == 2 and curr_idx not in self._raw_key_indices and steps < 200:
                    next_indices = self._px_graph[curr_idx].indices
                    if len(next_indices) == 1: break
                    next_idx = next_indices[0] if next_indices[1] == prev_idx else next_indices[1]
                    prev_idx, curr_idx, steps = curr_idx, next_idx, steps + 1
                connected_nodes.append(curr_idx if curr_idx in self._raw_key_indices or self._degrees[curr_idx] == 1 else None)
        return [n for n in connected_nodes if n is not None]

    def _merge_nodes(self, indices, edges, threshold):
        current_indices = set(indices)
        merged_map = {}
        points = np.array([self._node_xy_coords[i] for i in current_indices])
        indices_list = list(current_indices)
        
        for i in range(len(indices_list)):
            for j in range(i+1, len(indices_list)):
                idx1, idx2 = indices_list[i], indices_list[j]
                if idx1 not in current_indices or idx2 not in current_indices: continue
                y1, x1 = self._node_xy_coords[idx1]
                y2, x2 = self._node_xy_coords[idx2]
                if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < threshold:
                    merged_map[idx2] = idx1
                    current_indices.remove(idx2)
                    
        final_edges = set()
        for e in edges:
            e = (merged_map.get(e[0], e[0]), merged_map.get(e[1], e[1]))
            if e[0] in current_indices and e[1] in current_indices:
                final_edges.add(e)
        return current_indices, final_edges

    def _calc_local_span(self, x, y, radius=30):
        local_pts = [pt for pt in self._skel_set if abs(pt[0]-x)<=radius and abs(pt[1]-y)<=radius]
        if not local_pts: return 0.0, 0.0
        pts = np.array(local_pts)
        return round(float((np.max(pts[:, 0]) - np.min(pts[:, 0])) * self.res), 2), \
               round(float((np.max(pts[:, 1]) - np.min(pts[:, 1])) * self.res), 2)

    def _build_nav_graph(self):
        self._nav_graph = {n['id']: [] for n in self.nodes}
        self._node_pos = {n['id']: (n['x'], n['y']) for n in self.nodes}
        for e in self.edges:
            n1, n2 = e['from'], e['to']
            dist = math.hypot(self._node_pos[n2][0]-self._node_pos[n1][0], self._node_pos[n2][1]-self._node_pos[n1][1])
            self._nav_graph[n1].append((n2, dist))
            self._nav_graph[n2].append((n1, dist))

    # ==========================================
    # 公共接口：寻路与输出
    # ==========================================
    def plan(self, start_id: str, end_id: str) -> tuple:
        """
        A* 寻路接口
        :return: (path_list, total_distance)
        :raises PathNotFoundError: 节点不存在或不可达
        """
        if start_id not in self._nav_graph or end_id not in self._nav_graph:
            raise PathNotFoundError(f"节点不存在: {start_id} 或 {end_id}")
        if start_id == end_id:
            return [start_id], 0.0

        open_list = [(0, 0, start_id)]
        counter, g_costs, parents, closed_list = 1, {start_id: 0.0}, {start_id: None}, set()
        
        def heuristic(nid):
            return math.hypot(self._node_pos[end_id][0]-self._node_pos[nid][0], 
                              self._node_pos[end_id][1]-self._node_pos[nid][1])

        while open_list:
            _, _, curr_id = heapq.heappop(open_list)
            if curr_id == end_id:
                path = []
                while curr_id is not None: path.append(curr_id); curr_id = parents[curr_id]
                return path[::-1], g_costs[end_id]
            if curr_id in closed_list: continue
            closed_list.add(curr_id)
            
            for neighbor_id, edge_dist in self._nav_graph[curr_id]:
                if neighbor_id in closed_list: continue
                tentative_g = g_costs[curr_id] + edge_dist
                if tentative_g < g_costs.get(neighbor_id, float('inf')):
                    parents[neighbor_id] = curr_id
                    g_costs[neighbor_id] = tentative_g
                    heapq.heappush(open_list, (tentative_g + heuristic(neighbor_id), counter, neighbor_id))
                    counter += 1
                    
        raise PathNotFoundError(f"无法从 {start_id} 到达 {end_id}，拓扑不连通")

    def get_topology_dict(self) -> dict:
        """获取标准拓扑字典"""
        return {"nodes": self.nodes, "edges": self.edges}

    def visualize(self, save_path: str = "drg_result.png", highlight_path: list = None):
        """可视化并保存到本地 (ROS2 底层模块不建议直接返回 base64)"""
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(self.grid, cmap='gray', origin='upper', alpha=0.5)
        
        node_dict = {n["id"]: n for n in self.nodes}
        for n in self.nodes:
            px, py = n['x'] / self.res, n['y'] / self.res
            rect = patches.Rectangle((px - n['span_x']/2, py - n['span_y']/2), n['span_x'], n['span_y'],
                                     linewidth=1, edgecolor='black', facecolor='red', alpha=0.8)
            ax.add_patch(rect)
            ax.annotate(n['id'], (px, py), textcoords="offset points", xytext=(5, 5), fontsize=6, color='darkred')

        for e in self.edges:
            n1, n2 = node_dict[e["from"]], node_dict[e["to"]]
            ax.plot([n1['x']/self.res, n2['x']/self.res], [n1['y']/self.res, n2['y']/self.res], 'g-', linewidth=1.5)

        if highlight_path and len(highlight_path) > 1:
            for i in range(len(highlight_path)-1):
                n1, n2 = node_dict[highlight_path[i]], node_dict[highlight_path[i+1]]
                ax.plot([n1['x']/self.res, n2['x']/self.res], [n1['y']/self.res, n2['y']/self.res], 'b-', linewidth=3)

        ax.set_title("DRG Topology Map"); ax.set_xlim(0, self.W); ax.set_ylim(self.H, 0)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
