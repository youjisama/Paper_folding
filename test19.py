import sys
import igl
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random


# 美观化的一版，功能基本实现，但是现实哪一边处于被折叠状态，目前没有很好的点子
class OrigamiWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OrigamiWidget, self).__init__(parent)
        self.mode = "view"
        self.points = []
        self.selected_point = None
        self.camera_distance = 11.0
        self.lastPos = None
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0

        # 折叠功能
        self.fold_line_points = []
        self.fold_region = None
        self.fold_angle = 0
        self.fold_direction = 1
        self.fold_point_selection = 0

        # 添加颜色和动画相关属性
        self.colors = [
            [1.0, 0.7, 0.7],  # 粉红色
            [0.7, 1.0, 0.7],  # 浅绿色
            [0.7, 0.7, 1.0],  # 浅蓝色
            [1.0, 1.0, 0.7],  # 黄色
        ]
        self.current_color = 0
        # 移除自动颜色变化的计时器
        # self.animation_timer = QTimer()
        # self.animation_timer.timeout.connect(self.animate_colors)
        # self.animation_timer.start(2000)  # 每2秒更新一次颜色

        # 移除不需要的高亮颜色属性
        # self.highlight_color = [1.0, 0.9, 0.4]  # 高亮颜色（淡黄色）
        # self.highlight_intensity = 0.3  # 高亮强度

        # 初始化矩形模型
        self.V, self.F = self.init_refined_rectangle(n=20)  # 降低密度使其更容易操作
        self.points = self.V.tolist()
        self.boundary_vertices = igl.boundary_loop(self.F)

        self.vao = None
        self.vbo = None
        self.ibo = None

    def animate_colors(self):
        """动画效果：循环切换颜色"""
        self.current_color = (self.current_color + 1) % len(self.colors)
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 设置更友好的背景色
        glClearColor(0.95, 0.95, 1.0, 1.0)  # 淡蓝色背景

        # 设置相机
        glTranslatef(0.0, 0.0, -self.camera_distance)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)

        # 更新顶点数据（如果需要）
        if self.fold_region is not None:
            vertices = np.array(self.points, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        # 绘制网格
        if self.vao is not None:
            glBindVertexArray(self.vao)

            # 使用当前选择的颜色
            current_rgb = self.colors[self.current_color]
            glColor3f(*current_rgb)
            glDrawElements(GL_TRIANGLES, len(self.F.flatten()), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

        # 绘制折叠线和控制元素
        self.draw_fold_elements()

    def draw_fold_elements(self):
        """优化的折线元素绘制"""
        if not self.fold_line_points:  # 如果没有折线点，直接返回
            return

        # 使用顶点数组一次性绘制所有点
        if len(self.fold_line_points) >= 1:
            glDisable(GL_LIGHTING)  # 临时禁用光照以提高性能

            # 绘制第一个点
            glColor3f(1.0, 0.0, 0.0)  # 红色
            glPointSize(10.0)
            glBegin(GL_POINTS)
            glVertex3fv(self.fold_line_points[0])
            glEnd()

            # 如果有第二个点，一起绘制点和线
            if len(self.fold_line_points) == 2:
                # 绘制第二个点
                glColor3f(1.0, 1.0, 0.0)  # 黄色
                glBegin(GL_POINTS)
                glVertex3fv(self.fold_line_points[1])
                glEnd()

                # 绘制连接线
                glColor3f(0.5, 0.5, 0.5)  # 灰色
                glLineWidth(0.1)
                glBegin(GL_LINES)
                glVertex3fv(self.fold_line_points[0])
                glVertex3fv(self.fold_line_points[1])
                glEnd()

                # 如果需要绘制折叠方向箭头
                if self.fold_point_selection == 2:
                    self.draw_fold_direction_arrow()

            glEnable(GL_LIGHTING)  # 重新启用光照

    def draw_fold_direction_arrow(self):
        """绘制手形图标来指示折叠方向"""
        if len(self.fold_line_points) != 2:
            return

        # 计算手形图标位置
        start = np.array(self.fold_line_points[0])
        end = np.array(self.fold_line_points[1])
        mid = (start + end) / 2
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:  # 避免除以零
            return

        # 计算手形图标方向
        direction = direction / length
        normal = np.array([0, 0, 1])
        hand_direction = np.cross(direction, normal)
        if self.fold_direction < 0:
            hand_direction = -hand_direction

        # 将手形图标放置在折线一侧，稍微远离折线
        offset_distance = 0.5  # 距离折线的距离
        hand_center = mid + hand_direction * offset_distance

        # 手形图标的大小
        hand_size = 0.3

        # 禁用光照以确保手形图标清晰可见
        glDisable(GL_LIGHTING)
        
        # 绘制手形图标（简化版）
        # 手掌部分（圆形）
        glColor4f(0.9, 0.7, 0.5, 0.9)  # 肤色，半透明
        self.draw_circle(hand_center, hand_size * 0.5, 12)
        
        # 绘制五个手指（椭圆）
        finger_angles = [0, 45, 90, 135, 180]  # 手指角度分布
        finger_lengths = [0.7, 0.8, 0.9, 0.8, 0.7]  # 手指长度
        
        # 确定手指的基本方向（与折线垂直）
        base_direction = hand_direction
        
        for i, angle in enumerate(finger_angles):
            # 旋转手指方向
            rad_angle = np.radians(angle)
            cos_angle = np.cos(rad_angle)
            sin_angle = np.sin(rad_angle)
            
            # 创建旋转矩阵
            rot_matrix = np.array([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]
            ])
            
            # 计算手指方向
            finger_direction = np.dot(rot_matrix, base_direction)
            
            # 计算手指位置
            finger_length = hand_size * finger_lengths[i]
            finger_start = hand_center
            finger_end = finger_start + finger_direction * finger_length
            
            # 绘制手指（圆柱体）
            self.draw_finger(finger_start, finger_end, hand_size * 0.15)
        
        # 添加手形图标轮廓
        glLineWidth(2.0)
        glColor4f(0.6, 0.4, 0.2, 0.9)  # 深肤色边缘
        self.draw_circle(hand_center, hand_size * 0.5, 16, True)
        
        # 重新启用光照
        glEnable(GL_LIGHTING)
    
    def draw_circle(self, center, radius, segments, outline=False):
        """绘制圆形"""
        if outline:
            glBegin(GL_LINE_LOOP)
        else:
            glBegin(GL_TRIANGLE_FAN)
            glVertex3fv(center)
        
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            glVertex3f(x, y, z)
        
        glEnd()
    
    def draw_finger(self, start, end, radius):
        """绘制手指（简化为圆柱体）"""
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        
        # 绘制手指主体（简化为线段）
        glLineWidth(radius * 20)  # 使线段粗一些模拟手指
        glBegin(GL_LINES)
        glVertex3fv(start)
        glVertex3fv(end)
        glEnd()
        
        # 绘制手指尖端（圆形）
        self.draw_circle(end, radius, 8)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.0, 0.0, 0.0, 1.0)  # 纯黑背景

        # 设置光照模型
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])  # 适中的环境光

        # 设置主光源 - 从正面偏上方打光
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 2.0, 10.0, 0.0])  # 更正面的光源位置
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])

        # 设置背光源 - 弱补光
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [0.0, -1.0, -5.0, 0.0])  # 背面弱光源
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])

        # 设置正面材质 - 更亮的紫色
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.6, 0.6, 0.95, 1.0])  # 提高正面环境光反射
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.7, 0.7, 1.0, 1.0])  # 明亮的漫反射
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 20.0)

        # 设置背面材质 - 暗紫色
        glMaterialfv(GL_BACK, GL_AMBIENT, [0.2, 0.2, 0.4, 1.0])  # 降低背面环境光反射
        glMaterialfv(GL_BACK, GL_DIFFUSE, [0.3, 0.3, 0.6, 1.0])  # 较暗的漫反射
        glMaterialfv(GL_BACK, GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])
        glMaterialf(GL_BACK, GL_SHININESS, 5.0)

        # 禁用不需要的特性
        glDisable(GL_CULL_FACE)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_POLYGON_SMOOTH)

        # 启用混合
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 设置着色模型
        glShadeModel(GL_SMOOTH)

        # 创建VBO和IBO
        vertices = np.array(self.points, dtype=np.float32)
        indices = np.array(self.F, dtype=np.uint32)

        # 创建VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # 创建和设置VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # 创建和设置IBO
        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # 解绑
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        """鼠标按下事件处理"""
        if self.mode == "view":
            self.lastPos = event.pos()
        elif self.mode == "fold":
            if event.button() == Qt.RightButton and self.fold_point_selection == 2:
                # 右键切换折叠方向（切换主折叠面）
                self.fold_direction *= -1
                self.fold_angle = 0  # 重置折叠角度
                self.calculate_fold_region()  # 重新计算折叠区域
                self.update()
                return

            if self.fold_point_selection < 2:
                nearest_point_idx, nearest_point = self.get_nearest_boundary_point(event)
                if nearest_point_idx is not None and nearest_point is not None:
                    if self.fold_point_selection == 0:
                        self.fold_line_points = [nearest_point.tolist()]
                        self.fold_point_selection = 1
                    else:  # selecting second point
                        try:
                            first_point_idx = self.boundary_vertices.tolist().index(
                                self.get_point_index(self.fold_line_points[0])
                            )
                            if self.is_different_edges(first_point_idx, nearest_point_idx):
                                self.fold_line_points.append(nearest_point.tolist())
                                self.fold_point_selection = 2
                                self.calculate_fold_region()
                            else:
                                print("Points must be on different edges")
                        except Exception as e:
                            print(f"Error selecting second point: {e}")
                            self.fold_line_points = []
                            self.fold_point_selection = 0
            else:
                self.selected_point = self.get_nearest_point(event)

            self.lastPos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "view":
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            self.rotation_x += dy
            self.rotation_y += dx
            self.update()
            self.lastPos = event.pos()
        elif self.mode == "fold" and self.fold_point_selection == 2 and self.selected_point is not None:
            dx = event.x() - self.lastPos.x()
            # 增加旋转速度系数到2.0，使旋转更加灵敏
            rotation_speed = 2.0
            # 直接根据鼠标移动设置旋转角度
            self.fold_angle = dx * rotation_speed
            self.fold_angle = max(-180, min(180, self.fold_angle))  # 限制最大旋转角度
            self.apply_fold_transformation()
            self.update()
            self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        self.selected_point = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.camera_distance -= 0.5
        else:
            self.camera_distance += 0.5
        self.camera_distance = max(2, self.camera_distance)
        self.update()

    def get_nearest_point(self, event):
        x, y = self.get_3d_coords(event)
        distances = [np.linalg.norm(point - np.array([x, y, 0])) for point in self.points]
        return np.argmin(distances)

    def get_3d_coords(self, event):
        """将鼠标位置转换为OpenGL坐标"""
        width, height = self.width(), self.height()
        x = (event.x() / width * 2.0 - 1.0) * 5.0
        y = (1.0 - event.y() / height * 2.0) * 5.0

        # 应用旋转变换
        angle_x = np.radians(self.rotation_x)
        angle_y = np.radians(self.rotation_y)

        # 创建旋转矩阵
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        rot_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        # 组合旋转
        rot = np.dot(rot_y, rot_x)

        # 应用逆旋转到坐标
        point = np.array([x, y, 0])
        transformed = np.dot(np.linalg.inv(rot), point)

        return transformed[0], transformed[1]

    def get_nearest_boundary_point(self, event):
        """获取最近的边界点 - 向量化优化版本"""
        x, y = self.get_3d_coords(event)
        cursor_pos = np.array([x, y])

        # 一次性获取所有边界点的坐标
        boundary_points = self.V[self.boundary_vertices][:, :2]

        # 计算所有点到鼠标位置的距离
        distances = np.linalg.norm(boundary_points - cursor_pos, axis=1)

        # 检查角点
        n = int(np.sqrt(len(self.V))) - 1
        corner_indices = np.array([0, n, n * (n + 1), (n + 1) ** 2 - 1])

        # 找出在boundary_vertices中的角点索引
        corner_mask = np.isin(self.boundary_vertices, corner_indices)
        if np.any(corner_mask):
            corner_distances = distances[corner_mask]
            min_corner_dist = np.min(corner_distances)
            if min_corner_dist < 0.5:  # 角点阈值
                idx = np.where(corner_mask)[0][np.argmin(corner_distances)]
                return idx, self.V[self.boundary_vertices[idx]].copy()

        # 如果没有找到合适的角点，返回最近的边界点
        nearest_idx = np.argmin(distances)
        min_dist = distances[nearest_idx]

        if min_dist < 2.0:  # 边界点阈值
            return nearest_idx, self.V[self.boundary_vertices[nearest_idx]].copy()

        return None, None

    def find_corner_vertex_index(self, corner):
        """找到与给定角点坐标最接近的顶点索引"""
        min_dist = float('inf')
        corner_idx = None
        for i, v in enumerate(self.V):
            dist = np.linalg.norm(v - np.array(corner))
            if dist < min_dist:
                min_dist = dist
                corner_idx = i
        return corner_idx if min_dist < 0.1 else None

    def get_point_index(self, point):
        """获取点在V数组中的索引"""
        for i, v in enumerate(self.V):
            if np.allclose(v, point, rtol=1e-05, atol=1e-05):  # 使用更宽松的容差
                return i
        return None

    def is_different_edges(self, idx1, idx2):
        """检查两个点是否在不同边上"""
        try:
            boundary_vertices = self.boundary_vertices.tolist()
            n = len(boundary_vertices)

            # 确保索引在有效范围内
            idx1 = idx1 % n
            idx2 = idx2 % n

            # 计算两点之间的最短距离（考虑环形结构）
            dist = min(abs(idx1 - idx2), n - abs(idx1 - idx2))

            # 如果距离为1，说明在同一边上
            return dist > 1

        except Exception as e:
            print(f"Error in is_different_edges: {e}")
            return False

    def calculate_fold_region(self):
        """使用向量化操作计算折叠区域"""
        if len(self.fold_line_points) != 2:
            return

        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        line_vector = line_end - line_start

        # 将所有顶点转换为numpy数组进行向量化计算
        vertices = np.array(self.points)
        vertex_vectors = vertices - line_start

        # 计算叉积和点积，一次性处理所有点
        cross_products = np.cross(line_vector, vertex_vectors)
        dots = np.einsum('ij,j->i', cross_products, [0, 0, 1]) * self.fold_direction

        # 使用布尔索引找出所有需要折叠的点
        self.fold_region = np.where(dots > 0)[0].tolist()

    def apply_fold_transformation(self):
        """使用向量化操作应用折叠变换"""
        if self.fold_region is None or len(self.fold_line_points) != 2:
            return

        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        fold_axis = (line_end - line_start) / np.linalg.norm(line_end - line_start)

        # 将所有需要变换的点转换为numpy数组
        vertices = np.array([self.points[i] for i in self.fold_region])

        # 批量平移
        translated = vertices - line_start

        # 批量旋转
        angle = np.radians(self.fold_angle)
        rotated = self.rotate_points_around_axis(translated, fold_axis, angle)

        # 批量平移回原位置
        result = rotated + line_start

        # 更新点的位置
        for i, idx in enumerate(self.fold_region):
            self.points[idx] = result[i].tolist()

    def rotate_points_around_axis(self, points, axis, angle):
        """批量旋转多个点"""
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        # 计算叉积和点积（批量）
        cross_prods = np.cross(points, axis)
        dot_prods = np.einsum('ij,j->i', points, axis)

        # 使用罗德里格旋转公式进行批量计算
        rotated = (points * cos_theta +
                   cross_prods * sin_theta +
                   np.outer(dot_prods * (1 - cos_theta), axis))

        return rotated

    def init_refined_rectangle(self, n=20):  # 降低密度使其更容易操作
        """初始化更加细化的矩形模型，使用向量化操作提高性能"""
        scale = 5.0

        # 使用numpy的meshgrid生成网格点，更高效
        x = np.linspace(-scale / 2, scale / 2, n + 1)
        y = np.linspace(-scale / 2, scale / 2, n + 1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # 将网格点转换为顶点数组
        V = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # 使用向量化操作生成三角形面
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        v1 = i * (n + 1) + j
        v2 = (i + 1) * (n + 1) + j
        v3 = i * (n + 1) + (j + 1)
        v4 = (i + 1) * (n + 1) + (j + 1)

        # 创建三角形面（每个方格分为两个三角形）
        F1 = np.stack([v1.flatten(), v2.flatten(), v3.flatten()], axis=1)
        F2 = np.stack([v2.flatten(), v4.flatten(), v3.flatten()], axis=1)
        F = np.vstack([F1, F2])

        return V, F


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.glWidget = OrigamiWidget()
        self.initUI()

    def initUI(self):
        # 创建主窗口部件
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # 左侧控制面板
        control_panel = QFrame()
        control_panel.setFixedWidth(250)
        control_panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFB6C1, stop:1 #87CEEB);
                border-radius: 15px;
                border: 3px solid #FFFFFF;
                margin: 10px;
            }
        """)

        # 控制面板布局
        panel_layout = QVBoxLayout()
        panel_layout.setAlignment(Qt.AlignTop)
        panel_layout.setContentsMargins(20, 30, 20, 30)
        control_panel.setLayout(panel_layout)

        # 标题
        title = QLabel("魔法折纸")
        title.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font: bold 28px 'Comic Sans MS';
                qproperty-alignment: AlignCenter;
                margin-bottom: 20px;
            }
        """)
        panel_layout.addWidget(title)

        # 模式按钮
        mode_buttons = [
            ("🔍 观察模式", "#FF69B4", self.set_view_mode),
            ("✂️ 折叠模式", "#00BFFF", self.set_fold_mode),
            ("🎨 换色模式", "#FFD700", self.change_paper_color),  # 修改按钮功能名称
            ("❓ 帮助", "#FFA500", self.show_help)
        ]

        for text, color, func in mode_buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color};
                    color: white;
                    font: bold 16px 'Arial Rounded MT Bold';
                    border: 2px solid white;
                    border-radius: 15px;
                    padding: 15px;
                    margin: 10px;
                }}
                QPushButton:hover {{
                    background: {color};
                    border: 3px solid white;
                }}
            """)
            btn.clicked.connect(func)
            panel_layout.addWidget(btn)

        # 提示标签
        self.hint_label = QLabel("开始你的折纸魔法吧！")
        self.hint_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font: 14px 'Comic Sans MS';
                qproperty-alignment: AlignCenter;
                padding: 10px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                margin-top: 20px;
            }
        """)
        panel_layout.addWidget(self.hint_label)
        
        # 添加装饰元素
        decoration = QLabel()
        decoration.setPixmap(QPixmap("origami_decoration.png").scaled(200, 200, Qt.KeepAspectRatio))
        panel_layout.addWidget(decoration, alignment=Qt.AlignCenter)
        panel_layout.addStretch()

        # 右侧OpenGL窗口区域
        gl_container = QFrame()
        gl_container.setStyleSheet("""
            QFrame {
                background: #F0F8FF;
                border-radius: 20px;
                border: 3px dashed #87CEFA;
                margin: 10px;
            }
        """)
        gl_layout = QVBoxLayout(gl_container)
        gl_layout.addWidget(self.glWidget)

        # 添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(gl_container, stretch=1)

    def set_view_mode(self):
        self.glWidget.mode = "view"
        self.hint_label.setText("✨ 用鼠标转动折纸，欣赏你的作品吧！")

    def set_fold_mode(self):
        # 重置视角和旋转角度
        self.glWidget.camera_distance = 11.0  # 重置相机距离
        self.glWidget.rotation_x = 0          # 重置X轴旋转
        self.glWidget.rotation_y = 0          # 重置Y轴旋转
        self.glWidget.rotation_z = 0          # 重置Z轴旋转
        
        # 设置折叠模式
        self.glWidget.mode = "fold"
        self.glWidget.fold_line_points = []
        self.glWidget.fold_point_selection = 0
        self.glWidget.fold_region = None
        self.glWidget.fold_angle = 0
        self.glWidget.update()
        self.hint_label.setText("✨ 点击两个点画折线，然后拖动鼠标来折叠~")

    # 重命名方法，使其更清晰
    def change_paper_color(self):
        self.glWidget.current_color = (self.glWidget.current_color + 1) % len(self.glWidget.colors)
        self.glWidget.update()
        self.hint_label.setText("✨ 换个新颜色，创作更美的作品！")

    def show_help(self):
        QMessageBox.information(self, "折纸小帮手",
                                "欢迎来到魔法折纸乐园！✨\n\n"
                                "🔍 观察模式：\n"
                                "   转动鼠标来欣赏你的折纸作品\n\n"
                                "✂️ 折叠模式：\n"
                                "   1. 点击两个点来画折线\n"
                                "   2. 拖动鼠标来折叠纸张\n"
                                "   3. 右键点击切换折叠方向（手形图标指示将要旋转的一侧）\n\n"
                                "🎨 换色模式：\n"
                                "   点击按钮换不同的纸张颜色\n\n"
                                "开始创作你的魔法折纸吧！💫")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont()
    font.setFamily("Comic Sans MS")
    font.setPointSize(12)
    app.setFont(font)

    window = MainWindow()
    window.setWindowTitle("魔法折纸乐园")
    window.resize(1200, 800)
    window.show()
    app.exec_()