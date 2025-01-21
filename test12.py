import igl
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import random


# 点密度大，响应速度快，但是存在bug
class OrigamiWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OrigamiWidget, self).__init__(parent)
        self.mode = "view"
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

        # VBO和IBO
        self.vbo = None
        self.ibo = None
        self.vao = None

        # 缓存数据
        self.vertices_array = None
        self.indices_array = None
        self.fold_region_cache = None
        self.last_fold_angle = None

        # 初始化矩形模型（超高密度）
        self.V, self.F = self.init_refined_rectangle(n=200)
        self.points = self.V.tolist()
        self.boundary_vertices = igl.boundary_loop(self.F)

        # 预计算数据
        self.precompute_data()

    def precompute_data(self):
        """预计算和缓存常用数据"""
        # 转换为适合GPU的数据类型
        self.vertices_array = np.array(self.points, dtype=np.float32)
        self.indices_array = np.array(self.F, dtype=np.uint32)

        # 预计算边界信息
        n = int(np.sqrt(len(self.V)))
        self.boundary_indices = set([i for i in range(len(self.V)) if
                                     i < n or i % n == 0 or
                                     i % n == n - 1 or i >= len(self.V) - n])

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # 禁用背面剔除，确保两面都能看到
        glDisable(GL_CULL_FACE)

        # 设置光照
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

        # 创建VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # 创建和设置VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices_array.nbytes, self.vertices_array, GL_DYNAMIC_DRAW)

        # 创建和设置IBO
        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices_array.nbytes, self.indices_array, GL_STATIC_DRAW)

        # 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # 解绑
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0)

        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)

        # 设置材质属性
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

        # 使用VAO和IBO渲染
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)

        # 只在需要时更新VBO
        if self.fold_region is not None and self.last_fold_angle != self.fold_angle:
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.vertices_array.nbytes, self.vertices_array)
            self.last_fold_angle = self.fold_angle

        # 绘制正面
        glColor3f(0.5, 0.5, 0.9)
        glDrawElements(GL_TRIANGLES, len(self.indices_array), GL_UNSIGNED_INT, None)

        # 解绑
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # 绘制控制元素
        self.draw_fold_elements()

    def calculate_fold_region(self):
        """优化的折叠区域计算"""
        if len(self.fold_line_points) != 2:
            return

        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        line_vector = line_end - line_start

        # 使用向量化操作计算
        vertex_vectors = self.vertices_array - line_start
        cross_products = np.cross(line_vector, vertex_vectors)
        dots = np.einsum('ij,j->i', cross_products, [0, 0, 1]) * self.fold_direction

        # 使用布尔索引
        self.fold_region = np.where(dots > 0)[0]
        self.fold_region_cache = set(self.fold_region)  # 缓存结果

    def apply_fold_transformation(self):
        """优化的折叠变换应用"""
        if self.fold_region is None or len(self.fold_line_points) != 2:
            return

        # 只在角度改变时更新
        if self.last_fold_angle == self.fold_angle:
            return

        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        fold_axis = (line_end - line_start) / np.linalg.norm(line_end - line_start)

        # 只转换需要的点
        vertices_to_transform = self.vertices_array[self.fold_region]
        translated = vertices_to_transform - line_start

        # 批量旋转
        angle = np.radians(self.fold_angle)
        rotated = self.rotate_points_around_axis(translated, fold_axis, angle)

        # 更新顶点数组
        self.vertices_array[self.fold_region] = rotated + line_start

        # 更新points列表（如果需要）
        for i, idx in enumerate(self.fold_region):
            self.points[idx] = self.vertices_array[idx].tolist()

    def rotate_points_around_axis(self, points, axis, angle):
        """优化的批量旋转"""
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        # 使用向量化操作
        cross_prods = np.cross(points, axis)
        dot_prods = np.einsum('ij,j->i', points, axis)

        # 使用广播进行计算
        return (points * cos_theta +
                cross_prods * sin_theta +
                np.outer(dot_prods * (1 - cos_theta), axis))

    def __del__(self):
        """清理OpenGL资源"""
        if hasattr(self, 'vbo') and self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])
        if hasattr(self, 'ibo') and self.ibo is not None:
            glDeleteBuffers(1, [self.ibo])
        if hasattr(self, 'vao') and self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])

    def init_refined_rectangle(self, n=200):
        """初始化高密度矩形模型，使用优化的内存管理"""
        scale = 5.0

        # 使用线性空间创建更密集的网格
        x = np.linspace(-scale / 2, scale / 2, n + 1, dtype=np.float32)
        y = np.linspace(-scale / 2, scale / 2, n + 1, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # 创建顶点数组
        V = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # 生成三角形面片，确保正确的面朝向
        i, j = np.meshgrid(np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32), indexing='ij')
        v1 = i * (n + 1) + j
        v2 = (i + 1) * (n + 1) + j
        v3 = i * (n + 1) + (j + 1)
        v4 = (i + 1) * (n + 1) + (j + 1)

        # 创建两组三角形，确保正确的面朝向
        F1 = np.stack([v1.flatten(), v2.flatten(), v3.flatten()], axis=1)
        F2 = np.stack([v2.flatten(), v4.flatten(), v3.flatten()], axis=1)
        F = np.vstack([F1, F2]).astype(np.int32)

        return V, F

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def draw_fold_elements(self):
        """绘制折线和控制点等元素"""
        # 绘制已选择的折线点
        if len(self.fold_line_points) >= 1:
            glColor3f(1.0, 0.0, 0.0)
            glPointSize(10.0)
            glBegin(GL_POINTS)
            glVertex3fv(self.fold_line_points[0])
            glEnd()

            if len(self.fold_line_points) == 2:
                glColor3f(1.0, 1.0, 0.0)
                glPointSize(10.0)
                glBegin(GL_POINTS)
                glVertex3fv(self.fold_line_points[1])
                glEnd()

                glColor3f(0.5, 0.5, 0.5)
                glLineWidth(2.0)
                glBegin(GL_LINES)
                glVertex3fv(self.fold_line_points[0])
                glVertex3fv(self.fold_line_points[1])
                glEnd()

                if self.fold_point_selection == 2:
                    self.draw_fold_direction_arrow()

    def draw_fold_direction_arrow(self):
        """绘制表示折叠方向的箭头"""
        if len(self.fold_line_points) != 2:
            return

        # 计算折线中点
        start = np.array(self.fold_line_points[0])
        end = np.array(self.fold_line_points[1])
        mid_point = (start + end) / 2

        # 计算折线向量和垂直向量
        line_vector = end - start
        line_vector = line_vector / np.linalg.norm(line_vector)
        normal_vector = np.array([-line_vector[1], line_vector[0], 0])

        # 箭头参数
        arrow_length = 0.5
        arrow_width = 0.2
        direction = self.fold_direction

        # 箭头顶点
        arrow_tip = mid_point + normal_vector * arrow_length * direction
        arrow_left = arrow_tip - (normal_vector * arrow_width + line_vector * arrow_width) * direction
        arrow_right = arrow_tip - (normal_vector * arrow_width - line_vector * arrow_width) * direction

        # 绘制箭头
        glColor3f(1.0, 0.0, 0.0)  # 红色
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # 箭头主干
        glVertex3fv(mid_point)
        glVertex3fv(arrow_tip)
        # 箭头两翼
        glVertex3fv(arrow_tip)
        glVertex3fv(arrow_left)
        glVertex3fv(arrow_tip)
        glVertex3fv(arrow_right)
        glEnd()

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
            self.fold_angle = -dx * rotation_speed
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
        """获取最近的边界点"""
        x, y = self.get_3d_coords(event)
        min_dist = float('inf')
        nearest_idx = None
        nearest_point = None

        # 定义四个角点的索引
        n = int(np.sqrt(len(self.V))) - 1  # 从顶点总数计算n
        corner_indices = [
            0,  # 左下角
            n,  # 左上角
            n * (n + 1),  # 右下角
            (n + 1) ** 2 - 1  # 右上角
        ]

        # 首先检查是否点击了角点
        for corner_idx in corner_indices:
            if corner_idx in self.boundary_vertices:
                point = self.V[corner_idx]
                dist = np.linalg.norm(point[:2] - np.array([x, y]))
                if dist < 0.5:  # 角点使用更小的阈值
                    return self.boundary_vertices.tolist().index(corner_idx), point.copy()

        # 如果没有点击角点，检查其他边界点
        for i, boundary_idx in enumerate(self.boundary_vertices):
            point = self.V[boundary_idx]
            dist = np.linalg.norm(point[:2] - np.array([x, y]))
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                nearest_point = point.copy()

        threshold = 2.0
        if min_dist < threshold:
            print(f"Found nearest point: {nearest_idx} with distance {min_dist}")
            return nearest_idx, nearest_point
        else:
            print(f"No point found. Minimum distance was {min_dist}")
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


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.glWidget = OrigamiWidget()
        self.initUI()

    def initUI(self):
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        self.view_button = QPushButton('View')
        self.view_button.clicked.connect(self.set_view_mode)
        layout.addWidget(self.view_button)

        self.fold_button = QPushButton('Fold')
        self.fold_button.clicked.connect(self.set_fold_mode)
        layout.addWidget(self.fold_button)

        layout.addWidget(self.glWidget)
        self.setCentralWidget(container)

    def set_view_mode(self):
        self.glWidget.mode = "view"

    def set_fold_mode(self):
        self.glWidget.mode = "fold"
        self.glWidget.fold_line_points = []
        self.glWidget.fold_point_selection = 0
        self.glWidget.fold_region = None
        self.glWidget.fold_angle = 0
        self.glWidget.update()


app = QApplication([])
window = MainWindow()
window.setWindowTitle("Interactive Origami Viewer")
window.resize(1200, 1200)
window.show()
app.exec_()
