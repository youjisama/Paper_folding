import igl
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import random

# 实现了折叠往前往后的两个方向
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
        self.fold_direction = 1  # 1 表示向上折叠，-1 表示向下折叠
        self.fold_point_selection = 0  # 0: 未选择, 1: 已选择起点, 2: 已选择终点

        # 初始化矩形模型（细化）
        self.V, self.F = self.init_refined_rectangle(n=30)
        self.points = self.V.tolist()
        self.boundary_vertices = igl.boundary_loop(self.F)

    def init_refined_rectangle(self, n=30):
        """初始化更加细化的矩形模型"""
        scale = 5.0
        V = []

        # 确保n是偶数
        if n % 2 != 0:
            n += 1

        # 生成网格顶点，确保角点在正确的位置
        for i in range(n + 1):
            for j in range(n + 1):
                x = (i / n - 0.5) * scale
                y = (j / n - 0.5) * scale
                V.append([x, y, 0])
        V = np.array(V)

        # 创建三角形面
        F = []
        for i in range(n):
            for j in range(n):
                v1 = i * (n + 1) + j
                v2 = (i + 1) * (n + 1) + j
                v3 = i * (n + 1) + (j + 1)
                v4 = (i + 1) * (n + 1) + (j + 1)
                F.append([v1, v2, v3])
                F.append([v2, v4, v3])
        F = np.array(F)

        return V, F

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 0.0, -1.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])

        glEnable(GL_POLYGON_SMOOTH)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, 0xAAAA)

        # 设置绘制模式为线框
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0)

        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)

        # 绘制矩形
        glColor3f(0.5, 0.5, 0.9)
        glBegin(GL_TRIANGLES)
        for face in self.F:
            for i in face:
                glVertex3fv(self.points[i])
        glEnd()

        # 绘制边界点
        if self.mode == "fold" and self.fold_point_selection < 2:
            # 绘制普通边界点
            glColor3f(0.0, 1.0, 0.0)  # 绿色
            glPointSize(8.0)
            glBegin(GL_POINTS)
            for idx in self.boundary_vertices:
                if idx not in [0, int(np.sqrt(len(self.V))) - 1,
                               (int(np.sqrt(len(self.V))) - 1) ** 2,
                               len(self.V) - 1]:  # 不是角点
                    glVertex3fv(self.V[idx])
            glEnd()

            # 特别绘制角点
            glColor3f(1.0, 0.0, 1.0)  # 紫色
            glPointSize(12.0)
            glBegin(GL_POINTS)
            for idx in [0, int(np.sqrt(len(self.V))) - 1,
                        (int(np.sqrt(len(self.V))) - 1) ** 2,
                        len(self.V) - 1]:  # 角点
                if idx in self.boundary_vertices:
                    glVertex3fv(self.V[idx])
            glEnd()

        # 绘制已选择的折线点
        if len(self.fold_line_points) >= 1:
            glColor3f(1.0, 0.0, 0.0)  # 红色
            glPointSize(10.0)
            glBegin(GL_POINTS)
            glVertex3fv(self.fold_line_points[0])
            glEnd()

            if len(self.fold_line_points) == 2:
                glColor3f(1.0, 1.0, 0.0)  # 黄色
                glPointSize(10.0)
                glBegin(GL_POINTS)
                glVertex3fv(self.fold_line_points[1])
                glEnd()

                glColor3f(0.5, 0.5, 0.5)  # 灰色
                glLineWidth(2.0)
                glBegin(GL_LINES)
                glVertex3fv(self.fold_line_points[0])
                glVertex3fv(self.fold_line_points[1])
                glEnd()

                # 绘制折叠方向箭头
                if self.fold_point_selection == 2:
                    self.draw_fold_direction_arrow()

        # 绘制交互点
        glColor3f(0.0, 0.0, 0.0)
        glPointSize(2.5)
        glBegin(GL_POINTS)
        for point in self.points:
            glVertex3fv(point)
        glEnd()

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
            if self.fold_point_selection < 2:
                nearest_point_idx, nearest_point = self.get_nearest_boundary_point(event)
                print(f"Fold point selection: {self.fold_point_selection}, Nearest point: {nearest_point_idx}")

                if nearest_point_idx is not None and nearest_point is not None:
                    if self.fold_point_selection == 0:
                        self.fold_line_points = [nearest_point.tolist()]
                        self.fold_point_selection = 1
                        print("Selected start point")

                    else:  # selecting second point
                        try:
                            first_point_idx = self.boundary_vertices.tolist().index(
                                self.get_point_index(self.fold_line_points[0])
                            )

                            if self.is_different_edges(first_point_idx, nearest_point_idx):
                                self.fold_line_points.append(nearest_point.tolist())
                                self.fold_point_selection = 2
                                self.calculate_fold_region()
                                print("Selected end point")
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

        if event.button() == Qt.RightButton and self.fold_point_selection == 2:
            # 右键切换折叠方向
            self.fold_direction *= -1
            self.fold_angle = 0  # 重置折叠角度
            self.calculate_fold_region()  # 重新计算折叠区域
            self.update()
            return

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

    def calculate_fold_region(self):
        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        line_vector = line_end - line_start

        self.fold_region = []
        for i, vertex in enumerate(self.points):
            vertex_vector = np.array(vertex) - line_start
            # 根据叉积判断点在折线的哪一侧
            # 注意：这里不需要乘以fold_direction，因为旋转方向现在由mouseMoveEvent中的dx控制
            if np.dot(np.cross(line_vector, vertex_vector), [0, 0, 1]) > 0:
                self.fold_region.append(i)

    def apply_fold_transformation(self):
        if self.fold_region is not None and len(self.fold_line_points) == 2:
            line_start = np.array(self.fold_line_points[0])
            line_end = np.array(self.fold_line_points[1])
            fold_axis = (line_end - line_start) / np.linalg.norm(line_end - line_start)

            for i in self.fold_region:
                vertex = np.array(self.points[i])
                translated = vertex - line_start
                rotated = self.rotate_around_axis(translated, fold_axis, np.radians(self.fold_angle))  # 移除abs
                self.points[i] = rotated + line_start

    def rotate_around_axis(self, point, axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        cross_prod = np.cross(axis, point)
        dot_prod = np.dot(axis, point)

        rotated = (point * cos_theta +
                   cross_prod * sin_theta +
                   axis * dot_prod * (1 - cos_theta))
        return rotated


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
