import igl
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QPushButton, QVBoxLayout, QWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import random


# 初步处理了折叠的锯齿问题，存在新的bug需要调整，而且二段折叠基本必有锯齿，感觉根本问题就是不能够搭建一个点的密度足够高的面
class OrigamiWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OrigamiWidget, self).__init__(parent)
        self.mode = "view"  # 初始模式
        self.points = []  # 顶点
        self.selected_point = None  # 当前选中的点
        self.camera_distance = 11.0  # 摄像头初始距离，越小贴的越近
        self.lastPos = None  # 上次鼠标位置
        self.rotation_x = 0  # 模型旋转
        self.rotation_y = 0
        self.rotation_z = 0

        # 折叠功能
        self.fold_line_points = []  # 折叠线的两个端点
        self.fold_region = None  # 折叠区域点
        self.fold_angle = 0  # 当前折叠角度

        # 初始化矩形模型（细化）
        self.V, self.F = self.init_refined_rectangle(n=35)  # 增加细化程度（默认n=30）基本消除不了锯齿
        self.points = self.V.tolist()  # 顶点转列表
        self.boundary_vertices = igl.boundary_loop(self.F)  # 获取边界点

    def init_refined_rectangle(self, n=30):
        """初始化更加细化的矩形模型"""
        scale = 5.0  # 放大倍数
        V = []

        # 生成细化网格的顶点
        for i in range(n + 1):
            for j in range(n + 1):
                # V.append([i / n - 0.5, j / n - 0.5, 0])  # 按照比例生成矩形的顶点
                V.append([(i / n - 0.5) * scale, (j / n - 0.5) * scale, 0])
        V = np.array(V)

        F = []
        # 创建三角形面，每个四个顶点形成两个三角形
        for i in range(n):
            for j in range(n):
                v1 = i * (n + 1) + j
                v2 = (i + 1) * (n + 1) + j
                v3 = i * (n + 1) + (j + 1)
                v4 = (i + 1) * (n + 1) + (j + 1)

                # 第一个三角形面
                F.append([v1, v2, v3])
                # 第二个三角形面
                F.append([v2, v4, v3])

        F = np.array(F)

        return V, F

    def initializeGL(self):
        """初始化OpenGL设置"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)  # 启用双面光照
        # 添加额外光源
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 0.0, -1.0, 0.0])  # 从背面照亮
        # 提高漫反射分量，使背面亮度增强
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])  # 增强亮度

        # 提高镜面反射分量，让表面更闪亮
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])  # 增强亮度

        # 设置绘制模式为线框
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        # 启用多边形抗锯齿
        glEnable(GL_POLYGON_SMOOTH)
        # 启用虚线绘制
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, 0xAAAA)

    def resizeGL(self, w, h):
        """调整窗口大小"""
        glViewport(0, 0, w, h)  # 指定渲染内容在窗口中的位置和大小，使渲染结果适配窗口的变化
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100)  # 设置透视投影，让远处的物体看起来更小，从而模拟真实的三维视图。
        glMatrixMode(GL_MODELVIEW)  # 投影矩阵设置完成后，准备进行场景绘制。

    def paintGL(self):
        """绘制函数"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 确保每次绘制开始时，屏幕是干净的，没有残留的图形信息。
        glLoadIdentity()
        gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0)  # 设置观察点

        glRotatef(self.rotation_x, 1, 0, 0)  # 分别绕 X、Y、Z 轴旋转模型，允许通过外部交互（例如鼠标拖动）改变模型的观察角度。
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)

        # 绘制矩形（细化后的网格）
        glColor3f(0.5, 0.5, 0.9)  # 设置绘制的颜色为蓝紫色。
        glBegin(GL_TRIANGLES)  # 定义绘制模式为三角形。
        for face in self.F:  # 按照网格定义的三角形面逐个绘制整个模型。
            for i in face:
                glVertex3fv(self.points[i])
        glEnd()

        # 绘制交互点
        glColor3f(0.0, 0.0, 0.0)  # 设置绘制的颜色为黑色。
        glPointSize(2.5)  # 设置点的大小为 2.5 像素
        glBegin(GL_POINTS)
        for point in self.points:  # 将模型中的顶点可视化，便于交互和调试。
            glVertex3fv(point)
        glEnd()

        # 绘制折叠线
        if len(self.fold_line_points) == 2:  # 只有当 self.fold_line_points 有两个点时，才绘制折叠线。
            glColor3f(0.5, 0.5, 0.5)  # 设置折叠线颜色为淡紫色
            glLineWidth(2.0)  # 设置线宽为 2 像素。
            glBegin(GL_LINES)  # 定义绘制模式为线，绘制一条从 fold_line_points[0] 到 fold_line_points[1] 的直线。
            glVertex3fv(self.fold_line_points[0])
            glVertex3fv(self.fold_line_points[1])
            glEnd()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.mode == "view":  # 当前模式是 "view"，表示用户正在调整模型视图（例如旋转或缩放模型）
            self.lastPos = event.pos()  # 记录鼠标按下时的位置 event.pos()，用于后续的鼠标拖动事件（例如旋转模型），计算鼠标移动的偏移量
        elif self.mode == "fold":  # 当前模式是 "fold"，表示用户正在选择折叠相关的点或线
            self.selected_point = self.get_nearest_point(event)  # 通过鼠标点击的屏幕位置，找到模型中距离鼠标最近的顶点。
            self.lastPos = event.pos()  # 记录鼠标按下时的位置

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.mode == "view":
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()  # 计算鼠标位移
            self.rotation_x += dy  # 更新模型旋转
            self.rotation_y += dx
            self.update()  # 刷新视图，触发重新绘制模型以反映旋转的变化
            self.lastPos = event.pos()  # 记录新鼠标位置，用于计算下一次移动的偏移量
        elif self.mode == "fold" and self.selected_point is not None:
            dx = event.x() - self.lastPos.x()  # 计算鼠标水平方向位移
            self.fold_angle += dx * 0.1  # 更新折叠角度，每单位水平位移增加 0.1 度折叠角度。
            self.fold_angle = max(0, min(self.fold_angle, 180))  # 限制折叠角度范围为 [0, 180], 似乎没有发挥效果
            self.apply_fold_transformation()  # 根据新的折叠角度对模型进行变形。
            self.update()  # 刷新视图，触发重新绘制模型以反映折叠的变化
            self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.selected_point = None

    def wheelEvent(self, event):
        """鼠标滚轮实现视角缩放"""
        delta = event.angleDelta().y()  # 获取滚轮滚动值
        if delta > 0:
            self.camera_distance -= 0.5
        else:
            self.camera_distance += 0.5
        self.camera_distance = max(2, self.camera_distance)
        self.update()  # 刷新视图，触发重新绘制观察视角拉近或拉远

    def get_nearest_point(self, event):
        """获取最近的顶点索引"""
        x, y = self.get_3d_coords(event)  # 获取鼠标的3D坐标，将鼠标在窗口中的 2D 坐标（像素值）转换为 3D OpenGL 坐标。
        distances = [np.linalg.norm(point - np.array([x, y, 0])) for point in self.points]  # 计算鼠标与每个顶点的距离
        return np.argmin(distances)  # 找出距离最小的顶点索引

    def get_3d_coords(self, event):
        """将鼠标位置转换为OpenGL坐标"""
        width, height = self.width(), self.height()  # 计算并返回鼠标在 OpenGL 世界中对应的 2D 坐标 (x, y)
        x = (event.x() / width) * 2 - 1
        y = 1 - (event.y() / height) * 2
        return x, y

    # 从模型的边界顶点中随机选择两点生成一条折叠线，并确保这条线满足一定条件（不同边缘的约束）
    def set_random_fold_line(self):
        """随机生成一条折叠线"""
        while True:
            idx1, idx2 = random.sample(range(len(self.boundary_vertices)), 2)
            point1 = self.V[self.boundary_vertices[idx1]]
            point2 = self.V[self.boundary_vertices[idx2]]
            print(point1, "   ")
            print(point2, "   ")
            if self.is_different_edges(idx1, idx2):  # 不在同一条外层边缘上
                self.fold_line_points = [point1.tolist(), point2.tolist()]  # 把选择的两个点加入数组中为绘制折线做准备
                print(f"Random fold line: {self.fold_line_points}")
                self.calculate_fold_region()  #
                break

    def is_different_edges(self, idx1, idx2):
        """检查两个点是否在不同边上"""
        if abs(idx1 - idx2) == 1 or {idx1, idx2} == {0, len(self.boundary_vertices) - 1}:
            return False
        return True

    def handle_fold_line_intersections(self):
        """处理折线与三角形面片的交点"""
        if len(self.fold_line_points) != 2:
            return

        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        line_dir = line_end - line_start

        new_vertices = []  # 存储新生成的顶点
        new_faces = []  # 存储新的三角形面片
        processed_faces = set()  # 记录已处理的面片

        # 遍历所有三角形面片
        for face_idx, face in enumerate(self.F):
            if face_idx in processed_faces:
                continue

            # 获取三角形的三个顶点
            v0 = np.array(self.points[face[0]])
            v1 = np.array(self.points[face[1]])
            v2 = np.array(self.points[face[2]])

            # 检查折线是否与当前三角形相交
            intersections = []

            # 检查折线与三角形各边的交点
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            for edge in edges:
                # 计算交点
                p1, p2 = edge
                v = p2 - p1
                w = line_end - line_start

                # 解线性方程组找交点
                denom = np.cross(v[:2], w[:2])
                if abs(denom) > 1e-8:  # 避免除零
                    t = np.cross(p1[:2] - line_start[:2], w[:2]) / denom
                    s = np.cross(p1[:2] - line_start[:2], v[:2]) / denom

                    if 0 <= t <= 1 and 0 <= s <= 1:
                        # 找到交点
                        intersection = p1 + t * v
                        # 确保z坐标正确
                        intersection[2] = 0
                        intersections.append(intersection)

            if len(intersections) == 2:
                # 将新的顶点添加到顶点列表
                for intersection in intersections:
                    new_vertices.append(intersection)

                # 更新当前三角形的拓扑结构
                v_idx = len(self.points)
                self.points.extend([intersection.tolist() for intersection in intersections])

                # 根据交点位置重新构建三角形
                # 这里需要根据具体情况划分三角形，下面是一个简化的示例
                new_faces.extend([
                    [face[0], v_idx, v_idx + 1],
                    [face[1], v_idx, v_idx + 1],
                    [face[2], v_idx, v_idx + 1]
                ])

                processed_faces.add(face_idx)

        # 更新模型的顶点和面片信息
        if new_vertices:
            # 更新面片信息
            self.F = np.array(list(self.F) + new_faces)
            # 更新顶点信息已经在循环中完成

    def calculate_fold_region(self):
        """划分折叠区域并处理折线附近的点"""
        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        line_vector = line_end - line_start
        line_length = np.linalg.norm(line_vector)
        line_dir = line_vector / line_length

        self.fold_region = []
        # 存储需要投影到折线上的点的信息
        self.projected_points = {}  # 格式：{点索引: 投影点坐标}

        # 定义折线附近的阈值（可以根据需要调整）
        threshold = 0.1

        for i, vertex in enumerate(self.points):
            vertex = np.array(vertex)
            vertex_vector = vertex - line_start

            # 计算点到折线的投影
            proj_length = np.dot(vertex_vector, line_dir)
            proj_point = line_start + proj_length * line_dir

            # 计算点到折线的垂直距离
            dist_to_line = np.linalg.norm(vertex - proj_point)

            # 如果点在折线附近，将其投影到折线上
            if dist_to_line < threshold:
                self.projected_points[i] = proj_point

            # 判断点在折线的哪一侧
            if np.dot(np.cross(line_vector, vertex_vector), [0, 0, 1]) > 0:
                self.fold_region.append(i)

    def apply_fold_transformation(self):
        """应用折叠变形，确保折线附近的点对齐"""
        if self.fold_region is not None and len(self.fold_line_points) == 2:
            line_start = np.array(self.fold_line_points[0])
            line_end = np.array(self.fold_line_points[1])
            fold_axis = (line_end - line_start) / np.linalg.norm(line_end - line_start)

            # 首先处理需要投影到折线上的点
            for i, proj_point in self.projected_points.items():
                if i in self.fold_region:
                    # 如果点在折叠区域内，先将其投影到折线上，然后再进行旋转
                    translated = proj_point - line_start
                    rotated = self.rotate_around_axis(translated, fold_axis, np.radians(self.fold_angle))
                    self.points[i] = rotated + line_start
                else:
                    # 如果点不在折叠区域内，直接将其投影到折线上
                    self.points[i] = proj_point.tolist()

            # 处理其他需要折叠的点
            for i in self.fold_region:
                if i not in self.projected_points:
                    vertex = np.array(self.points[i])
                    translated = vertex - line_start
                    rotated = self.rotate_around_axis(translated, fold_axis, np.radians(self.fold_angle))
                    self.points[i] = rotated + line_start

    # 用于将一个点绕任意轴旋转指定角度，返回旋转后的点的坐标。
    def rotate_around_axis(self, point, axis, angle):
        """绕任意轴旋转"""
        axis = axis / np.linalg.norm(axis)  # 将输入的旋转轴 axis 归一化，确保其为单位向量。
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)  # 输入的旋转角度 angle（弧度制）计算对应的余弦值和正弦值。
        cross_prod = np.cross(axis, point)  # 计算轴向量与点向量的叉积。
        dot_prod = np.dot(axis, point)  # 计算轴向量与点向量的点积。

        rotated = (point * cos_theta +
                   cross_prod * sin_theta +
                   axis * dot_prod * (1 - cos_theta))  # 应用 Rodrigues 旋转公式， 计算旋转后的点坐标
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
        self.glWidget.set_random_fold_line()


# PyQt应用的启动逻辑，用于创建、配置和启动一个交互式窗口程序
app = QApplication([])  # 初始化应用对象
window = MainWindow()  # 自定义的主窗口类，继承自 QMainWindow
window.setWindowTitle("Interactive Origami Viewer")
window.resize(900, 900)  # 调整窗口尺寸大小
window.show()
app.exec_()  # 启动 Qt 的事件循环，进入应用的主循环。持续监听用户事件（如鼠标、键盘）并分发到对应的事件处理函数。
