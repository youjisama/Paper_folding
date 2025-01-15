import igl
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QPushButton, QVBoxLayout, QWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import random
# 蛮有质感的一版
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
        self.V, self.F = self.init_refined_rectangle(n=30)  # 增加细化程度（默认n=30）
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

    def resizeGL(self, w, h):
        """调整窗口大小"""
        glViewport(0, 0, w, h) # 指定渲染内容在窗口中的位置和大小，使渲染结果适配窗口的变化
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100) # 设置透视投影，让远处的物体看起来更小，从而模拟真实的三维视图。
        glMatrixMode(GL_MODELVIEW) # 投影矩阵设置完成后，准备进行场景绘制。

    def paintGL(self):
        """绘制函数"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # 确保每次绘制开始时，屏幕是干净的，没有残留的图形信息。
        glLoadIdentity()
        gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0) # 设置观察点

        glRotatef(self.rotation_x, 1, 0, 0) # 分别绕 X、Y、Z 轴旋转模型，允许通过外部交互（例如鼠标拖动）改变模型的观察角度。
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)

        # 绘制矩形（细化后的网格）
        glColor3f(0.5, 0.5, 0.9) # 设置绘制的颜色为蓝紫色。
        glBegin(GL_TRIANGLES) # 定义绘制模式为三角形。
        for face in self.F: #  按照网格定义的三角形面逐个绘制整个模型。
            for i in face:
                glVertex3fv(self.points[i])
        glEnd()

        # 绘制交互点
        glColor3f(0.0, 0.0, 0.0) # 设置绘制的颜色为黑色。
        glPointSize(2.5) # 设置点的大小为 2.5 像素
        glBegin(GL_POINTS)
        for point in self.points: # 将模型中的顶点可视化，便于交互和调试。
            glVertex3fv(point)
        glEnd()

        # 绘制折叠线
        if len(self.fold_line_points) == 2: # 只有当 self.fold_line_points 有两个点时，才绘制折叠线。
            glColor3f(0.5, 0.5, 0.5) # 设置折叠线颜色为淡紫色
            glLineWidth(2.0) # 设置线宽为 2 像素。
            glBegin(GL_LINES) # 定义绘制模式为线，绘制一条从 fold_line_points[0] 到 fold_line_points[1] 的直线。
            glVertex3fv(self.fold_line_points[0])
            glVertex3fv(self.fold_line_points[1])
            glEnd()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.mode == "view": # 当前模式是 "view"，表示用户正在调整模型视图（例如旋转或缩放模型）
            self.lastPos = event.pos() # 记录鼠标按下时的位置 event.pos()，用于后续的鼠标拖动事件（例如旋转模型），计算鼠标移动的偏移量
        elif self.mode == "fold": #  当前模式是 "fold"，表示用户正在选择折叠相关的点或线
            self.selected_point = self.get_nearest_point(event) # 通过鼠标点击的屏幕位置，找到模型中距离鼠标最近的顶点。
            self.lastPos = event.pos() # 记录鼠标按下时的位置

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.mode == "view":
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y() # 计算鼠标位移
            self.rotation_x += dy # 更新模型旋转
            self.rotation_y += dx
            self.update() # 刷新视图，触发重新绘制模型以反映旋转的变化
            self.lastPos = event.pos() # 记录新鼠标位置，用于计算下一次移动的偏移量
        elif self.mode == "fold" and self.selected_point is not None:
            dx = event.x() - self.lastPos.x() # 计算鼠标水平方向位移
            self.fold_angle += dx * 0.1 # 更新折叠角度，每单位水平位移增加 0.1 度折叠角度。
            self.fold_angle = max(0, min(self.fold_angle, 180))  # 限制折叠角度范围为 [0, 180], 似乎没有发挥效果
            self.apply_fold_transformation() # 根据新的折叠角度对模型进行变形。
            self.update() # 刷新视图，触发重新绘制模型以反映折叠的变化
            self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.selected_point = None

    def wheelEvent(self, event):
        """鼠标滚轮实现视角缩放"""
        delta = event.angleDelta().y() # 获取滚轮滚动值
        if delta > 0:
            self.camera_distance -= 0.5
        else:
            self.camera_distance += 0.5
        self.camera_distance = max(2, self.camera_distance)
        self.update() # 刷新视图，触发重新绘制观察视角拉近或拉远

    def get_nearest_point(self, event):
        """获取最近的顶点索引"""
        x, y = self.get_3d_coords(event) # 获取鼠标的3D坐标，将鼠标在窗口中的 2D 坐标（像素值）转换为 3D OpenGL 坐标。
        distances = [np.linalg.norm(point - np.array([x, y, 0])) for point in self.points] # 计算鼠标与每个顶点的距离
        return np.argmin(distances) # 找出距离最小的顶点索引

    def get_3d_coords(self, event):
        """将鼠标位置转换为OpenGL坐标"""
        width, height = self.width(), self.height() # 计算并返回鼠标在 OpenGL 世界中对应的 2D 坐标 (x, y)
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
            print(point1,"   ")
            print(point2,"   ")
            if self.is_different_edges(idx1, idx2): # 不在同一条外层边缘上
                self.fold_line_points = [point1.tolist(), point2.tolist()] # 把选择的两个点加入数组中为绘制折线做准备
                print(f"Random fold line: {self.fold_line_points}")
                self.calculate_fold_region() #
                break

    def is_different_edges(self, idx1, idx2):
        """检查两个点是否在不同边上"""
        if abs(idx1 - idx2) == 1 or {idx1, idx2} == {0, len(self.boundary_vertices) - 1}:
            return False
        return True
    # 用于将折线方向向量左侧的所有点加入折叠区域的顶点
    def calculate_fold_region(self):
        """划分折叠区域"""
        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1]) # 提取折叠线的两个端点
        line_vector = line_end - line_start # 计算折叠线方向向量

        self.fold_region = [] # 初始化折叠区域
        for i, vertex in enumerate(self.points): # 遍历所有顶点：self.points
            vertex_vector = np.array(vertex) - line_start # 计算当前顶点相对折叠线的向量
            if np.dot(np.cross(line_vector, vertex_vector), [0, 0, 1]) > 0:  # 经过一段计算得到的标量的符号决定顶点相对于折叠线的位置关系，正值：顶点在折叠线的“左侧”（按右手法则）。
                self.fold_region.append(i) # 只对折线方向向量左侧折叠

    # 用于在模型中对指定折叠区域的顶点应用变形。其目的是将属于折叠区域的顶点按照指定的折叠线（轴）和折叠角度进行旋转，模拟折叠效果
    def apply_fold_transformation(self):
        """应用折叠变形"""
        if self.fold_region is not None and len(self.fold_line_points) == 2: # 有折线，被折叠区域有点
            line_start = np.array(self.fold_line_points[0])
            line_end = np.array(self.fold_line_points[1]) # 折线的起点和终点
            fold_axis = (line_end - line_start) / np.linalg.norm(line_end - line_start) # 计算出折叠线的方向轴

            for i in self.fold_region: # 遍历被折叠区域的所有顶点
                vertex = np.array(self.points[i])
                translated = vertex - line_start # 将顶点平移到折叠线起点为原点的局部坐标系中，便于绕旋转轴计算，translated是三维坐标点
                rotated = self.rotate_around_axis(translated, fold_axis, np.radians(self.fold_angle)) # 将点 translated 绕指定的折叠轴 fold_axis 按 fold_angle 的弧度值旋转。
                self.points[i] = rotated + line_start # 旋转后的顶点坐标仍在局部坐标系中。将旋转结果平移回全局坐标系。

    # 用于将一个点绕任意轴旋转指定角度，返回旋转后的点的坐标。
    def rotate_around_axis(self, point, axis, angle):
        """绕任意轴旋转"""
        axis = axis / np.linalg.norm(axis) # 将输入的旋转轴 axis 归一化，确保其为单位向量。
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle) # 输入的旋转角度 angle（弧度制）计算对应的余弦值和正弦值。
        cross_prod = np.cross(axis, point) # 计算轴向量与点向量的叉积。
        dot_prod = np.dot(axis, point) # 计算轴向量与点向量的点积。

        rotated = (point * cos_theta +
                   cross_prod * sin_theta +
                   axis * dot_prod * (1 - cos_theta)) # 应用 Rodrigues 旋转公式， 计算旋转后的点坐标
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
app = QApplication([]) # 初始化应用对象
window = MainWindow() # 自定义的主窗口类，继承自 QMainWindow
window.setWindowTitle("Interactive Origami Viewer")
window.resize(900, 900) # 调整窗口尺寸大小
window.show()
app.exec_() # 启动 Qt 的事件循环，进入应用的主循环。持续监听用户事件（如鼠标、键盘）并分发到对应的事件处理函数。
