import sys
import igl
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random



class OrigamiWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OrigamiWidget, self).__init__(parent) # 调用父类的 __init__ 进行初始化
        self.mode = "view"
        self.points = []  # 存储场景中的顶点数据。
        self.selected_point = None  # 当用户点击某个点时，程序会找到合法折线点中距离鼠标最近的点，并将其索引赋值给 self.selected_point。
        self.camera_distance = 11.0  # 控制相机视角的远近。
        self.lastPos = None  # 鼠标拖动时记录上一次鼠标位置，用于计算旋转角度。
        self.rotation_x = 0  # X轴旋转
        self.rotation_y = 0  # Y轴旋转
        self.rotation_z = 0  # Z轴旋转

        # 折叠功能
        self.fold_line_points = []  # 记录折叠线的两个端点
        self.fold_region = None  # 记录需要折叠的区域
        self.fold_angle = 0  # 折叠角度
        self.fold_direction = 1  # 折叠方向
        self.fold_point_selection = 0  # 记录折叠线选取状态（0/1/2）

        # 添加颜色和动画相关属性
        self.colors = [  # 预定义了 5 种颜色，后续用于网格着色。
            [1.0, 0.7, 0.7],  # 粉红色
            [0.7, 1.0, 0.7],  # 浅绿色
            [0.7, 0.7, 1.0],  # 浅蓝色
            [1.0, 1.0, 0.7],  # 黄色
            [0.5, 0.5, 0.5],  # 灰色
        ]
        self.current_color = 0  # 当前使用的颜色索引。

        # 初始化矩形模型
        # 生成初始的矩形网格数据（顶点 V 和面 F）。
        self.V, self.F = self.init_refined_rectangle(n=250)  # 降低密度使其更容易操作
        self.points = self.V.tolist()  # 转为python列表
        self.boundary_vertices = igl.boundary_loop(self.F) # igl计算网格边界上的顶点索引，用于折叠操作。

        self.vao = None  # 顶点数组对象
        self.vbo = None  # 顶点缓冲对象
        self.ibo = None  # 索引缓冲对象

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
        glTranslatef(0.0, 0.0, -self.camera_distance)   # Z 轴负方向 平移整个场景，使其远离观察者。
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)  # self.rotation_x：旋转角度（度）。(1.0, 0.0, 0.0)：表示围绕 X 轴 旋转。
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)

        # 更新顶点数据（如果需要）
        if self.fold_region is not None:
            vertices = np.array(self.points, dtype=np.float32)
            # 绑定 VBO，表示接下来的操作作用于 self.vbo
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            # 确保当修改顶点时，GPU 能够及时接收到新的顶点数据，以便渲染时显示这些变化。
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        # 绘制网格
        if self.vao is not None:
            glBindVertexArray(self.vao) # 绑定当前的顶点数组对象（VAO）

            # 使用当前选择的颜色
            current_rgb = self.colors[self.current_color]
            glColor3f(*current_rgb)
            # 使用索引缓冲区（EBO）来绘制三角形网格
            # GL_TRIANGLES 表示 OpenGL 使用三角形绘制，len(self.F.flatten()) 表示需要绘制的三角形数目。
            glDrawElements(GL_TRIANGLES, len(self.F.flatten()), GL_UNSIGNED_INT, None)
            # 渲染完成后解绑
            glBindVertexArray(0)

        # 绘制折叠线和控制元素
        self.draw_fold_elements()

    def draw_fold_elements(self):
        """优化的折线元素绘制"""
        if not self.fold_line_points:  # 如果没有折线点，直接返回
            return

        # 使用顶点数组一次性绘制所有点
        if len(self.fold_line_points) >= 1:  # 如果折线点的数量大于等于 1，开始绘制折线。
            glDisable(GL_LIGHTING)  # 临时禁用光照以提高性能

            # 绘制第一个点
            glColor3f(1.0, 1.0, 1.0)  # 白色
            glPointSize(10.0)
            glBegin(GL_POINTS)
            glVertex3fv(self.fold_line_points[0])  # 绘制
            glEnd()  # 绘制结束

            # 如果有第二个点，绘制该点并绘制连接这两个点的线段。
            if len(self.fold_line_points) == 2:
                # 绘制第二个点
                glColor3f(1.0, 1.0, 1.0)  # 白色
                glBegin(GL_POINTS)
                glVertex3fv(self.fold_line_points[1])
                glEnd()

                # 绘制连接线
                glColor3f(0.5, 0.5, 0.5)  # 灰色
                glLineWidth(0.1)
                glBegin(GL_LINES)  # 告诉 OpenGL 接下来的绘制是线段。
                glVertex3fv(self.fold_line_points[0])
                glVertex3fv(self.fold_line_points[1])  # 将两个点传递给 OpenGL，绘制连接这两个点的线段。
                glEnd()  # 绘制结束

            glEnable(GL_LIGHTING)  # 重新启用光照

    # 存在显示问题，暂时未使用
    def draw_fold_direction_arrow(self):
        """优化的折叠方向箭头绘制"""
        if len(self.fold_line_points) != 2:
            return

        # 计算箭头位置
        start = np.array(self.fold_line_points[0])
        end = np.array(self.fold_line_points[1])
        mid = (start + end) / 2
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:  # 避免除以零
            return

        # 计算箭头方向
        direction = direction / length
        normal = np.array([0, 0, 1])
        arrow_direction = np.cross(direction, normal)
        if self.fold_direction < 0:
            arrow_direction = -arrow_direction

        # 计算箭头顶点 - 更大更明显的红色箭头
        arrow_size = 0.3  # 更大的箭头
        arrow_tip = mid + arrow_direction * arrow_size
        arrow_base1 = mid + arrow_direction * (arrow_size * 0.5) - direction * (arrow_size * 0.3)
        arrow_base2 = mid + arrow_direction * (arrow_size * 0.5) + direction * (arrow_size * 0.3)

        # 一次性绘制箭头
        glColor3f(1.0, 0.0, 0.0)  # 鲜红色
        glBegin(GL_TRIANGLES)
        glVertex3fv(arrow_tip)
        glVertex3fv(arrow_base1)
        glVertex3fv(arrow_base2)
        glEnd()

        # 添加箭头边缘轮廓以增强可见性
        glLineWidth(2.0)
        glColor3f(1.0, 0.3, 0.3)  # 稍浅的红色边缘
        glBegin(GL_LINE_LOOP)
        glVertex3fv(arrow_tip)
        glVertex3fv(arrow_base1)
        glVertex3fv(arrow_base2)
        glEnd()

    def mousePressEvent(self, event):
        """鼠标按下事件处理"""
        if self.mode == "view":  # 检查当前模式是否是 "view"（视图模式）。
            self.lastPos = event.pos()  # 记录鼠标点击位置 self.lastPos = event.pos()，用于后续的拖动或旋转操作。
        elif self.mode == "fold":  # 如果当前模式是 "fold"（折叠模式），进入折叠处理逻辑。
            if event.button() == Qt.RightButton and self.fold_point_selection == 2:  # 如果鼠标右键被按下 (event.button() == Qt.RightButton) 并且已经选择了 2 个折叠点 (self.fold_point_selection == 2)，则执行折叠方向的切换
                # 右键切换折叠方向（切换主折叠面），但不显示箭头
                self.fold_direction *= -1
                self.fold_angle = 0  # 重置折叠角度
                self.calculate_fold_region()  # 重新计算折叠区域
                self.update()
                return
            # 如果折叠点数量小于 2（表示用户还没有选完折叠线的两个端点），则执行点选择逻辑。
            if self.fold_point_selection < 2:
                # 获取鼠标点击位置最近的边界点及其索引。
                nearest_point_idx, nearest_point = self.get_nearest_boundary_point(event)
                if nearest_point_idx is not None and nearest_point is not None:
                    if self.fold_point_selection == 0:  # 如果是第一个点
                        self.fold_line_points = [nearest_point.tolist()]  # 作为折叠线的起点
                        self.fold_point_selection = 1  # 表示已经选择了第一个点。
                    else:  # selecting second point
                        try:
                            first_point_idx = self.boundary_vertices.tolist().index(
                                self.get_point_index(self.fold_line_points[0])
                            )
                            # 检查两个点是否在不同的边界上
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
            # 记录鼠标点击位置 (self.lastPos = event.pos())，用于后续拖动或其他操作。
            self.lastPos = event.pos()
            self.update()
    # 处理鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.mode == "view":
            # event.x() 和 event.y() 是当前鼠标位置，self.lastPos 是上次鼠标位置，计算鼠标在x和y轴的移动量
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            self.rotation_x += dy  # 水平方向的移动控制y轴的旋转
            self.rotation_y += dx
            self.update()
            self.lastPos = event.pos()
        # 在折叠模式且已经选择了 2 个折叠点
        elif self.mode == "fold" and self.fold_point_selection == 2 and self.selected_point is not None:
            dx = event.x() - self.lastPos.x() # 根据水平位移进行旋转
            # 增加旋转速度系数到2.0，使旋转更加灵敏
            rotation_speed = 2.0
            # 直接根据鼠标移动设置旋转角度
            self.fold_angle = dx * rotation_speed
            self.fold_angle = max(-180, min(180, self.fold_angle))  # 限制最大旋转角度
            # self.fold_angle = max(-80, min(80, self.fold_angle))  # 限制最大旋转角度
            self.apply_fold_transformation()  # 计算折叠变换并更新模型，使折叠的部分正确变形。
            self.update()
            self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        self.selected_point = None
    # 处理鼠标滚轮事件
    def wheelEvent(self, event):
        delta = event.angleDelta().y()  # 获取滚轮滚动的值
        if delta > 0:
            self.camera_distance -= 0.5  # 向上滚动，相机拉近
        else:
            self.camera_distance += 0.5
        self.camera_distance = max(2, self.camera_distance)  # 最小距离 = 2，防止缩放过度。穿模
        self.update()

    def get_nearest_point(self, event):
        x, y = self.get_3d_coords(event)  # 将鼠标的 屏幕坐标 (2D) 转换为 世界坐标 (3D)，涉及 鼠标投射到 3D 空间 的计算
        # 遍历计算每个顶点到鼠标点击点的距离，后续优化算法逻辑
        distances = [np.linalg.norm(point - np.array([x, y, 0])) for point in self.points]
        return np.argmin(distances)  # 获取最近点的索引，self.points中的对应索引的点就是目标点

    def get_3d_coords(self, event):
        """将鼠标位置转换为OpenGL坐标"""
        width, height = self.width(), self.height()  # 获取当前窗口的像素尺寸。
        aspect_ratio = width / height  # 计算宽高比（长宽比）

        # 基础缩放因子
        base_scale = 5.0

        # 根据窗口尺寸调整缩放因子
        if aspect_ratio > 1.0:  # 宽大于高
            x_scale = base_scale * aspect_ratio
            y_scale = base_scale
        else:  # 高大于宽
            x_scale = base_scale
            y_scale = base_scale / aspect_ratio

        # 计算鼠标在 OpenGL 坐标中的位置
        # 将 x 从 像素坐标（0~width） 映射到 OpenGL 坐标（-1~1）
        x = (event.x() / width * 2.0 - 1.0) * x_scale  #  乘以2再减一映射到（-1~1）
        y = (1.0 - event.y() / height * 2.0) * y_scale
        # 尝试经过相机旋转矫正的 3D 坐标，但是每次折叠已经初始化旋转角度了，以下代码就没有作用，实际得到的就是上面的x与y坐标
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
    # 在鼠标点击位置附近找到最近的边界点，优化算法
    def get_nearest_boundary_point(self, event):
        """获取最近的边界点 - 向量化优化版本，提高精度"""
        x, y = self.get_3d_coords(event)  # 计算鼠标的 3D 位置
        cursor_pos = np.array([x, y])

        # 一次性获取所有边界点的坐标
        boundary_points = self.V[self.boundary_vertices][:, :2]

        # 计算所有点到鼠标位置的距离，np.linalg.norm计算欧氏距离
        distances = np.linalg.norm(boundary_points - cursor_pos, axis=1)

        # 检查角点 - 优先选择四个角点
        n = int(np.sqrt(len(self.V))) - 1
        corner_indices = np.array([0, n, n * (n + 1), (n + 1) ** 2 - 1])

        # 计算自适应阈值 - 根据窗口大小和相机距离调整
        width, height = self.width(), self.height()
        min_dimension = min(width, height)
        # 自适应阈值 - 随着窗口变大或相机拉远而增加
        adaptive_threshold = 0.5 * (1.0 + self.camera_distance / 10.0) * (1000.0 / min_dimension)
        '''相机距离项 (1 + camera_distance / 10)：
        目的：相机越远，模型视觉尺寸越小，需要更大的点击容差。
        示例：当camera_distance = 20时，该项值为3.0（原值2.1）。
        窗口尺寸项(1000.0 / min_dimension)：
        目的：窗口越小，像素密度越高，需要更大的点击容差。
        示例：4K屏幕（3840x2160）下min_dimension = 2160，该项值为0.46。
        基础系数 0.5：
        调节建议：若需要更高精度，可减小为
        0.3；若需要宽松操作，可增大至
        1.0。'''
        # 找出在boundary_vertices中的角点索引
        corner_mask = np.isin(self.boundary_vertices, corner_indices)
        if np.any(corner_mask):
            corner_distances = distances[corner_mask]
            min_corner_dist = np.min(corner_distances)
            if min_corner_dist < adaptive_threshold:  # 使用自适应阈值
                idx = np.where(corner_mask)[0][np.argmin(corner_distances)]
                return idx, self.V[self.boundary_vertices[idx]].copy()

        # 如果没有找到合适的角点，返回最近的边界点
        nearest_idx = np.argmin(distances)
        min_dist = distances[nearest_idx]

        # 提高边界点选择的精度，使用自适应阈值
        edge_threshold = adaptive_threshold * 2.0  # 边缘点可以有更大的阈值
        if min_dist < edge_threshold:
            return nearest_idx, self.V[self.boundary_vertices[nearest_idx]].copy()

        return None, None

    def get_point_index(self, point):
        """获取点在V数组中的索引 - 提高精度"""
        # 使用向量化操作提高性能和精度
        points = np.array(self.V)
        point_array = np.array(point)

        # 计算所有点到目标点的距离
        distances = np.linalg.norm(points - point_array, axis=1)

        # 找到最近的点
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # 计算自适应阈值 - 根据窗口大小和相机距离调整
        width, height = self.width(), self.height()
        min_dimension = min(width, height)
        # 更精确的自适应阈值，用于精确点匹配
        adaptive_threshold = 1e-4 * (1.0 + self.camera_distance / 10.0) * (1000.0 / min_dimension)

        # 使用更严格的阈值确保精确匹配
        if min_dist < adaptive_threshold:
            return min_idx

        # 如果没有找到精确匹配，使用传统方法作为备选
        # 使用自适应的相对和绝对容差
        rel_tol = 1e-5 * (1.0 + self.camera_distance / 10.0)
        abs_tol = 1e-5 * (1.0 + self.camera_distance / 10.0)

        for i, v in enumerate(self.V):
            if np.allclose(v, point, rtol=rel_tol, atol=abs_tol):
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
    # 通过叉积正负，计算需要被折叠的区域
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

        # 计算叉积，一次性处理所有点
        cross_products = np.cross(line_vector, vertex_vectors)
        dots = np.einsum('ij,j->i', cross_products, [0, 0, 1]) * self.fold_direction

        # 使用布尔索引找出所有需要折叠的点，叉积大于的0的点定义为被折叠点，右侧
        self.fold_region = np.where(dots > 0)[0].tolist()
    # 围绕折叠轴旋转某个区域的顶点，实现仿真折叠的效果。
    def apply_fold_transformation(self):
        """使用向量化操作应用折叠变换"""
        if self.fold_region is None or len(self.fold_line_points) != 2:  # 完整性检查：存在被折叠区域且折线存在
            return
        # 获取折线的起点和终点
        line_start = np.array(self.fold_line_points[0])
        line_end = np.array(self.fold_line_points[1])
        # 计算折叠轴的单位向量
        fold_axis = (line_end - line_start) / np.linalg.norm(line_end - line_start)

        # 将所有需要变换的被折叠面的点转换为numpy数组
        vertices = np.array([self.points[i] for i in self.fold_region])

        # 批量平移，把所有点移动到以折叠轴起点 line_start 为原点的坐标系。本质是减去起始点
        translated = vertices - line_start

        # 批量旋转
        angle = np.radians(self.fold_angle)  # 旋转角度转弧度
        # 将点集合translated围绕单位向量旋转angle弧度
        rotated = self.rotate_points_around_axis(translated, fold_axis, angle)

        # 批量平移回原位置，本质是加上起始点
        result = rotated + line_start

        # 更新点的位置
        for i, idx in enumerate(self.fold_region):
            self.points[idx] = result[i].tolist()

    def rotate_points_around_axis(self, points, axis, angle):
        """批量旋转多个点"""
        axis = axis / np.linalg.norm(axis)  # 计算旋转轴 axis 的单位向量
        cos_theta = np.cos(angle)  # 旋转角度余弦值
        sin_theta = np.sin(angle)  # 旋转角度正弦值

        # 计算叉积和点积（批量）
        cross_prods = np.cross(points, axis)
        dot_prods = np.einsum('ij,j->i', points, axis)

        # 使用罗德里格旋转公式进行批量计算
        rotated = (points * cos_theta +
                   cross_prods * sin_theta +
                   np.outer(dot_prods * (1 - cos_theta), axis))
        # 旋转后的点坐标
        return rotated
    # 初始化一个细化的矩形网格模型
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
    # 调整 OpenGL 渲染窗口大小
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100)  # gluPerspective 设置透视投影，以保证物体在屏幕上具有正确的深度感。
        glMatrixMode(GL_MODELVIEW)

# 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.glWidget = OrigamiWidget() # 3D 折纸绘制窗口
        self.initUI() # 负责界面布局和UI组件创建

    def initUI(self):
        # 创建主窗口部件
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget) #  采用水平布局 (QHBoxLayout)
        self.setCentralWidget(main_widget) # 设置为 QMainWindow 的中心部件

        # 左侧控制面板
        control_panel = QFrame()
        control_panel.setFixedWidth(280)  # 增加宽度从250到280
        # #FFB6C1（浅粉色）到 #87CEEB（天蓝色）
        # 圆角 + 白色边框 让 UI 更加美观。
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
        panel_layout = QVBoxLayout() # 垂直排列子控件，从上到下依次放置。
        panel_layout.setAlignment(Qt.AlignTop) # 紧贴顶部排列
        panel_layout.setContentsMargins(20, 30, 20, 30) # 布局的边距（margins）
        control_panel.setLayout(panel_layout) # 将 panel_layout布局 绑定到 control_panel

        # 标题
        title = QLabel("折纸模拟器") # 显示标题
        # Comic Sans MS 字体，字体大小 28px，居中对齐。
        title.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font: bold 28px 'Comic Sans MS';
                qproperty-alignment: AlignCenter;
                margin-bottom: 20px;
            }
        """)
        panel_layout.addWidget(title)
# arisu
        # 模式按钮
        # 包括文字、颜色、点击事件
        mode_buttons = [
            ("🔍 观察模式", "#FF69B4", self.set_view_mode),
            ("✂️ 折叠模式", "#00BFFF", self.set_fold_mode),
            ("🔄 重置", "#F44336", self.reset_model),
            ("📏 显示网格", "#9C27B0", self.toggle_grid),
            ("🎨 换色模式", "#FFD700", self.change_paper_color),
            ("❓ 帮助", "#FFA500", self.show_help),
            ("↩️ 撤销", "#66CCCC", self.undo),
            ("🔍 辅助点", "#FF0000", self.toggle_auxiliary_points),
        ]
        # 动态创建所有按钮
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
                margin-bottom: 10px;
                min-height: 50px;
            }
        """)
        self.hint_label.setWordWrap(True) # 允许自动换行
        self.hint_label.setMinimumWidth(200) # 限制最小宽度
        panel_layout.addWidget(self.hint_label)

        # 添加装饰元素
        # try:
        #     decoration = QLabel()
        #     decoration.setPixmap(QPixmap("origami_decoration.png").scaled(200, 200, Qt.KeepAspectRatio))
        #     panel_layout.addWidget(decoration, alignment=Qt.AlignCenter)
        # except:
        #     pass  # 如果图片不存在，跳过
        panel_layout.addStretch() # 弹性空白区域

        # 右侧OpenGL窗口区域
        gl_container = QFrame()
        # 背景色 #F0F8FF（淡蓝色）,边框 3px 虚线 #87CEFA（浅蓝色）
        gl_container.setStyleSheet("""
            QFrame {
                background: #F0F8FF;
                border-radius: 20px;
                border: 3px dashed #87CEFA;
                margin: 10px;
            }
        """)
        gl_layout = QVBoxLayout(gl_container) # 垂直布局管理器
        gl_layout.addWidget(self.glWidget) # 将 OpenGL 视图添加到 gl_container，使其填充整个 QFrame。

        # 添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(gl_container, stretch=1)
    # 观察模式
    def set_view_mode(self):
        self.glWidget.mode = "view" # 触发后进入 观察模式
        self.hint_label.setText("✨ 转动鼠标欣赏作品") # 切换提示标签的字体
    # 折叠模式
    def set_fold_mode(self):
        # 重置视角和旋转角度
        self.glWidget.camera_distance = 11.0  # 重置相机距离
        self.glWidget.rotation_x = 0  # 重置X轴旋转
        self.glWidget.rotation_y = 0  # 重置Y轴旋转
        self.glWidget.rotation_z = 0  # 重置Z轴旋转

        # 设置折叠模式
        # 初始化折叠点和折叠角度
        self.glWidget.mode = "fold"
        self.glWidget.fold_line_points = []
        self.glWidget.fold_point_selection = 0
        self.glWidget.fold_region = None
        self.glWidget.fold_angle = 0
        self.glWidget.update()
        self.hint_label.setText("✨ 点两点画折线\n拖动鼠标来折叠")
    # 颜色切换
    def change_paper_color(self):
        self.glWidget.current_color = (self.glWidget.current_color + 1) % len(self.glWidget.colors)
        self.glWidget.update()
        self.hint_label.setText("✨ 换个颜色\n创作更美作品")
    # 帮助消息框
    def show_help(self):
        # 创建自定义消息框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("折纸小帮手")

        # 使用与main.py相同风格的文本
        help_text = (
            "欢迎来到折纸模拟器！✨\n\n"
            "🔍 观察模式：\n"
            "   左键拖动旋转视角\n"
            "   右键拖动缩放视角\n\n"
            "✂️ 折叠模式：\n"
            "   左键点击两点确定折线\n"
            "   右键点击改变折叠方向\n"
            "   拖动鼠标控制折叠角度\n\n"
            "🔄 重置：\n"
            "   点击按钮重置模型到初始状态\n\n"
            "📏 显示网格：\n"
            "   点击按钮切换网格显示\n\n"
            "🎨 换色模式：\n"
            "   点击按钮换不同的纸张颜色\n\n"
            "↩️ 撤销：\n"
            "   点击按钮撤销上一步操作\n\n"
            "🔍 辅助点：\n"
            "   点击按钮切换辅助点显示\n"
            "   红色点标记四个角点位置\n"
            "   便于精确选择折线位置\n\n"
            "开始创作你的魔法折纸吧！💫"
        )

        # 设置消息框内容
        msg_box.setText(help_text)

        # 设置消息框字体 - 使用Comic Sans MS保持卡通风格
        font = QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        msg_box.setFont(font)

        # 设置图标，显示 ℹ️（信息图标）
        msg_box.setIcon(QMessageBox.Information)

        # 显示消息框，阻塞（模态）运行消息框，直到用户关闭。
        msg_box.exec_()

    def undo(self):
        self.hint_label.setText("✨ 撤销功能尚未实现")

    def toggle_auxiliary_points(self):
        self.hint_label.setText("✨ 辅助点功能尚未实现")

    def reset_model(self):
        """重置模型到初始状态"""
        self.hint_label.setText("✨ 重置功能尚未实现")

    def toggle_grid(self):
        """切换网格显示"""
        self.hint_label.setText("✨ 网格显示功能尚未实现")


if __name__ == "__main__":
    app = QApplication(sys.argv) # 管理 应用程序事件循环

    # 设置全局字体
    font = QFont()
    font.setFamily("Comic Sans MS")
    font.setPointSize(12)
    app.setFont(font)
    # 实例化 MainWindow，用于创建主应用窗口
    window = MainWindow()
    window.setWindowTitle("折纸交互模拟系统")

    # 设置窗口大小为1200 * 800
    window.resize(1200, 800)
    window.show()

    # 设置全屏显示
    # window.showMaximized()  # 最大化窗口
    # 或者使用下面的代码完全全屏（无边框）
    # window.showFullScreen()

    # 启动应用程序事件循环，开始处理用户 输入事件（如鼠标点击、键盘输入等）。
    app.exec_()