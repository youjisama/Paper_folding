import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# 测试高密度用例
class HighDensityMesh(QOpenGLWidget):
    def __init__(self, parent=None):
        super(HighDensityMesh, self).__init__(parent)
        self.vertex_array = None
        self.index_array = None
        self.VAO = None
        self.VBO = None
        self.EBO = None
        self.shader_program = None
        self.mesh_size = 100  # 网格密度（可调）
        self.zoom = 1.0       # 缩放因子
        self.rotation = [0, 0]  # 旋转角度
        self.last_mouse_pos = None

    def initializeGL(self):
        """初始化 OpenGL 环境"""
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        # 设置绘制模式为线框
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # 编译着色器
        self.shader_program = self.compile_shader()

        # 初始化网格
        self.generate_mesh(self.mesh_size)

        # 设置缓冲区对象
        self.setup_buffers()

    def resizeGL(self, w, h):
        """调整窗口大小"""
        glViewport(0, 0, w, h)

    def paintGL(self):
        """渲染网格"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program)

        # 设置变换矩阵
        projection = np.identity(4, dtype=np.float32)
        modelview = np.identity(4, dtype=np.float32)

        # 应用缩放
        for i in range(3):
            projection[i][i] = self.zoom

        # 应用旋转
        modelview = np.dot(modelview, self.rotation_matrix(self.rotation[0], [1, 0, 0]))
        modelview = np.dot(modelview, self.rotation_matrix(self.rotation[1], [0, 1, 0]))

        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "modelview"), 1, GL_TRUE, modelview)

        # 绘制网格
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.index_array), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def generate_mesh(self, size):
        """动态生成高密度网格"""
        self.vertex_array = []
        self.index_array = []

        # 顶点生成
        for y in range(size + 1):
            for x in range(size + 1):
                self.vertex_array.extend([x / size - 0.5, y / size - 0.5, 0.0])  # 网格顶点坐标

        # 索引生成（每两个三角形组成一个方格）
        for y in range(size):
            for x in range(size):
                top_left = y * (size + 1) + x
                top_right = top_left + 1
                bottom_left = top_left + size + 1
                bottom_right = bottom_left + 1

                self.index_array.extend([top_left, bottom_left, top_right,  # 第一个三角形
                                         top_right, bottom_left, bottom_right])  # 第二个三角形

        self.vertex_array = np.array(self.vertex_array, dtype=np.float32)
        self.index_array = np.array(self.index_array, dtype=np.uint32)

    def setup_buffers(self):
        """设置 VAO、VBO 和 EBO"""
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_array.nbytes, self.vertex_array, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.index_array.nbytes, self.index_array, GL_STATIC_DRAW)

        # 定义顶点属性指针
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * self.vertex_array.itemsize, None)
        glEnableVertexAttribArray(0)

        glBindVertexArray(0)

    def compile_shader(self):
        """编译着色器程序"""
        vertex_src = """
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform mat4 projection;
        uniform mat4 modelview;
        void main() {
            gl_Position = projection * modelview * vec4(position, 1.0);
        }
        """
        fragment_src = """
        #version 330 core
        out vec4 outColor;
        void main() {
            outColor = vec4(0.8, 0.8, 0.8, 1.0);
        }
        """
        return compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                              compileShader(fragment_src, GL_FRAGMENT_SHADER))

    def rotation_matrix(self, angle, axis):
        """生成旋转矩阵"""
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        axis = np.array(axis, dtype=np.float32)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        return np.array([
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s, 0],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.rotation[0] += dy * 0.5
            self.rotation[1] += dx * 0.5
            self.update()
        self.last_mouse_pos = event.pos()

    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() / 1200.0
        self.zoom = max(0.1, self.zoom)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("High-Density Paper Mesh")
        self.resize(800, 600)
        self.mesh_widget = HighDensityMesh(self)
        self.setCentralWidget(self.mesh_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
