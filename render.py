import os
import time
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
from skimage.io import imsave
from OpenGL.EGL import *
from OpenGL.GL import *


class FaceRenderer:
    def __init__(self, triangles, src_size=(256, 256), far=512, out_size=(256, 256)):
        w, h = src_size
        ortho = np.array([
            [2./w,   0., 0., -1.],
            [0., 2./h,   0., -1.],
            [0., 0., -2./far, 1.],
            [0., 0., 0.,      1.],
        ], dtype=np.float32)
        self._init_egl()
        self._init_program()

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles, GL_STATIC_DRAW)

        glEnable(GL_DEPTH_TEST)

        render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, render_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, *out_size)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_buffer)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        glViewport(0, 0, *out_size)
        glClearDepth(1.0)
        glUniformMatrix4fv(1, 1, GL_TRUE, ortho)

        self.out_size = out_size
        self.nr_vertices = triangles.size

    def draw_batch(self, pos_batch):
        glBufferData(GL_ARRAY_BUFFER, pos_batch, GL_STREAM_DRAW)
        pos_size_per_img = pos_batch[0].size
        img = np.empty((len(pos_batch), self.out_size[1], self.out_size[0]), dtype=np.float32)
        for i in range(len(pos_batch)):
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(i * pos_size_per_img * 4))
            glClear(GL_DEPTH_BUFFER_BIT)
            glDrawElements(GL_TRIANGLES, self.nr_vertices, GL_UNSIGNED_INT, None)
            glReadPixels(0,0,self.out_size[0],self.out_size[1], GL_DEPTH_COMPONENT, GL_FLOAT, array=img[i])
        return img

    def _init_egl(self):
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        major = EGLint()
        minor = EGLint()
        eglInitialize(display, major, minor)
        attrib_list = [
            EGL_CONFIG_CAVEAT, EGL_NONE,
            EGL_CONFORMANT, EGL_OPENGL_BIT,
            EGL_NONE,
        ]
        attrib_list = arrays.GLintArray.asArray(attrib_list)
        configs = EGLConfig()
        num_config = EGLint()
        eglChooseConfig(display, attrib_list, configs, 1, num_config)
        eglBindAPI(EGL_OPENGL_API)
        ctx = eglCreateContext(display, configs, EGL_NO_CONTEXT, None)
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)

    def _init_program(self):
        vertexsource = '''#version 430
in  vec3 in_Position;

layout(location = 1) uniform mat4 trans;

void main(void) {
    gl_Position = trans * vec4(in_Position, 1.0);
}
'''
        fragmentsource = '''#version 430
// It was expressed that some drivers required this next line to function properly
precision highp float;

out vec4 gl_FragColor;

void main(void) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
'''
        vertexshader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexshader, vertexsource)
        glCompileShader(vertexshader)
        # print(glGetShaderInfoLog(vertexshader))
        assert glGetShaderiv(vertexshader, GL_COMPILE_STATUS)

        fragmentshader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentshader, fragmentsource)
        glCompileShader(fragmentshader)
        assert glGetShaderiv(fragmentshader, GL_COMPILE_STATUS)

        shaderprogram = glCreateProgram()
        glAttachShader(shaderprogram, vertexshader)
        glAttachShader(shaderprogram, fragmentshader)
        glBindAttribLocation(shaderprogram, 0, "in_Position")
        glLinkProgram(shaderprogram)
        assert glGetProgramiv(shaderprogram, GL_LINK_STATUS)
        glUseProgram(shaderprogram)

def main():
    pos = np.load('test_pos.npy').astype(np.float32)
    face_ind = np.loadtxt('Data/uv-data/face_ind.txt', dtype=np.int32)
    triangles = np.loadtxt('Data/uv-data/triangles.txt', dtype=np.int32)
    # pos = pos.reshape((-1,3))[face_ind]
    triangles = face_ind[triangles]
    print(pos[:, 2].max())
    print(pos[:, 2].min())

    r = FaceRenderer(triangles)
    img = r.draw_batch(pos)

    m = img.min(axis=(1,2), keepdims=True)
    img = (1 - img) / (1 - m)
    for i, im in enumerate(img):
        imsave('test_out/' + str(i) + '.png', im)


if __name__ == '__main__':
    main()
