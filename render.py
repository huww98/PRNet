import os
import time
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
from skimage.io import imsave
from OpenGL.EGL import *
from OpenGL.GL import *

def main():
    pos = np.load('test_pos.npy').astype(np.float32)
    face_ind = np.loadtxt('Data/uv-data/face_ind.txt', dtype=np.int32)
    triangles = np.loadtxt('Data/uv-data/triangles.txt', dtype=np.int32)
    # pos = pos.reshape((-1,3))[face_ind]
    triangles = face_ind[triangles]
    print(pos[:, 2].max())
    print(pos[:, 2].min())

    w, h = 388, 467
    far = 512
    ortho = np.array([
        [2./w,   0., 0., -1.],
        [0., 2./h,   0., -1.],
        [0., 0., -2./far, 1.],
        [0., 0., 0.,      1.],
    ], dtype=np.float32)
    print(ortho)
    print(np.matmul(ortho, [1.,1.,1.,1.]))

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    major = EGLint()
    minor = EGLint()
    eglInitialize(display, major, minor)
    attrib_list = [
        EGL_CONFIG_CAVEAT, EGL_NONE,
        EGL_CONFORMANT, EGL_OPENGL_BIT,
        EGL_NONE,
    ]
    attrib_list = arrays.GLintArray.asArray( attrib_list )
    configs = EGLConfig()
    num_config = EGLint()
    eglChooseConfig(display, attrib_list, configs, 1, num_config)
    eglBindAPI(EGL_OPENGL_API)
    ctx = eglCreateContext(display, configs, EGL_NO_CONTEXT, None)
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(2)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, pos, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles, GL_STATIC_DRAW)

    vertexsource = '''#version 430
in  vec3 in_Position;

layout(location = 1) uniform mat4 trans;

void main(void) {
    gl_Position = trans * vec4(in_Position, 1.0);
}
'''
    fragmentsource = '''
#version 430
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

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)

    out_scale = 1
    out_size = (w * out_scale, h * out_scale)

    render_buffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, render_buffer)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, *out_size)

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_buffer)
    assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

    glViewport(0, 0, *out_size)
    glClearColor(0.2, 0., 0., 1.)
    glClearDepth(1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUniformMatrix4fv(1, 1, GL_TRUE, ortho)
    glDrawElements(GL_TRIANGLES, triangles.size, GL_UNSIGNED_INT, None)

    print(glGetError())
    img = np.empty(tuple(reversed(out_size)), dtype=np.float32)
    glReadPixels(0,0,out_size[0],out_size[1], GL_DEPTH_COMPONENT, GL_FLOAT, array=img)

    m = img.min()
    img = (1 - img) / (1 - m)
    imsave('test.png', img)


if __name__ == '__main__':
    main()
