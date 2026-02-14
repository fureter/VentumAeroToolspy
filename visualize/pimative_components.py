import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate

vert = """
// Uniforms
// ------------------------------------
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   vec4 u_color;
uniform   mat4 u_model;
uniform   vec4 u_position;

// Attributes
// ------------------------------------
attribute vec3 a_position;
attribute vec4 a_color;
attribute vec3 a_normal;

// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    v_color = a_color * u_color;
    gl_Position = u_projection * u_view * (u_model * vec4(a_position,1.0) + u_position);
}
"""


frag = """
// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
"""


# -----------------------------------------------------------------------------
def Arrow():
    """
    Build vertices for a colored cube.

    V  is the vertices
    I1 is the indices for a filled cube (use with GL_TRIANGLES)
    I2 is the indices for an outline cube (use with GL_LINES)
    """
    vtype = [('a_position', np.float32, 3),
             ('a_normal', np.float32, 3),
             ('a_color', np.float32, 4)]
    # Vertices positions
    v = [[.1, .1, .1],
         [-.1, .1, .1],
         [-.1, -.1, .1],
         [.1, -.1, .1],
         [.1, -.1, 0],
         [.1, .1, 0],
         [-.1, .1, 0],
         [-.1, -.1, 0]]
    # Face Normals
    n = [[0, 0, 1], [1, 0, 0], [0, 1, 0],
         [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    # Vertice colors
    c = [[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1],
         [0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]

    V = np.array([(v[0], n[0], c[0]), (v[1], n[0], c[1]),
                  (v[2], n[0], c[2]), (v[3], n[0], c[3]),
                  (v[0], n[1], c[0]), (v[3], n[1], c[3]),
                  (v[4], n[1], c[4]), (v[5], n[1], c[5]),
                  (v[0], n[2], c[0]), (v[5], n[2], c[5]),
                  (v[6], n[2], c[6]), (v[1], n[2], c[1]),
                  (v[1], n[3], c[1]), (v[6], n[3], c[6]),
                  (v[7], n[3], c[7]), (v[2], n[3], c[2]),
                  (v[7], n[4], c[7]), (v[4], n[4], c[4]),
                  (v[3], n[4], c[3]), (v[2], n[4], c[2]),
                  (v[4], n[5], c[4]), (v[7], n[5], c[7]),
                  (v[6], n[5], c[6]), (v[5], n[5], c[5])],
                 dtype=vtype)
    I1 = np.resize(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), 6 * (2 * 3))
    I1 += np.repeat(4 * np.arange(2 * 3, dtype=np.uint32), 6)

    I2 = np.resize(
        np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.uint32), 6 * (2 * 4))
    I2 += np.repeat(4 * np.arange(6, dtype=np.uint32), 8)

    return V, I1, I2
