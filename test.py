import glfw
from OpenGL.GL import *
import numpy as np

# Initialize GLFW and OpenGL context
def init_window(width, height, title):
    if not glfw.init():
        raise Exception("GLFW can't be initialized")
    
    # Set window hints for GLFW (OpenGL version 3.3 and compatibility)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    # Create the window
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")
    
    # Make the context current
    glfw.make_context_current(window)
    
    return window

# Define the vertex data (a simple triangle)
def create_triangle():
    # Vertices of a triangle
    vertices = np.array([
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0
    ], dtype=np.float32)
    
    # Create a Vertex Buffer Object (VBO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    # Define the layout of the vertex data (position only)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

def main():
    # Initialize the window
    window = init_window(800, 600, "OpenGL Triangle")

    # Specify the color to clear the screen
    glClearColor(0.1, 0.1, 0.1, 1.0)
    
    # Define the triangle vertices
    create_triangle()

    # Main rendering loop
    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Render the triangle
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Clean up and close the window
    glfw.terminate()

if __name__ == "__main__":
    main()
