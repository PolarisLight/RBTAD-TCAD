import os

print(os.environ.get("MUJOCO_GL"))
import mujoco  # noqa: F401
import OpenGL.GL as GL  # noqa: F401

print("egl import ok")
