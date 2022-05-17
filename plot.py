import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.linspace(0,2 * np.pi, 400)
a = np.linspace(0, 3, 20)
x,a=np.meshgrid(x,a)

z=np.arctan(np.sin(x)/(a-np.cos(x)))
#绘制图像
fig = plt.figure()
ax = plt.axes(projection='3d')
#调用绘制线框图的函数plot_wireframe()
ax.plot_wireframe(x, a, z, color='black')
ax.set_xlabel('sita')
ax.set_ylabel('ob/oa')
ax.set_zlabel('z')
ax.set_title('wireframe')
plt.show()


