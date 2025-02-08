import numpy as np
import matplotlib.pyplot as plt

# 给定的两个坐标
x1, y1 = 1, 2
x2, y2 = 3, 10

# 定义分段函数
def piecewise_function(x):
    if x < x1:
        # 区间 1
        a1, b1, d1 = 2, 1, 0.5  # 自定义参数
        return a1 * np.exp(b1 * (x - x1)) + d1
    elif x1 <= x <= x2:
        # 区间 2 (使用逻辑斯谛函数来模拟S型)
        L = max(y1, y2)
        k = 8  # 可以调整斜率
        x0 = (x1 + x2) / 2
        return L / (1 + np.exp(-k * (x - x0)))
    else:
        # 区间 3
        a3, b3, d3 = 2, -1, 6  # 自定义参数
        return a3 * np.exp(b3 * (x - x2)) + d3

# 生成 x 值并计算对应的 y 值
x_values = np.linspace(0, 4, 100)
y_values = [piecewise_function(x) for x in x_values]

# 绘制曲线和给定点
plt.plot(x_values, y_values, label='S-Shape Piecewise Function')
plt.scatter([x1, x2], [y1, y2], color='red')  # 标记给定点
plt.text(x1, y1, f'({x1}, {y1})', fontsize=12, verticalalignment='bottom')
plt.text(x2, y2, f'({x2}, {y2})', fontsize=12, verticalalignment='bottom')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise S-Shape Curve through Given Points')
plt.legend()
plt.grid(True)
plt.show()