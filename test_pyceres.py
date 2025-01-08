
import numpy as np
import pyceres
#from pyceres import Problem, CostFunction, SolverOptions, solve

# 定义代价函数（Cost Function）
class LinearCostFunction(pyceres.CostFunction):
    def __init__(self, x, y):
        super(LinearCostFunction, self).__init__()
        self.x = x
        self.y = y
        # 设置残差维度为1
        self.set_num_residuals(1)
        # 设置每个参数块的维度
        self.set_parameter_block_sizes([1, 1])
        
    def Evaluate(self, parameters, residuals, jacobians):
        m = parameters[0][0]
        c = parameters[1][0]

        # 计算残差
        residuals[0] = m * self.x + c - self.y

        # 如果需要计算雅可比矩阵
        if jacobians is not None:
            jacobians[0][0] = self.x  # 对 m 的导数
            jacobians[1][0] = 1.0      # 对 c 的导数

        return True  # 返回True表示计算成功

# 创建问题实例
problem = pyceres.Problem()

# 观测数据
data = [
    (0.2, 0.3),
    (0.8, 0.9),
    (1.1, 1.5),
    (1.7, 2.0),
    (2.4, 2.4),
    (3.0, 2.9)
]

# 初始参数值
initial_m = np.array([0.0], dtype=np.float64)
initial_c = np.array([0.0], dtype=np.float64)

# 添加残差块
for x, y in data:
    cost_function = LinearCostFunction(x, y)
    problem.add_residual_block(cost_function, None, [initial_m, initial_c])

# 配置求解器选项
options = pyceres.SolverOptions()
options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout = True

# 运行求解器
summary = pyceres.SolverSummary()
try:
    pyceres.solve(options, problem,summary)
except Exception as e:
    print(f"An error occurred: {e}")



# 输出结果
#print("Summary: ", summary)
print("Final m: ", initial_m)
print("Final c: ", initial_c)