import LIF_model

# 创建 LIF 神经元模型实例
lif_params = {'tau_m': 20e-3, 'tau_ref': 2e-3, 'V_rest': -70e-3, 'V_reset': -80e-3, 'V_th': -50e-3}
lif_neuron_model = LIF_model.NeuronModels(neuron_type='LIF', **lif_params)



# import numpy as np
# import matplotlib.pyplot as plt
#
## 定义AdEx神经元模型类
# class AdExNeuron:
#     def __init__(self, C, gL, EL, VT, DeltaT, a, b, tau_w):
#         # 初始化神经元参数
#         self.C = C
#         self.gL = gL
#         self.EL = EL
#         self.VT = VT
#         self.DeltaT = DeltaT
#         self.a = a
#         self.b = b
#         self.tau_w = tau_w
#         self.vm = EL  # 初始膜电位等于静息电位
#         self.u = 0  # 初始恢复变量等于0
#
#     # 更新神经元状态的方法
#     def update(self, I, dt):
#         # 计算膜电位的变化
#         dv = (self.gL * (self.EL - self.vm) + self.gL * self.DeltaT * np.exp((self.vm - self.VT) / self.DeltaT) - self.u + I) / self.C
#         self.vm += dv * dt  # 更新膜电位
#
#         # 计算恢复变量的变化
#         du = (self.a * (self.vm - self.EL) - self.u) / self.tau_w
#         self.u += du * dt  # 更新恢复变量
#
#         # 如果膜电位超过阈值，则发放尖峰并重置膜电位和恢复变量
#         if self.vm >= self.VT:
#             self.vm = self.EL
#             self.u += self.b
#
# # 模拟函数
# def simulate_adex_neuron(neuron, I, dt, duration):
#     vm_record = []  # 用于记录膜电位的列表
#     for _ in range(int(duration / dt)):
#         neuron.update(I, dt)  # 更新神经元状态
#         vm_record.append(neuron.vm)  # 记录膜电位
#     return vm_record
#
# # 参数设置
# C = 281e-12  # 膜电容 (F)
# gL = 30e-9  # 漏导 (S)
# EL = -70.6e-3  # 静息电位 (V)
# VT = -50.4e-3  # 阈值电位 (V)# 参数设置
# tau_m = 20e-3  # 膜时间常数 (s)
# tau_ref = 2e-3  # 不应期时间 (s)
# V_rest = -70e-3  # 静息电位 (V)
# V_reset = -80e-3  # 重置电位 (V)
# V_th = -50e-3  # 阈值电位 (V)
# DeltaT = 2e-3  # 阈值斜率因子 (V)
# a = 4e-9  # 恢复变量的最大增长率 (A)
# b = 0.0805e-9  # 恢复变量的增加大小 (A)
# tau_w = 144e-3  # 恢复变量的时间常数 (s)
#
# # 创建AdEx神经元实例
# neuron = AdExNeuron(C, gL, EL, VT, DeltaT, a, b, tau_w)
#
# # 模拟
# dt = 0.1e-3  # 时间步长 (s)
# duration = 100e-3  # 模拟持续时间 (s)
# I = 200e-11  # 注入电流 (A)
# vm_record = simulate_adex_neuron(neuron, I, dt, duration)
#
# # 结果可视化
# import matplotlib.pyplot as plt
# time = np.arange(0, duration, dt) * 1000  # 将时间转换为毫秒
# plt.plot(time, np.array(vm_record) * 1000)  # 将膜电位转换为毫伏并绘制图形
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane potential (mV)')
# plt.show()

##################################################################################################
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义LIF神经元模型类
# class LIFNeuron:
#     def __init__(self, tau_m, tau_ref, V_rest, V_reset, V_th):
#         self.tau_m = tau_m  # 膜时间常数 (s)
#         self.tau_ref = tau_ref  # 不应期时间 (s)
#         self.V_rest = V_rest  # 静息电位 (V)
#         self.V_reset = V_reset  # 重置电位 (V)
#         self.V_th = V_th  # 阈值电位 (V)
#         self.Vm = V_rest  # 初始膜电位等于静息电位
#         self.refractory = 0  # 不应期计数器
#
#     # 更新神经元状态的方法
#     def update(self, I, dt):
#         if self.refractory > 0:  # 如果仍在不应期内
#             self.refractory -= dt  # 减少不应期剩余时间
#             self.Vm = self.V_reset  # 重置膜电位
#         else:  # 如果不在不应期内
#             dV = (-(self.Vm - self.V_rest) + I) / self.tau_m * dt  # 计算膜电位的变化
#             self.Vm += dV  # 更新膜电位
#             if self.Vm >= self.V_th:  # 如果膜电位超过阈值
#                 self.Vm = self.V_reset  # 重置膜电位
#                 self.refractory = self.tau_ref  # 进入不应期
#
# # 模拟函数
# def simulate_lif_neuron(neuron, I, dt, duration):
#     vm_record = []  # 用于记录膜电位的列表
#     for _ in range(int(duration / dt)):
#         neuron.update(I, dt)  # 更新神经元状态
#         vm_record.append(neuron.Vm)  # 记录膜电位
#     return vm_record
#
# # 参数设置
# tau_m = 20e-3  # 膜时间常数 (s)
# tau_ref = 2e-3  # 不应期时间 (s)
# V_rest = -70e-3  # 静息电位 (V)
# V_reset = -80e-3  # 重置电位 (V)
# V_th = -50e-3  # 阈值电位 (V)
#
# # 创建LIF神经元实例
# neuron = LIFNeuron(tau_m, tau_ref, V_rest, V_reset, V_th)
#
# # 模拟
# dt = 0.1e-3  # 时间步长 (s)
# duration = 100e-3  # 模拟持续时间 (s)
# I = 200e-12  # 注入电流 (A)
# vm_record = simulate_lif_neuron(neuron, I, dt, duration)
#
# # 结果可视化
# time = np.arange(0, duration, dt) * 1000  # 将时间转换为毫秒
# plt.plot(time, np.array(vm_record) * 1000)  # 将膜电位转换为毫伏并绘制图形
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane potential (mV)')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义QIF神经元模型类
# class QIFNeuron:
#     def __init__(self, tau, a, b, delta_T, V_th):
#         self.tau = tau  # 膜时间常数 (s)
#         self.a = a  # 参数a
#         self.b = b  # 参数b
#         self.delta_T = delta_T  # 参数delta_T
#         self.V_th = V_th  # 阈值电位 (V)
#         self.Vm = 0  # 初始膜电位为0
#
#     # 更新神经元状态的方法
#     def update(self, I, dt):
#         dV = (self.a * (self.Vm - self.delta_T) * (self.Vm - self.b) - self.Vm + I) / self.tau * dt  # 计算膜电位的变化
#         self.Vm += dV  # 更新膜电位
#         if self.Vm >= self.V_th:  # 如果膜电位超过阈值
#             self.Vm = 0  # 重置膜电位
#
# # 模拟函数
# def simulate_qif_neuron(neuron, I, dt, duration):
#     vm_record = []  # 用于记录膜电位的列表
#     for _ in range(int(duration / dt)):
#         neuron.update(I, dt)  # 更新神经元状态
#         vm_record.append(neuron.Vm)  # 记录膜电位
#     return vm_record
#
# # 参数设置
# tau = 10e-3  # 膜时间常数 (s)
# a = 0.02  # 参数a
# b = 0.2  # 参数b
# delta_T = 2  # 参数delta_T
# V_th = 1  # 阈值电位 (V)
#
# # 创建QIF神经元实例
# neuron = QIFNeuron(tau, a, b, delta_T, V_th)
#
# # 模拟
# dt = 0.1e-3  # 时间步长 (s)
# duration = 100e-3  # 模拟持续时间 (s)
# I = 2  # 注入电流 (任意单位)
# vm_record = simulate_qif_neuron(neuron, I, dt, duration)
#
# # 结果可视化
# time = np.arange(0, duration, dt) * 1000  # 将时间转换为毫秒
# plt.plot(time, np.array(vm_record))  # 绘制膜电位图形
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane potential')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义随机LIF神经元模型类
# class StochasticLIFNeuron:
#     def __init__(self, tau, R, v_reset, v_th):
#         self.tau = tau  # 膜时间常数 (s)
#         self.R = R  # 膜电阻 (Ω)
#         self.v_reset = v_reset  # 重置电位 (V)
#         self.v_th = v_th  # 阈值电位 (V)
#         self.vm = 0  # 初始膜电位为0
#
#     # 更新神经元状态的方法
#     def update(self, I, dt, sigma):
#         dV = ((-self.vm + I * self.R) / self.tau) * dt  # 计算膜电位的变化
#         dV += np.random.normal(0, sigma) * np.sqrt(dt)  # 添加随机项
#         self.vm += dV  # 更新膜电位
#         if self.vm >= self.v_th:  # 如果膜电位超过阈值
#             self.vm = self.v_reset  # 重置膜电位
#
# # 模拟函数
# def simulate_stochastic_lif_neuron(neuron, I, dt, duration, sigma):
#     vm_record = []  # 用于记录膜电位的列表
#     for _ in range(int(duration / dt)):
#         neuron.update(I, dt, sigma)  # 更新神经元状态
#         vm_record.append(neuron.vm)  # 记录膜电位
#     return vm_record
#
# # 参数设置
# tau = 20e-3  # 膜时间常数 (s)
# R = 100e6  # 膜电阻 (Ω)
# v_reset = 0  # 重置电位 (V)
# v_th = 1  # 阈值电位 (V)
#
# # 创建随机LIF神经元实例
# neuron = StochasticLIFNeuron(tau, R, v_reset, v_th)
#
# # 模拟
# dt = 0.1e-3  # 时间步长 (s)
# duration = 100e-3  # 模拟持续时间 (s)
# I = 200e-10  # 注入电流 (A)
# sigma = 0.02  # 随机项的标准差
# vm_record = simulate_stochastic_lif_neuron(neuron, I, dt, duration, sigma)
#
# # 结果可视化
# time = np.arange(0, duration, dt) * 1000  # 将时间转换为毫秒
# plt.plot(time, np.array(vm_record) * 1000)  # 将膜电位转换为毫伏并绘制图形
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane potential (mV)')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# def lif_neuron_simulation(timesteps, dt, R, C, V_rest, V_th, V_reset, I):
#     V = np.zeros(timesteps)
#     spikes = []
#
#     for t in range(1, timesteps):
#         dV = (-V[t-1] + V_rest + R * I[t-1]) / (R * C)
#         V[t] = V[t-1] + dt * dV
#
#         if V[t] >= V_th:
#             spikes.append(t)
#             V[t] = V_reset
#
#     return spikes
#
# def analyze_isi(spikes):
#     isi = np.diff(spikes)
#     mean_isi = np.mean(isi)
#     isi_std = np.std(isi)
#
#     return isi, mean_isi, isi_std
#
# # 模拟参数设置
# timesteps = 1000
# dt = 0.1
# R = 1.0
# C = 1.0
# V_rest = 0.0
# V_th = 1.0
# V_reset = 0.0
#
# # 不同输入电流
# I_values = [1.1, 1.2, 1.3, 1.4, 1.5]  # 不同电流值
# isi_values = []
#
# # 进行模拟和分析
# for I_value in I_values:
#     I = np.zeros(timesteps)
#     I[300:600] = I_value  # 修改输入电流的模式
#
#     spikes = lif_neuron_simulation(timesteps, dt, R, C, V_rest, V_th, V_reset, I)
#     isi, _, _ = analyze_isi(spikes)
#     isi_values.append(np.mean(isi))  # 这里取平均 ISI 作为代表性值
#
# # 绘制ISI曲线
# plt.figure(figsize=(8, 6))
# plt.plot(I_values, isi_values, marker='o', linestyle='-')
# plt.title('ISI vs Input Current')
# plt.xlabel('Input Current')
# plt.ylabel('Mean ISI (ms)')
# plt.grid(True)
# plt.show()

