#该模块是对LIF、随机LIF、AdEx、QIF模型的封装，定义了四个神经元模型的类，并通过定义NeuronModels类将四个类统一起来

import numpy as np
import matplotlib.pyplot as plt


# 定义LIF神经元模型类
class LIFNeuron:
    def __init__(self, tau_m=20e-3, tau_ref=2e-3, V_rest=-70e-3, V_reset=-80e-3, V_th=-50e-3):
        self.tau_m = tau_m  # 膜时间常数 (s)
        self.tau_ref = tau_ref  # 不应期时间 (s)
        self.V_rest = V_rest  # 静息电位 (V)
        self.V_reset = V_reset  # 重置电位 (V)
        self.V_th = V_th  # 阈值电位 (V)
        self.Vm = V_rest  # 初始膜电位等于静息电位
        self.refractory = 0  # 不应期计数器

    # 更新神经元状态的方法
    def update(self, I, dt):
        if self.refractory > 0:  # 如果仍在不应期内
            self.refractory -= dt  # 减少不应期剩余时间
            self.Vm = self.V_reset  # 重置膜电位
        else:  # 如果不在不应期内
            dV = (-(self.Vm - self.V_rest) + I) / self.tau_m * dt  # 计算膜电位的变化
            self.Vm += dV  # 更新膜电位
            if self.Vm >= self.V_th:  # 如果膜电位超过阈值
                self.Vm = self.V_reset  # 重置膜电位
                self.refractory = self.tau_ref  # 进入不应期
    # 模拟函数
    def simulate_lif_neuron(self,I, dt, duration):
        Vm_record = []  # 用于记录膜电位的列表
        for _ in range(int(duration / dt)):
            self.update(I, dt)  # 更新神经元状态
            Vm_record.append(self.Vm)  # 记录膜电位
        return Vm_record

# 定义随机LIF神经元模型类
class StochasticLIFNeuron:
    def __init__(self, tau, R, v_reset, v_th):
        self.tau = tau  # 膜时间常数 (s)
        self.R = R  # 膜电阻 (Ω)
        self.v_reset = v_reset  # 重置电位 (V)
        self.v_th = v_th  # 阈值电位 (V)
        self.Vm = 0  # 初始膜电位为0

    # 更新神经元状态的方法
    def update(self, I, dt, sigma):
        dV = ((-self.Vm + I * self.R) / self.tau) * dt  # 计算膜电位的变化
        dV += np.random.normal(0, sigma) * np.sqrt(dt)  # 添加随机项
        self.Vm += dV  # 更新膜电位
        if self.Vm >= self.v_th:  # 如果膜电位超过阈值
            self.Vm = self.v_reset  # 重置膜电位

    # 模拟函数
    def simulate_stochastic_lif_neuron(self, I, dt, duration, sigma):
        Vm_record = []  # 用于记录膜电位的列表
        for _ in range(int(duration / dt)):
            self.update(I, dt, sigma)  # 更新神经元状态
            Vm_record.append(self.Vm)  # 记录膜电位
        return Vm_record

# 定义AdEx神经元模型类
class AdExNeuron:
    def __init__(self, C, gL, EL, VT, DeltaT, a, b, tau_w):
        # 初始化神经元参数
        self.C = C
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.DeltaT = DeltaT
        self.a = a
        self.b = b
        self.tau_w = tau_w
        self.Vm = EL  # 初始膜电位等于静息电位
        self.u = 0  # 初始恢复变量等于0

    # 更新神经元状态的方法
    def update(self, I, dt):
        # 计算膜电位的变化
        dv = (self.gL * (self.EL - self.Vm) + self.gL * self.DeltaT * np.exp((self.Vm - self.VT) / self.DeltaT) - self.u + I) / self.C
        self.Vm += dv * dt  # 更新膜电位

        # 计算恢复变量的变化
        du = (self.a * (self.Vm - self.EL) - self.u) / self.tau_w
        self.u += du * dt  # 更新恢复变量

        # 如果膜电位超过阈值，则发放尖峰并重置膜电位和恢复变量
        if self.Vm >= self.VT:
            self.Vm = self.EL
            self.u += self.b

    # 模拟函数
    def simulate_adex_neuron(self, I, dt, duration):
        Vm_record = []  # 用于记录膜电位的列表
        for _ in range(int(duration / dt)):
            self.update(I, dt)  # 更新神经元状态
            Vm_record.append(self.Vm)  # 记录膜电位
        return Vm_record

# 定义QIF神经元模型类
class QIFNeuron:
    def __init__(self, tau, a, b, delta_T, V_th):
        self.tau = tau  # 膜时间常数 (s)
        self.a = a  # 参数a
        self.b = b  # 参数b
        self.delta_T = delta_T  # 参数delta_T
        self.V_th = V_th  # 阈值电位 (V)
        self.Vm = 0  # 初始膜电位为0

    # 更新神经元状态的方法
    def update(self, I, dt):
        dV = (self.a * (self.Vm - self.delta_T) * (self.Vm - self.b) - self.Vm + I) / self.tau * dt  # 计算膜电位的变化
        self.Vm += dV  # 更新膜电位
        if self.Vm >= self.V_th:  # 如果膜电位超过阈值
            self.Vm = 0  # 重置膜电位

    # 模拟函数
    def simulate_qif_neuron(self, I, dt, duration):
        Vm_record = []  # 用于记录膜电位的列表
        for _ in range(int(duration / dt)):
            self.update(I, dt)  # 更新神经元状态
            Vm_record.append(self.Vm)  # 记录膜电位
        return Vm_record

##############################################################
class NeuronModels:
    def __init__(self, neuron_type, **params):
        self.neuron_type = neuron_type

        if self.neuron_type == 'LIF':
            self.neuron = LIFNeuron(**params)
        elif self.neuron_type == 'StochasticLIF':
            self.neuron = StochasticLIFNeuron(**params)
        elif self.neuron_type == 'AdEx':
            self.neuron = AdExNeuron(**params)
        elif self.neuron_type == 'QIF':
            self.neuron = QIFNeuron(**params)
        else:
            raise ValueError(f"Unsupported neuron type: {self.neuron_type}")

    def simulate_neuron(self, I, dt, duration, sigma=None):
        Vm_record = []  # 用于记录膜电位的列表
        for _ in range(int(duration / dt)):
            if self.neuron_type == 'StochasticLIF':
                self.neuron.update(I, dt, sigma)  # 更新神经元状态（包含额外的 sigma 参数）
            else:
                self.neuron.update(I, dt)  # 更新神经元状态
            Vm_record.append(self.neuron.Vm)  # 记录膜电位
        return Vm_record


# 使用方式演示示例：
if __name__ == "__main__":
    # 创建 LIF 神经元模型实例
    lif_params = {'tau_m': 20e-3, 'tau_ref': 2e-3, 'V_rest': -70e-3, 'V_reset': -80e-3, 'V_th': -50e-3}
    lif_neuron_model = NeuronModels(neuron_type='LIF', **lif_params)

    # 创建 StochasticLIF 神经元模型实例
    stochastic_lif_params = {'tau': 20e-3, 'R': 100e6, 'v_reset': -80e-3, 'v_th': -50e-3}
    stochastic_lif_neuron_model = NeuronModels(neuron_type='StochasticLIF', **stochastic_lif_params)

    # 创建 AdEx 神经元模型实例
    adex_params = {'C': 200e-12, 'gL': 10e-9, 'EL': -70e-3, 'VT': -50e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'b': 0.02e-9, 'tau_w': 30e-3}
    adex_neuron_model = NeuronModels(neuron_type='AdEx', **adex_params)

    # 创建 QIF 神经元模型实例
    qif_params = {'tau': 10e-3, 'a': 0.001, 'b': 0.005, 'delta_T': 0.01, 'V_th': 0.5}
    qif_neuron_model = NeuronModels(neuron_type='QIF', **qif_params)

    # 模拟神经元
    I_input = 0.5e-9  # 输入电流
    dt_simulation = 1e-4  # 模拟步长
    duration_simulation = 0.5  # 模拟时长

    # 模拟 LIF 神经元
    lif_Vm_record = lif_neuron_model.simulate_neuron(I_input, dt_simulation, duration_simulation)

    # 模拟 StochasticLIF 神经元
    sigma_stochastic_lif = 1e-12  # 随机LIF的额外参数
    stochastic_lif_Vm_record = stochastic_lif_neuron_model.simulate_neuron(I_input, dt_simulation, duration_simulation, sigma=sigma_stochastic_lif)

    # 模拟 AdEx 神经元
    adex_Vm_record = adex_neuron_model.simulate_neuron(I_input, dt_simulation, duration_simulation)

    # 模拟 QIF 神经元
    qif_Vm_record = qif_neuron_model.simulate_neuron(I_input, dt_simulation, duration_simulation)

    # 结果可视化
    time = np.arange(0, duration_simulation, dt_simulation) * 1000  # 将时间转换为毫秒
    plt.plot(time, np.array(qif_Vm_record) * 1000)  # 将膜电位转换为毫伏并绘制图形
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.show()
