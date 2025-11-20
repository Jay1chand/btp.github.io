import numpy as np
import matplotlib.pyplot as plt

class VoltageSource:
    def __init__(self, name, node_pos, node_neg, v_type='dc', amplitude=5.0, frequency=50.0, phase=0.0):
        self.name = name
        self.node_pos = node_pos
        self.node_neg = node_neg
        self.v_type = v_type.lower()  # 'dc' or 'ac' or 'pulse'
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.type = "VoltageSource"

    def v(self, t):
        if self.v_type == 'dc':
            return np.full_like(t, self.amplitude)
        elif self.v_type == 'ac':
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        else:
            raise ValueError(f"Unsupported voltage type: {self.v_type}")

    def __repr__(self):
        return f"<VoltageSource {self.name} ({self.v_type.upper()} {self.amplitude}V)>"

    def plot_iv(self, t_max=0.1, steps=1000):
        t = np.linspace(0, t_max, steps)
        v_vals = self.v(t)

        plt.figure(figsize=(8, 5))
        plt.plot(t, v_vals)
        plt.title(f"Voltage Source {self.name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
