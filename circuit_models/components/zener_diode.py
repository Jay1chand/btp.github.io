import numpy as np
import matplotlib.pyplot as plt

class ZenerDiode:
    def __init__(self, name, node_anode, node_cathode, I_s=1e-12, n=1.0, V_T=0.025, V_z=5.6):
        self.name = name
        self.node_anode = node_anode
        self.node_cathode = node_cathode
        self.I_s = I_s
        self.n = n
        self.V_T = V_T
        self.V_z = V_z
        self.type = "ZenerDiode"

    def i(self, V):
        """Current-voltage characteristic."""
        V_d = V
        I_fwd = self.I_s * (np.exp(V_d / (self.n * self.V_T)) - 1)  # forward bias
        I_rev = -self.I_s * (np.exp(-(V_d + self.V_z) / (self.n * self.V_T)))  # zener reverse region
        return np.where(V_d >= 0, I_fwd, I_rev)

    def __repr__(self):
        return f"<ZenerDiode {self.name} ({self.type})>"

    def plot_iv(self, V_range=(-10, 10), steps=200):
        V_vals = np.linspace(*V_range, steps)
        I_vals = self.i(V_vals)

        plt.figure(figsize=(8, 6))
        plt.plot(V_vals, I_vals)
        plt.title(f"Zener Diode {self.name} I-V Characteristics")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
