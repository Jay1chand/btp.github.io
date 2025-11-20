import numpy as np
import matplotlib.pyplot as plt

class IdealOpAmp:
    def __init__(self, name, v_plus, v_minus, v_out, gain=1e5, v_supply=15):
        self.name = name
        self.v_plus = v_plus
        self.v_minus = v_minus
        self.v_out = v_out
        self.gain = gain          # Open-loop gain A
        self.v_supply = v_supply  # Vcc (assumes symmetric ±Vcc)
        self.type = "OpAmp"

    def v_output(self, v_p, v_n):
        v_diff = v_p - v_n
        v_out = self.gain * v_diff
        # Clipping to supply rails
        v_out = np.clip(v_out, -self.v_supply, self.v_supply)
        return v_out

    def __repr__(self):
        return f"<OpAmp {self.name} (Gain={self.gain})>"

    def plot_iv(self, v_range=(-0.1, 0.1), steps=1000):
        v_diff = np.linspace(*v_range, steps)
        v_out = self.v_output(v_diff, np.zeros_like(v_diff))  # v_p = v_diff, v_n = 0

        plt.figure(figsize=(8, 5))
        plt.plot(v_diff, v_out)
        plt.title(f"Op-Amp {self.name} Transfer Curve")
        plt.xlabel("Differential Input Voltage (V+ - V-) [V]")
        plt.ylabel("Output Voltage [V]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
