import numpy as np
import matplotlib.pyplot as plt
class Inductor:
    def __init__(self, name, node1, node2, inductance=1e-3):
        self.name = name
        self.node1=node1
        self.node2=node2
        self.L=inductance
        self.type ="Inductor"

    def v(self,di_dt):
        return self.L*di_dt
    def i(self,v,dt, i0=0):
        return i0+(v*dt/self.L)
    def __repr__(self):
        return f"<Inductor {self.name} ({self.type})>"
    def plot_iv(self, di_dt_range=(0, 1), steps=100):
        di_dt_vals = np.linspace(*di_dt_range, steps)
        V_vals = self.v(di_dt_vals)

        plt.figure(figsize=(8, 6))
        plt.plot(di_dt_vals, V_vals)
        plt.title(f"Inductor {self.name} V vs dI/dt")
        plt.xlabel("dI/dt (A/s)")
        plt.ylabel("V (V)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()    