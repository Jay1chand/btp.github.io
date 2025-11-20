import numpy as np
import matplotlib.pyplot as plt

class Capacitor:
    def __init__(self, name, node1, node2, permittivity=8.85e-12, area=1e-4, distance=1e-6):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.permittivity = permittivity  # ε
        self.area = area                  # A
        self.distance = distance          # d
        self.type = "Capacitor"

    @property
    def capacitance(self):
        return self.permittivity * self.area / self.distance  # C = εA/d

    def q(self, V):
        """Charge stored at voltage V."""
        return self.capacitance * V

    def i(self, V, dVdt):
        """Current i = C * dV/dt (assumes known dV/dt)"""
        return self.capacitance * dVdt

    def __repr__(self):
        return f"<Capacitor {self.name} ({self.type})>"

    def plot_iv(self, V_range=(0, 1), steps=100):
        """Plot Q-V characteristics."""
        V_vals = np.linspace(*V_range, steps)
        Q_vals = self.q(V_vals)

        plt.figure(figsize=(8, 6))
        plt.plot(V_vals, Q_vals)
        plt.title(f"Capacitor {self.name} Q-V Characteristics")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Charge (C)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
