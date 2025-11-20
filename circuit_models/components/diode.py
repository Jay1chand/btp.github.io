import numpy as np
import matplotlib.pyplot as plt  # Add this import if not already present

class Diode:
    def __init__(self, name, anode, cathode, Is=1e-12, n=1, Vt=0.025):
        self.name = name
        self.anode = anode
        self.cathode = cathode
        self.Is = Is  # Saturation current
        self.n = n    # Ideality factor
        self.Vt = Vt  # Thermal voltage

    def current(self, V):
        """Returns current through the diode for a given voltage or array of voltages (V = Vanode - Vcathode)"""
        return self.Is * (np.exp(V / (self.n * self.Vt)) - 1)

    def plot_iv(self):
        """Plot the I-V curve of the diode"""
        V = np.linspace(-0.7, 0.7, 200)
        I = self.current(V)
        plt.plot(V, I)
        plt.title(f"Diode {self.name} I-V Curve")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.grid(True)
        plt.show()

    def __repr__(self):
        return f"<Diode {self.name}: {self.anode}->{self.cathode}>"
