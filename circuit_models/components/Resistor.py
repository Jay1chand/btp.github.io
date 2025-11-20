import numpy as np
import matplotlib.pyplot as plt

class Resistor:
    def __init__(self, name, Resistivity=10, Length=1, Area=0.1):
        self.name = name
        self.Length = Length
        self.Area = Area
        self.Resistivity = Resistivity
        self.type = "Resistor"  # Add this to avoid errors

    def Resistance(self):
        """Calculate resistance using R = ρ * L / A"""
        return self.Resistivity * self.Length / self.Area

    def i(self, V):
        """Current I = V / R"""
        return V / self.Resistance()

    def __repr__(self):
        return f"<Resistor {self.name} ({self.type})>"

    def plot_iv(self, V_range=(0, 1), steps=100):
        V_vals = np.linspace(*V_range, steps)
        I_vals = self.i(V_vals)

        plt.figure(figsize=(8, 6))
        plt.plot(V_vals, I_vals, label=f'R = {self.Resistance():.2f} Ω')

        plt.title(f"Resistor {self.name} Output Characteristics")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
