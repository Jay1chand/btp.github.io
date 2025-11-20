import numpy as np
import matplotlib.pyplot as plt
class BJT:
    def __init__(self, name, collector, base, emitter, type="NPN", Is=1e-15, beta_f=100, beta_r=1):
        self.name = name
        self.collector = collector
        self.base = base
        self.emitter = emitter
        self.type = type.upper()
        self.Is = Is      # Saturation current
        self.beta_f = beta_f  # Forward current gain
        self.beta_r = beta_r  # Reverse current gain
        self.Vt = 0.025   # Thermal voltage (at 25°C)

    def ic(self, Vbe, Vbc):
        """Collector current based on Vbe and Vbc"""
        if self.type == "NPN":
            return self.Is * (np.exp(Vbe / self.Vt) - np.exp(Vbc / self.Vt))
        elif self.type == "PNP":
            return -self.Is * (np.exp(-Vbc / self.Vt) - np.exp(-Vbe / self.Vt))
        else:
            raise ValueError("Unsupported BJT type")

    def ib(self, Vbe, Vbc):
        """Base current based on Vbe and Vbc"""
        if self.type == "NPN":
            return (self.Is / self.beta_f) * (np.exp(Vbe / self.Vt) - 1) + \
                   (self.Is / self.beta_r) * (np.exp(Vbc / self.Vt) - 1)
        elif self.type == "PNP":
            return -((self.Is / self.beta_f) * (np.exp(-Vbe / self.Vt) - 1) + \
                   (self.Is / self.beta_r) * (np.exp(-Vbc / self.Vt) - 1))
        else:
            raise ValueError("Unsupported BJT type")

    def ie(self, Vbe, Vbc):
        """Emitter current from KCL"""
        return self.ic(Vbe, Vbc) + self.ib(Vbe, Vbc)

    def __repr__(self):
        return f"<BJT {self.name} ({self.type}): C={self.collector}, B={self.base}, E={self.emitter}>"


    def plot_iv(self, Vbe_range=(0, 1), Vce_range=(0, 5), steps=100):
     if(self.type == "NPN"):
        Vbe_vals = np.linspace(*Vbe_range, steps)
        Vce_vals = np.linspace(*Vce_range, steps)

        plt.figure(figsize=(8, 6))
        for Vbe in Vbe_vals[::steps // 5]:  # Plot 5 curves for clarity
            Ic_vals = []
            for Vce in Vce_vals:
                Vbc = Vbe - Vce
                ic = self.ic(Vbe, Vbc)
                Ic_vals.append(ic)
            plt.plot(Vce_vals, Ic_vals, label=f"Vbe = {Vbe:.2f} V")

        plt.title(f"BJT {self.type} Output Characteristics")
        plt.xlabel("Vce (V)")
        plt.ylabel("Ic (A)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
     else:
        Vbe_vals = np.linspace(-0.9, 0, steps)
        Vce_vals = np.linspace(-5, 0, steps)

        plt.figure(figsize=(8, 6))
        for Vbe in Vbe_vals[::steps // 5]:  # Plot 5 curves for clarity
            Ic_vals = []
            for Vce in Vce_vals:
                Vbc = Vbe - Vce
                ic = self.ic(Vbe, Vbc)
                Ic_vals.append(ic)
            plt.plot(Vce_vals, Ic_vals, label=f"Vbe = {Vbe:.2f} V")

        plt.title(f"BJT {self.type} Output Characteristics")
        plt.xlabel("Vce (V)")
        plt.ylabel("Ic (A)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
