import numpy as np
import matplotlib.pyplot as plt
class MOSFET:
    def __init__(self, name, drain, gate, source, body, type="NMOS", Vth=1.0, Kn=1e-3, lambda_=0):
        self.name = name
        self.drain = drain
        self.gate = gate
        self.source = source
        self.body = body
        self.type = type.upper()  # "NMOS" or "PMOS"
        self.Vth = Vth            # Threshold voltage
        self.Kn = Kn              # Process gain (μn*Cox*W/L)
        self.lambda_ = lambda_    # Channel-length modulation (optional)

    def ids(self, Vgs, Vds):
        """Compute drain current for given Vgs and Vds"""
        if self.type == "NMOS":
            return self._ids_nmos(Vgs, Vds)
        elif self.type == "PMOS":
            return self._ids_pmos(Vgs, Vds)
        else:
            raise ValueError("Unsupported MOSFET type")

    def _ids_nmos(self, Vgs, Vds):
        if Vgs < self.Vth:
            return 0  # Cutoff
        elif Vds < (Vgs - self.Vth):
            # Triode region
            return self.Kn * ((Vgs - self.Vth) * Vds - 0.5 * Vds**2) * (1 + self.lambda_ * Vds)
        else:
            # Saturation region
            return 0.5 * self.Kn * (Vgs - self.Vth)**2 * (1 + self.lambda_ * Vds)

    def _ids_pmos(self, Vgs, Vds):
        Vsg = -Vgs
        Vsd = -Vds
        if Vsg < self.Vth:
            return 0  # Cutoff
        elif Vsd < (Vsg - self.Vth):
            return -self.Kn * ((Vsg - self.Vth) * Vsd - 0.5 * Vsd**2) * (1 + self.lambda_ * Vsd)
        else:
            return -0.5 * self.Kn * (Vsg - self.Vth)**2 * (1 + self.lambda_ * Vsd)

    def __repr__(self):
        return f"<MOSFET {self.name} ({self.type}): D={self.drain}, G={self.gate}, S={self.source}, B={self.body}>"

    def plot_iv(self, Vgs_range=(0, 5), Vds_range=(0, 5), steps=100):
     if(self.type == "NMOS"):
       Vgs_vals = np.linspace(*Vgs_range, steps)
       Vds_vals = np.linspace(*Vds_range, steps)

       plt.figure(figsize=(8, 6))
       for Vgs in Vgs_vals[::steps // 5]: 
        Ids_vals = []
        for Vds in Vds_vals:
         ids = self.ids(Vgs, Vds)
         Ids_vals.append(ids)
        plt.plot(Vds_vals, Ids_vals, label=f"Vgs = {Vgs:.2f} V")

       plt.title(f"MOSFET {self.type} Output Characteristics")
       plt.xlabel("Vds (V)")
       plt.ylabel("Ids (A)")
       plt.grid(True)
       plt.legend()
       plt.tight_layout()
       plt.show()
     else :
       Vgs_vals = np.linspace(-5, 0, 100)  # PMOS Vgs from -5V to 0V
       Vds_vals = np.linspace(0, -5, 100)  # PMOS Vds from 0V to -5V

       plt.figure(figsize=(8, 6))
       for Vgs in Vgs_vals[::steps // 5]: 
        Ids_vals = []
        for Vds in Vds_vals:
         ids = self.ids(Vgs, Vds)
         Ids_vals.append(ids)
        plt.plot(Vds_vals, Ids_vals, label=f"Vgs = {Vgs:.2f} V")

       plt.title(f"MOSFET {self.type} Output Characteristics")
       plt.xlabel("Vds (V)")
       plt.ylabel("Ids (A)")
       plt.grid(True)
       plt.legend()
       plt.tight_layout()
       plt.show()