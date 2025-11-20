import numpy as np
import matplotlib.pyplot as plt

class OpAmp:
    def __init__(self, name, non_inverting, inverting, output, gain=1e5, input_resistance=1e6, output_resistance=100):
        self.name = name
        self.non_inverting = non_inverting
        self.inverting = inverting
        self.output = output
        self.gain = gain
        self.input_resistance = input_resistance
        self.output_resistance = output_resistance

    def vout(self, vin_plus, vin_minus):
        return self.gain * (vin_plus - vin_minus)

    def __repr__(self):
        return f"<OpAmp {self.name} Gain={self.gain} Rin={self.input_resistance} Rout={self.output_resistance}>"

    def plot_iv(self, vin_range=(-0.1, 0.1), steps=100):
        vin = np.linspace(*vin_range, steps)
        vout = self.vout(vin, 0)  # assume vin_minus = 0
        plt.figure()
        plt.plot(vin, vout)
        plt.title(f"OpAmp {self.name} Transfer Characteristics")
        plt.xlabel("V_in (V)")
        plt.ylabel("V_out (V)")
        plt.grid(True)
        plt.show()
