class CurrentSource:
    def __init__(self, name, pos_node, neg_node, current=0.001):
        self.name = name
        self.pos_node = pos_node
        self.neg_node = neg_node
        self.current = current  # in Amperes

    def i(self, V):
        # Ideal current source: constant current
        return self.current

    def __repr__(self):
        return f"<CurrentSource {self.name}: {self.current}A from {self.neg_node} to {self.pos_node}>"

    def plot_iv(self, V_range=(-1, 1), steps=100):
        import numpy as np
        import matplotlib.pyplot as plt

        V_vals = np.linspace(*V_range, steps)
        I_vals = np.full_like(V_vals, self.current)

        plt.figure()
        plt.plot(V_vals, I_vals, label=f"{self.name} ({self.current} A)")
        plt.title("Current Source I-V Characteristics")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.grid(True)
        plt.legend()
        plt.show()
