import numpy as np

class DCSolver:
    def __init__(self, components):
        self.components = components
        self.node_map = {}   # maps node label → index
        self.next_index = 1  # node 0 reserved for ground

    def _get_node_index(self, node):
        if node == "0" or node.lower() == "gnd":
            return 0
        if node not in self.node_map:
            self.node_map[node] = self.next_index
            self.next_index += 1
        return self.node_map[node]

    def build_mna(self):
        n = self.next_index - 1
        G = np.zeros((n, n))
        I = np.zeros(n)

        for comp in self.components:
            # Resistor
            if comp.type == "R":
                n1 = self._get_node_index(comp.n1)
                n2 = self._get_node_index(comp.n2)
                g = 1.0 / comp.value
                if n1 > 0:
                    G[n1-1, n1-1] += g
                if n2 > 0:
                    G[n2-1, n2-1] += g
                if n1 > 0 and n2 > 0:
                    G[n1-1, n2-1] -= g
                    G[n2-1, n1-1] -= g

            # Current source
            elif comp.type == "I":
                n1 = self._get_node_index(comp.n1)  # + terminal
                n2 = self._get_node_index(comp.n2)  # - terminal
                if n1 > 0:
                    I[n1-1] -= comp.value
                if n2 > 0:
                    I[n2-1] += comp.value

            # Voltage source (extended MNA)
            elif comp.type == "V":
                n1 = self._get_node_index(comp.n1)
                n2 = self._get_node_index(comp.n2)
                size = G.shape[0]
                G = np.pad(G, ((0,1),(0,1)), 'constant')
                I = np.pad(I, (0,1), 'constant')

                row = size
                if n1 > 0:
                    G[row, n1-1] = 1
                    G[n1-1, row] = 1
                if n2 > 0:
                    G[row, n2-1] = -1
                    G[n2-1, row] = -1
                I[row] = comp.value

        return G, I

    def solve(self):
        G, I = self.build_mna()
        try:
            V = np.linalg.solve(G, I)
            voltages = {"0": 0.0}
            for node, idx in self.node_map.items():
                voltages[node] = V[idx-1]
            return voltages
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Matrix solve failed: {e}")
