# circuit_models/netlist_parser.py
import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ---------- Base ----------
@dataclass
class BaseComponent:
    name: str
    ctype: str
    nodes: Tuple[str, ...] = field(default_factory=tuple)

# ---------- Passive ----------
@dataclass
class Resistor(BaseComponent):
    value: float = 0.0  # ohm
    def __init__(self, name, n1, n2, value_ohm):
        super().__init__(name, "R", (n1, n2))
        self.value = float(value_ohm)
    def plot_iv(self):
        v = np.linspace(-5, 5, 400)
        i = v / self.value
        plt.plot(v, i, label=f"{self.name} (R={self.value}Ω)")
        plt.xlabel("Voltage V (V)"); plt.ylabel("Current I (A)"); plt.legend(); plt.grid(True, alpha=.3)

@dataclass
class Capacitor(BaseComponent):
    value: float = 0.0  # F
    def __init__(self, name, n1, n2, value_f):
        super().__init__(name, "C", (n1, n2))
        self.value = float(value_f)
    def plot_ac(self):
        f = np.logspace(0, 6, 500)  # 1 Hz to 1 MHz
        w = 2*np.pi*f
        Z = 1/(1j*w*self.value)
        plt.semilogx(f, np.abs(Z))
        plt.title(f"|Z_C| vs f — {self.name} (C={self.value}F)")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("|Z| (Ω)"); plt.grid(True, which="both", alpha=.3)

@dataclass
class Inductor(BaseComponent):
    value: float = 0.0  # H
    def __init__(self, name, n1, n2, value_h):
        super().__init__(name, "L", (n1, n2))
        self.value = float(value_h)
    def plot_ac(self):
        f = np.logspace(0, 6, 500)
        w = 2*np.pi*f
        Z = 1j*w*self.value
        plt.semilogx(f, np.abs(Z))
        plt.title(f"|Z_L| vs f — {self.name} (L={self.value}H)")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("|Z| (Ω)"); plt.grid(True, which="both", alpha=.3)

# ---------- Sources (DC only for OP) ----------
@dataclass
class VSource(BaseComponent):
    value: float = 0.0  # V
    def __init__(self, name, nplus, nminus, value_v):
        super().__init__(name, "V", (nplus, nminus))
        self.value = float(value_v)

@dataclass
class ISource(BaseComponent):
    value: float = 0.0  # A (from nplus to nminus)
    def __init__(self, name, nplus, nminus, value_a):
        super().__init__(name, "I", (nplus, nminus))
        self.value = float(value_a)

# ---------- Diode ----------
@dataclass
class Diode(BaseComponent):
    Is: float = 1e-12
    n: float = 1.0
    Vt: float = 0.02585
    def __init__(self, name, anode, cathode, Is=1e-12, n=1.0):
        super().__init__(name, "D", (anode, cathode))
        self.Is = float(Is); self.n = float(n)
    def plot_iv(self):
        v = np.linspace(-0.8, 0.9, 600)
        i = self.Is*(np.exp(np.clip(v, -1, 1)/(self.n*0.02585)) - 1)
        plt.plot(v, i, label=f"{self.name} (Is={self.Is:.1e}, n={self.n})")
        plt.xlabel("Voltage V (V)"); plt.ylabel("Current I (A)"); plt.legend(); plt.grid(True, alpha=.3)

# ---------- MOSFET (simple square-law with channel-length modulation) ----------
@dataclass
class MOSFET(BaseComponent):
    model: str = "NMOS"
    Vth: float = 1.0
    k: float = 1e-3   # A/V^2
    lmbda: float = 0.0
    ptype: bool = False  # True if PMOS
    def __init__(self, name, d, g, s, b, model="NMOS", Vth=1.0, k=1e-3, lmbda=0.0):
        super().__init__(name, "M", (d, g, s, b))
        self.model = model
        self.Vth = float(Vth); self.k = float(k); self.lmbda = float(lmbda)
        self.ptype = model.upper().startswith("PMOS")
    def plot_iv(self):
        # Plot Id vs Vds for several Vgs
        Vgs_list = [0.0, 1.0, 2.0, 3.0] if not self.ptype else [0.0, -1.0, -2.0, -3.0]
        Vds = np.linspace(0, 5, 400) if not self.ptype else np.linspace(-5, 0, 400)
        for Vgs in Vgs_list:
            Id = np.zeros_like(Vds)
            if not self.ptype:
                # NMOS
                for i, vds in enumerate(Vds):
                    if Vgs <= self.Vth:
                        Id[i] = 0.0
                        continue
                    Vdsat = Vgs - self.Vth
                    if vds < Vdsat:
                        Id[i] = self.k*((Vgs - self.Vth)*vds - 0.5*vds*vds)
                    else:
                        Id[i] = 0.5*self.k*(Vgs - self.Vth)**2*(1+self.lmbda*(vds))
            else:
                # PMOS (mirror signs)
                for i, vds in enumerate(Vds):
                    if Vgs >= -self.Vth:
                        Id[i] = 0.0
                        continue
                    Vdsat = Vgs + self.Vth  # negative
                    if vds > Vdsat:
                        Id[i] = -self.k*((-Vgs - self.Vth)*(-vds) - 0.5*vds*vds)
                    else:
                        Id[i] = -0.5*self.k*(-Vgs - self.Vth)**2*(1+self.lmbda*(vds))
            plt.plot(Vds, Id, label=f"Vgs={Vgs} V")
        plt.title(f"{self.name} ({self.model}) Id–Vds")
        plt.xlabel("Vds (V)"); plt.ylabel("Id (A)"); plt.legend(); plt.grid(True, alpha=.3)

# ---------- BJT (simple hybrid: Ic vs Vce for Vbe set; Early effect) ----------
@dataclass
class BJT(BaseComponent):
    model: str = "NPN"
    Is: float = 1e-15
    beta: float = 100.0
    VA: float = 100.0
    ptype: bool = False  # True if PNP
    def __init__(self, name, c, b, e, model="NPN", Is=1e-15, beta=100.0, VA=100.0):
        super().__init__(name, "Q", (c, b, e))
        self.model = model; self.Is = float(Is); self.beta = float(beta); self.VA = float(VA)
        self.ptype = model.upper().startswith("PNP")
    def plot_iv(self):
        Vce = np.linspace(0, 10, 400) if not self.ptype else np.linspace(-10, 0, 400)
        Vbe_list = [0.65, 0.68, 0.70] if not self.ptype else [-0.65, -0.68, -0.70]
        Vt = 0.02585
        for Vbe in Vbe_list:
            Ic0 = self.Is*np.exp(Vbe/(Vt if not self.ptype else -Vt))
            Ic = Ic0*(1 + Vce/self.VA)
            plt.plot(Vce, Ic if not self.ptype else -Ic, label=f"Vbe={Vbe} V")
        plt.title(f"{self.name} ({self.model}) Ic–Vce")
        plt.xlabel("Vce (V)"); plt.ylabel("Ic (A)"); plt.legend(); plt.grid(True, alpha=.3)

# ---------- Helpers ----------
def _parse_value(val_str):
    mult = {"T":1e12,"G":1e9,"M":1e6,"k":1e3,"m":1e-3,"u":1e-6,"n":1e-9,"p":1e-12,"f":1e-15}
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)([TGMkmunpf]?)\s*$", val_str)
    if not m: return float(val_str)
    return float(m.group(1)) * mult.get(m.group(2), 1.0)

def parse_netlist(path) -> List[BaseComponent]:
    comps: List[BaseComponent] = []
    diode_models: Dict[str, Dict] = {}
    mos_models: Dict[str, Dict] = {}
    bjt_models: Dict[str, Dict] = {}

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith(("*",";","//"))]

    # .model
    for s in lines:
        if s.lower().startswith(".model"):
            # .model NAME TYPE (params)
            m = re.match(r"^\.model\s+(\S+)\s+(\S+)\s*\((.*?)\)\s*$", s, re.I)
            if not m: continue
            name, mtype, params = m.group(1), m.group(2).upper(), m.group(3)
            kv = {}
            for tok in re.split(r"[,\s]+", params):
                if "=" in tok:
                    k,v = tok.split("=",1); kv[k.strip()] = v.strip()
            if mtype in ("D",):
                diode_models[name] = {
                    "Is": float(kv.get("Is", 1e-12)),
                    "n": float(kv.get("n", 1.0)),
                }
            elif mtype in ("NMOS","PMOS"):
                mos_models[name] = {
                    "type": mtype,
                    "Vth": float(kv.get("VTO", kv.get("Vth", 1.0))),
                    "k": float(kv.get("KP", kv.get("k", 1e-3))),
                    "lambda": float(kv.get("LAMBDA", kv.get("lambda", 0.0))),
                }
            elif mtype in ("NPN","PNP"):
                bjt_models[name] = {
                    "type": mtype,
                    "Is": float(kv.get("IS", kv.get("Is", 1e-15))),
                    "beta": float(kv.get("BF", kv.get("beta", 100.0))),
                    "VA": float(kv.get("VA", 100.0)),
                }

    # components
    for s in lines:
        sl = s.lower()
        if sl.startswith(".model") or sl.startswith(".end") or sl.startswith(".include"):
            continue
        toks = s.split()
        ref = toks[0]; kind = ref[0].upper()

        if kind == "R" and len(toks) >= 4:
            comps.append(Resistor(ref, toks[1], toks[2], _parse_value(toks[3]))); continue
        if kind == "C" and len(toks) >= 4:
            comps.append(Capacitor(ref, toks[1], toks[2], _parse_value(toks[3]))); continue
        if kind == "L" and len(toks) >= 4:
            comps.append(Inductor(ref, toks[1], toks[2], _parse_value(toks[3]))); continue
        if kind == "V" and len(toks) >= 4:
            comps.append(VSource(ref, toks[1], toks[2], _parse_value(toks[3]))); continue
        if kind == "I" and len(toks) >= 4:
            comps.append(ISource(ref, toks[1], toks[2], _parse_value(toks[3]))); continue
        if kind == "D" and len(toks) >= 3:
            model = toks[3] if len(toks)>=4 else None
            Is, n = 1e-12, 1.0
            if model and model in diode_models:
                Is = diode_models[model]["Is"]; n = diode_models[model]["n"]
            comps.append(Diode(ref, toks[1], toks[2], Is=Is, n=n)); continue
        if kind == "M" and len(toks) >= 6:
            # Mname d g s b model [params ignored for now]
            model = toks[5]
            mm = mos_models.get(model, {"type":"NMOS","Vth":1.0,"k":1e-3,"lambda":0.0})
            comps.append(MOSFET(ref, toks[1], toks[2], toks[3], toks[4],
                                model=mm["type"], Vth=mm["Vth"],
                                k=mm["k"], lmbda=mm["lambda"]))
            continue
        if kind == "Q" and len(toks) >= 5:
            # Qname c b e model
            model = toks[4]
            bm = bjt_models.get(model, {"type":"NPN","Is":1e-15,"beta":100.0,"VA":100.0})
            comps.append(BJT(ref, toks[1], toks[2], toks[3],
                             model=bm["type"], Is=bm["Is"], beta=bm["beta"], VA=bm["VA"]))
            continue

    return comps

# ---------- Schematic data (simple net graph) ----------
def build_netgraph(components: List[BaseComponent]) -> Tuple[List[str], List[Tuple[str,str,str]]]:
    """
    Returns (nodes, edges) where edges are (n1, n2, label).
    """
    edges = []
    nodes = set()
    for c in components:
        if len(c.nodes) >= 2:
            n1, n2 = c.nodes[0], c.nodes[1]
            nodes.add(n1); nodes.add(n2)
            edges.append((n1, n2, c.name))
    return sorted(nodes), edges

# ---------- DC Operating Point (linear MNA: R + V + I only) ----------
def dc_operating_point(components: List[BaseComponent]) -> Dict[str, float]:
    """
    Solve node voltages for resistors and independent DC sources.
    Ground is node '0' or 'gnd' (case-insensitive). Nonlinear devices ignored.
    """
    # Collect nodes
    nodes = set()
    vsources = []
    conductances = []
    currents = []  # current sources
    for c in components:
        if len(c.nodes) < 2: continue
        n1, n2 = c.nodes[0], c.nodes[1]
        nodes.update([n1, n2])
        if c.ctype == "R":
            conductances.append((n1, n2, 1.0/c.value))
        elif c.ctype == "I":  # current from n1 -> n2
            currents.append((n1, n2, c.value))
        elif c.ctype == "V":
            vsources.append((n1, n2, c.value))

    # Map nodes to indices, excluding ground
    def is_gnd(name): return name.lower() in ("0", "gnd", "ground")
    all_nodes = [n for n in sorted(nodes) if not is_gnd(n)]
    N = len(all_nodes); M = len(vsources)
    if N == 0 and M == 0:
        return {}

    node_index = {n:i for i,n in enumerate(all_nodes)}

    # Build MNA matrix
    A = np.zeros((N+M, N+M))
    z = np.zeros(N+M)

    # Stamp resistors (G matrix)
    for n1, n2, g in conductances:
        if not is_gnd(n1):
            i = node_index[n1]; A[i,i] += g
        if not is_gnd(n2):
            j = node_index[n2]; A[j,j] += g
        if (not is_gnd(n1)) and (not is_gnd(n2)):
            i, j = node_index[n1], node_index[n2]
            A[i,j] -= g; A[j,i] -= g

    # Stamp current sources (I vector): +I at n1, -I at n2
    for n1, n2, I in currents:
        if not is_gnd(n1): z[node_index[n1]] += I
        if not is_gnd(n2): z[node_index[n2]] -= I

    # Voltage sources (B, C, D blocks)
    for k, (nplus, nminus, V) in enumerate(vsources):
        row = N + k
        if not is_gnd(nplus):
            i = node_index[nplus]
            A[i, row] += 1.0
            A[row, i] += 1.0
        if not is_gnd(nminus):
            j = node_index[nminus]
            A[j, row] -= 1.0
            A[row, j] -= 1.0
        z[row] = V

    # Solve
    try:
        x = np.linalg.solve(A, z)
    except Exception:
        x, *_ = np.linalg.lstsq(A, z, rcond=None)

    # Pack results
    Vnode = {"0": 0.0, "gnd": 0.0, "GND": 0.0}
    for n in all_nodes:
        Vnode[n] = float(x[node_index[n]])
    return Vnode
