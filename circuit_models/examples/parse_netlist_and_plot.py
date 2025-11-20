import sys
import os
from circuit_models.netlist_parser import parse_netlist

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_netlist_And_plot.py <netlist_file>")
        return

    netlist_path = sys.argv[1]

    if not os.path.isfile(netlist_path):
        print(f"File not found: {netlist_path}")
        return

    try:
        components = parse_netlist(netlist_path)
    except Exception as e:
        print(f"Error parsing netlist: {e}")
        return

    for comp in components:
        print(f"Component: {comp.name} ({comp.__class__.__name__})")
        try:
            if hasattr(comp, 'plot_iv'):
                comp.plot_iv()
            else:
                print(f"Component {comp.name} does not support plotting.")
        except Exception as e:
            print(f"Error plotting {comp.name}: {e}")

if __name__ == "__main__":
    main()
