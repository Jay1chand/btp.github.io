import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, Scrollbar, END
import matplotlib.pyplot as plt

from circuit_models.netlist_parser import parse_netlist

class GUIPlotter:
    def __init__(self, master):
        self.master = master
        master.title("Circuit Component Plotter")

        # Button to select netlist file
        self.load_button = tk.Button(master, text="Load Netlist", command=self.load_netlist)
        self.load_button.pack()

        # Listbox to show components
        self.listbox = Listbox(master, height=10, width=50)
        self.listbox.pack(pady=10)

        # Scrollbar for Listbox
        scrollbar = Scrollbar(master)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        # Plot Button
        self.plot_button = tk.Button(master, text="Plot Selected", command=self.plot_component)
        self.plot_button.pack()

        # Components will be stored after parsing
        self.components = []

    def load_netlist(self):
        filepath = filedialog.askopenfilename(filetypes=[("Netlist Files", "*.cir *.sp *.txt"), ("All files", "*.*")])
        if not filepath:
            return

        try:
            self.components = parse_netlist(filepath)
            self.listbox.delete(0, END)
            for comp in self.components:
                self.listbox.insert(END, f"{comp.name} ({comp.__class__.__name__})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load netlist:\n{e}")

    def plot_component(self):
        selected_index = self.listbox.curselection()
        if not selected_index:
            messagebox.showwarning("No Selection", "Please select a component to plot.")
            return

        comp = self.components[selected_index[0]]

        try:
            if hasattr(comp, 'plot_iv'):
                comp.plot_iv()
            else:
                messagebox.showinfo("No Plot", f"{comp.name} doesn't support plotting.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot component:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    gui = GUIPlotter(root)
    root.mainloop()
