"""
S-curve Modeling Tool
---------------------
Interactive construction of a planar geometric curve by nonlinear optimization.

Mathematical model
    k(s) = a*s^2 + b*s + c
    phi(s) = phi1 + integral_0^s k(t) dt
    x(s) = x1 + integral_0^s cos(phi(t)) dt
    y(s) = y1 + integral_0^s sin(phi(t)) dt

The GUI provides:
    - endpoint/intermediate constraints
    - sliders + numeric entries for the intermediate constraints
    - BFGS and L-BFGS-B optimization
    - separate visualization of the curve, tangent angle,
      curvature, and curvature derivative

Requirements:
    numpy
    scipy
    matplotlib
    tkinter
"""

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import quad
from scipy.optimize import Bounds, minimize


# ---------------------------------------------------------------------------
# Mathematical model
# ---------------------------------------------------------------------------

def curvature(s, a, b, c):
    """Return k(s) = a*s^2 + b*s + c."""
    return a * s**2 + b * s + c


def tangent_angle(s, a, b, c, phi1):
    """Return the tangent angle phi(s)."""
    return phi1 + a * s**3 / 3.0 + b * s**2 / 2.0 + c * s


def curve_x(s, a, b, c, x1, phi1):
    """Return x(s) by numerical integration."""
    value, _ = quad(
        lambda t: np.cos(tangent_angle(t, a, b, c, phi1)),
        0.0,
        s,
    )
    return x1 + value


def curve_y(s, a, b, c, y1, phi1):
    """Return y(s) by numerical integration."""
    value, _ = quad(
        lambda t: np.sin(tangent_angle(t, a, b, c, phi1)),
        0.0,
        s,
    )
    return y1 + value


def target_bfgs(parameters, p):
    """Objective for the unconstrained BFGS formulation."""
    a, b, c, s_p, s_end = parameters

    dx_end = p.x2 - p.x1 - (
        curve_x(s_end, a, b, c, p.x1, p.phi1) - p.x1
    )
    dy_end = p.y2 - p.y1 - (
        curve_y(s_end, a, b, c, p.y1, p.phi1) - p.y1
    )
    dphi_end = p.phi2 - tangent_angle(
        s_end, a, b, c, p.phi1
    )

    dx_p = p.xp - p.x1 - (
        curve_x(s_p, a, b, c, p.x1, p.phi1) - p.x1
    )
    dphi_p = p.phip - tangent_angle(
        s_p, a, b, c, p.phi1
    )

    return dx_end**2 + dy_end**2 + dphi_end**2 + dx_p**2 + dphi_p**2


def target_lbfgsb(parameters, p):
    """Objective for the bounded L-BFGS-B formulation."""
    a, b, c, fraction, s_end = parameters
    s_p = fraction * s_end

    dx_end = p.x2 - p.x1 - (
        curve_x(s_end, a, b, c, p.x1, p.phi1) - p.x1
    )
    dy_end = p.y2 - p.y1 - (
        curve_y(s_end, a, b, c, p.y1, p.phi1) - p.y1
    )
    dphi_end = p.phi2 - tangent_angle(
        s_end, a, b, c, p.phi1
    )

    dx_p = p.xp - p.x1 - (
        curve_x(s_p, a, b, c, p.x1, p.phi1) - p.x1
    )
    dphi_p = p.phip - tangent_angle(
        s_p, a, b, c, p.phi1
    )

    return dx_end**2 + dy_end**2 + dphi_end**2 + dx_p**2 + dphi_p**2


class Parameters:
    """Input parameters defining the curve constraints."""

    def __init__(
        self,
        x1=0.0,
        y1=0.0,
        phi1=0.0,
        x2=2.75,
        y2=1.0,
        phi2=0.0,
        xp=0.7,
        phip=0.20944,
    ):
        self.x1 = x1
        self.y1 = y1
        self.phi1 = phi1
        self.x2 = x2
        self.y2 = y2
        self.phi2 = phi2
        self.xp = xp
        self.phip = phip


def solve_curve(p, method):
    """Solve for curve parameters using the selected optimizer."""
    endpoint_distance = np.hypot(p.x2 - p.x1, p.y2 - p.y1)
    x0 = np.array([0.0, 0.0, 0.0, 0.5 * endpoint_distance, endpoint_distance])

    if method == "BFGS":
        result = minimize(
            target_bfgs,
            x0,
            args=(p,),
            method="BFGS",
        )
    elif method == "L-BFGS-B":
        d0 = abs(p.xp - p.x1)
        d1 = np.hypot(p.y2 - p.y1, p.x2 - p.x1)
        d2 = 1.2 * d1

        # fraction = s_p / s_end
        lower_fraction = d0 / d2 if d2 > 0 else 0.0

        bounds = Bounds(
            [-np.inf, -np.inf, -np.inf, lower_fraction, d1],
            [np.inf, np.inf, np.inf, 1.0, d2],
        )

        result = minimize(
            target_lbfgsb,
            x0,
            args=(p,),
            method="L-BFGS-B",
            bounds=bounds,
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return result


def evaluate_curve(parameters, p, samples=200):
    """Evaluate x, y, phi, k and k' on the solved parameterization."""
    a, b, c, s_p, s_end = parameters

    s_values = np.linspace(0.0, s_end, samples)

    x_values = np.array([
        curve_x(s, a, b, c, p.x1, p.phi1) for s in s_values
    ])
    y_values = np.array([
        curve_y(s, a, b, c, p.y1, p.phi1) for s in s_values
    ])
    phi_values = np.array([
        tangent_angle(s, a, b, c, p.phi1) for s in s_values
    ])
    k_values = curvature(s_values, a, b, c)
    k_derivative = 2.0 * a * s_values + b

    return s_values, x_values, y_values, phi_values, k_values, k_derivative


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CurveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("S-Curve Modeling Tool")
        self.root.geometry("980x680")
        self.root.minsize(900, 620)

        self.parameters = Parameters()
        self.result = None

        self.plot_window = None
        self.canvas = None
        self.figure = None
        self.axes = None
        self._closing = False

        self._configure_style()
        self._build_controls()

        # IMPORTANT:
        # Closing the main window through X uses the same cleanup as Exit.
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

    def _configure_style(self):
        style = ttk.Style(self.root)

        # Use a platform-native ttk theme when available.
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.configure("Value.TLabel", font=("Consolas", 10))
        style.configure("Small.TLabel", font=("Segoe UI", 9))

    def _build_controls(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        # Header
        ttk.Label(
            outer,
            text="S-Curve Modeling Tool",
            style="Title.TLabel",
        ).pack(anchor="w")
        """
        ttk.Label(
            outer,
            text=(
                "Define geometric constraints, optimize the curve, "
                "then inspect its shape, tangent angle and curvature."
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 16))
        """
        formula_frame = tk.Frame(outer)
        formula_frame.pack(fill="x", padx=10, pady=2)

        formulas = [
            "k(s) = a·s² + b·s + c",
            "φ(s) = φ₁ + ∫₀ˢ k(t) dt",
            "x(s) = x₁ + ∫₀ˢ cos(φ(t)) dt" ,
            "y(s) = y₁ + ∫₀ˢ sin(φ(t)) dt"
        ]

        for row, text in enumerate(formulas):
            label = tk.Label(
                formula_frame,
                text=text,
                font=("Arial", 11),
                anchor="w"
            )
            label.grid(
                row=row,
                column=0,
                sticky="w",
                padx=0,
                pady=1
            )
            #label.pack(anchor="center", pady=0)

        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # ---------------------------------------------------------------
        # Left: geometric constraints
        # ---------------------------------------------------------------
        input_frame = ttk.LabelFrame(
            content,
            text="Geometric constraints",
            style="Section.TLabelframe",
            padding=14,
        )
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        input_frame.columnconfigure(1, weight=1)

        fields = [
            ("Start X", "x1"),
            ("Start Y", "y1"),
            ("Start angle φ₁", "phi1"),
            ("End X", "x2"),
            ("End Y", "y2"),
            ("End angle φ₂", "phi2"),
        ]
        fields = [
            ("x1", "x1"),
            ("y1", "y1"),
            ("φ₁", "phi1"),
            ("x2", "x2"),
            ("y2", "y2"),
            ("φ₂", "phi2"),
        ]

        self.entries = {}

        for row, (label, attribute) in enumerate(fields):
            ttk.Label(
                input_frame,
                text=label,
            ).grid(row=row, column=0, padx=(0, 10), pady=5, sticky="w")

            variable = tk.StringVar(
                value=str(getattr(self.parameters, attribute))
            )
            self.entries[attribute] = variable

            ttk.Entry(
                input_frame,
                textvariable=variable,
                width=16,
            ).grid(row=row, column=1, padx=0, pady=5, sticky="ew")

        ttk.Separator(input_frame).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=12
        )

        ttk.Label(
            input_frame,
            text="Intermediate point",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 5))

        # xp slider
        self.xp_var = tk.DoubleVar(value=self.parameters.xp)
        self.xp_entry_var = tk.StringVar(value=f"{self.parameters.xp:.5g}")

        ttk.Label(input_frame, text="Point Xₚ").grid(
            row=8, column=0, padx=(0, 10), pady=5, sticky="w"
        )

        xp_frame = ttk.Frame(input_frame)
        xp_frame.grid(row=8, column=1, sticky="ew")
        xp_frame.columnconfigure(0, weight=1)

        self.xp_scale = ttk.Scale(
            xp_frame,
            from_=0.0,
            to=max(self.parameters.x2, 1.0),
            variable=self.xp_var,
            command=self._xp_slider_changed,
        )
        self.xp_scale.grid(row=0, column=0, sticky="ew")

        ttk.Entry(
            xp_frame,
            textvariable=self.xp_entry_var,
            width=9,
        ).grid(row=0, column=1, padx=(8, 0))

        self.xp_entry_var.trace_add("write", self._xp_entry_changed)

        # phip slider
        self.phip_var = tk.DoubleVar(value=self.parameters.phip)
        self.phip_entry_var = tk.StringVar(value=f"{self.parameters.phip:.5g}")

        ttk.Label(input_frame, text="Point angle φₚ").grid(
            row=9, column=0, padx=(0, 10), pady=5, sticky="w"
        )

        phip_frame = ttk.Frame(input_frame)
        phip_frame.grid(row=9, column=1, sticky="ew")
        phip_frame.columnconfigure(0, weight=1)

        self.phip_scale = ttk.Scale(
            phip_frame,
            from_=-np.pi,
            to=np.pi,
            variable=self.phip_var,
            command=self._phip_slider_changed,
        )
        self.phip_scale.grid(row=0, column=0, sticky="ew")

        ttk.Entry(
            phip_frame,
            textvariable=self.phip_entry_var,
            width=9,
        ).grid(row=0, column=1, padx=(8, 0))

        self.phip_entry_var.trace_add("write", self._phip_entry_changed)
        """"
        ttk.Label(
            input_frame,
            text="Angles are in radians.",
            style="Small.TLabel",
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 0))
        """
        # ---------------------------------------------------------------
        # Right: optimization + result
        # ---------------------------------------------------------------
        right = ttk.Frame(content)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(1, weight=1)

        method_frame = ttk.LabelFrame(
            right,
            text="Optimization",
            style="Section.TLabelframe",
            padding=14,
        )
        method_frame.grid(row=0, column=0, sticky="ew")
        method_frame.columnconfigure(1, weight=1)

        ttk.Label(
            method_frame,
            text="Method",
        ).grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")

        self.method = tk.StringVar(value="BFGS")
        ttk.Combobox(
            method_frame,
            textvariable=self.method,
            values=("BFGS", "L-BFGS-B"),
            state="readonly",
            width=16,
        ).grid(row=0, column=1, pady=5, sticky="w")
        """
        ttk.Label(
            method_frame,
            text="BFGS: unconstrained  |  L-BFGS-B: bounded",
            style="Small.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        """
        result_frame = ttk.LabelFrame(
            right,
            text="Optimized parameters",
            style="Section.TLabelframe",
            padding=14,
        )
        result_frame.grid(row=1, column=0, sticky="nsew", pady=(14, 0))

        result_frame.columnconfigure(1, weight=1)

        self.result_variables = {}

        for row, name in enumerate(("a", "b", "c", "s_p", "S"), start=0):
            ttk.Label(
                result_frame,
                text=name,
            ).grid(row=row, column=0, padx=(0, 10), pady=6, sticky="w")

            variable = tk.StringVar(value="—")
            self.result_variables[name] = variable

            ttk.Label(
                result_frame,
                textvariable=variable,
                style="Value.TLabel",
            ).grid(row=row, column=1, pady=6, sticky="w")

        ttk.Separator(result_frame).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=10
        )

        self.status_variable = tk.StringVar(
            value="Ready. Set the constraints and press Solve."
        )

        ttk.Label(
            result_frame,
            textvariable=self.status_variable,
            style="Small.TLabel",
            wraplength=350,
        ).grid(row=6, column=0, columnspan=2, sticky="w")

        # ---------------------------------------------------------------
        # Bottom action bar
        # ---------------------------------------------------------------
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(16, 0))

        ttk.Button(
            actions,
            text="Solve and Plot",
            style="Action.TButton",
            command=self.solve_and_plot,
        ).pack(side="left")

        ttk.Button(
            actions,
            text="Close Plot",
            command=self.close_plot_window,
        ).pack(side="left", padx=(10, 0))

        ttk.Button(
            actions,
            text="Exit",
            command=self.close_application,
        ).pack(side="right")

    # -------------------------------------------------------------------
    # Slider synchronization
    # -------------------------------------------------------------------

    def _xp_slider_changed(self, _value):
        value = self.xp_var.get()
        self.xp_entry_var.set(f"{value:.5g}")

    def _phip_slider_changed(self, _value):
        value = self.phip_var.get()
        self.phip_entry_var.set(f"{value:.5g}")

    def _xp_entry_changed(self, *_args):
        try:
            value = float(self.xp_entry_var.get())
        except ValueError:
            return

        low = float(self.xp_scale.cget("from"))
        high = float(self.xp_scale.cget("to"))
        value = min(max(value, low), high)

        # Avoid unnecessary trace recursion.
        if abs(self.xp_var.get() - value) > 1e-12:
            self.xp_var.set(value)

    def _phip_entry_changed(self, *_args):
        try:
            value = float(self.phip_entry_var.get())
        except ValueError:
            return

        value = min(max(value, -np.pi), np.pi)

        if abs(self.phip_var.get() - value) > 1e-12:
            self.phip_var.set(value)

    # -------------------------------------------------------------------
    # Parameters and solving
    # -------------------------------------------------------------------

    def read_parameters(self):
        """Read and validate numerical values from the GUI."""
        values = {}

        for attribute, variable in self.entries.items():
            try:
                values[attribute] = float(variable.get())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value for {attribute}: {variable.get()}"
                ) from exc

        try:
            values["xp"] = float(self.xp_entry_var.get())
            values["phip"] = float(self.phip_entry_var.get())
        except ValueError as exc:
            raise ValueError("Invalid intermediate-point value.") from exc

        if values["x1"] == values["x2"] and values["y1"] == values["y2"]:
            raise ValueError("Start and end points must be different.")

        return Parameters(**values)

    def solve_and_plot(self):
        try:
            p = self.read_parameters()
            method = self.method.get()
            result = solve_curve(p, method)
        except (ValueError, FloatingPointError) as exc:
            messagebox.showerror("Input error", str(exc), parent=self.root)
            return

        self.parameters = p
        self.result = result

        a, b, c, s_p_or_fraction, s_end = result.x

        # For L-BFGS-B the fourth variable is a fraction, not s_p.
        if method == "L-BFGS-B":
            s_p = s_p_or_fraction * s_end
        else:
            s_p = s_p_or_fraction

        self.result_variables["a"].set(f"{a:.8g}")
        self.result_variables["b"].set(f"{b:.8g}")
        self.result_variables["c"].set(f"{c:.8g}")
        self.result_variables["s_p"].set(f"{s_p:.8g}")
        self.result_variables["S"].set(f"{s_end:.8g}")

        self.status_variable.set(
            f"Success: {result.success}    "
            f"Objective: {result.fun:.4e}\n"
            f"{result.message}"
        )

        self.plot_solution()

        print(f"Optimization success: {result.success}")
        print(f"Objective value: {result.fun:.8e}")
        print(f"Message: {result.message}")

    # -------------------------------------------------------------------
    # Plot window
    # -------------------------------------------------------------------

    def plot_solution(self):
        if self.plot_window is None or not self.plot_window.winfo_exists():
            self.plot_window = tk.Toplevel(self.root)
            self.plot_window.title("S-Curve — Analysis")
            self.plot_window.geometry("900x700")
            self.plot_window.minsize(760, 580)

            # Closing the plot with X uses our cleanup method.
            self.plot_window.protocol(
                "WM_DELETE_WINDOW",
                self.close_plot_window,
            )

            self.figure, self.axes = plt.subplots(
                2, 2,
                figsize=(8, 6),
                dpi=100,
            )

            self.canvas = FigureCanvasTkAgg(
                self.figure,
                master=self.plot_window,
            )
            self.canvas.get_tk_widget().pack(
                fill="both",
                expand=True,
                padx=8,
                pady=8,
            )
        else:
            for axis in self.axes.flat:
                axis.clear()

        p = self.parameters
        values = evaluate_curve(self.result.x, p)

        s, x, y, phi, k, k_prime = values

        self.axes[0, 0].plot(x, y)
        self.axes[0, 0].set_title("Geometric curve")
        self.axes[0, 0].set_xlabel("x")
        self.axes[0, 0].set_ylabel("y")
        self.axes[0, 0].axis("equal")
        self.axes[0, 0].grid(True, alpha=0.3)

        self.axes[0, 1].plot(s, phi)
        self.axes[0, 1].set_title("Tangent angle φ(s)")
        self.axes[0, 1].set_xlabel("s")
        self.axes[0, 1].set_ylabel("φ")
        self.axes[0, 1].grid(True, alpha=0.3)

        self.axes[1, 0].plot(s, k)
        self.axes[1, 0].set_title("Curvature k(s)")
        self.axes[1, 0].set_xlabel("s")
        self.axes[1, 0].set_ylabel("k")
        self.axes[1, 0].grid(True, alpha=0.3)

        self.axes[1, 1].plot(s, k_prime)
        self.axes[1, 1].set_title("Curvature derivative k′(s)")
        self.axes[1, 1].set_xlabel("s")
        self.axes[1, 1].set_ylabel("k′")
        self.axes[1, 1].grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def close_plot_window(self):
        """Completely close the embedded Matplotlib figure/window."""
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        if self.figure is not None:
            plt.close(self.figure)

        if self.plot_window is not None:
            try:
                if self.plot_window.winfo_exists():
                    self.plot_window.destroy()
            except tk.TclError:
                pass

        self.plot_window = None
        self.canvas = None
        self.figure = None
        self.axes = None

    # -------------------------------------------------------------------
    # Application shutdown
    # -------------------------------------------------------------------

    def close_application(self):
        """
        Cleanly terminate the Tkinter/Matplotlib application.

        This is used both by the Exit button and by the main-window X.
        """
        if self._closing:
            return

        self._closing = True

        self.close_plot_window()

        # Close any remaining pyplot figures belonging to this process.
        plt.close("all")

        # Stop Tk's event loop first, then destroy the root window.
        try:
            self.root.quit()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = CurveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
