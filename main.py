"""
Data Analyzer Pro
=======================
Copyright (c) 2025 Yehonatan Smaja

Description:
A comprehensive GUI tool for analyzing experimental data.
Developed as part of M.Sc. research in Applied Physics at Bar-Ilan University.
Features include automated signal processing, FFT analysis, curve fitting, and calculations.

License:
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Credits:
Concept & Physics Logic: Yehonatan Smaja
Implementation Assistance: AI Tools & Automation
"""

import os
import sys
import codecs
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
from pathlib import Path

# External Libraries
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, filtfilt, cheby1, sosfilt, sosfreqz, sosfilt_zi, savgol_filter
from scipy.optimize import curve_fit
import numexpr as ne
import ctypes

class ToolTip:
    """Lightweight tooltip: appears on <Enter>, disappears on <Leave>."""
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, *_):
        if self.tip_window or not self.text:
            return
        # screen position: slightly below/right of the widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)            # remove window decorations
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#FFFFE0", relief="solid",
            borderwidth=1, font=("TkDefaultFont", 9)
        )
        label.pack(ipadx=4, ipady=2)

    def _hide(self, *_):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

def _read_table_auto_encoding(path, **pd_kwargs):
    """
    Robust wrapper around pandas.read_csv():
      • detects UTF‑16 BOM
      • tries utf‑8 → cp1252 → latin1
    """
    import pandas as _pd
    with open(path, "rb") as f:
        if f.read(2) in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
            return _pd.read_csv(path, encoding="utf‑16", **pd_kwargs)

    for enc in ("utf‑8", "cp1252", "latin1"):
        try:
            return _pd.read_csv(path, encoding=enc, **pd_kwargs)
        except UnicodeDecodeError:
            continue
    # last attempt will raise the UnicodeDecodeError
    return _pd.read_csv(path, encoding="utf‑8", **pd_kwargs)

def resource_path(rel_path: str) -> Path:
    """Return absolute path to resource (works also inside PyInstaller EXE)."""
    if hasattr(sys, "_MEIPASS"):           # running from bundled exe
        return Path(sys._MEIPASS) / rel_path   # type: ignore[attr-defined]
    return Path(__file__).parent / rel_path

class DataAnalyzerApp:

    def __init__(self):
        try:
            myappid = 'yehonatan.data.analyzer.pro.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
        # appearance
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # window
        self.window = ctk.CTk()
        self.window.title("Data Analyzer Pro")
        self.window.geometry("1280x800")
        self.window.minsize(900, 600)

        # -------- Dynamic Icon Loading  --------
        # This creates a path relative to the script location
        icon_path = resource_path("icon.ico")

        try:
            if icon_path.exists():
                self.window.iconbitmap(default=str(icon_path))
            else:
                print(f"Icon not found at {icon_path}")
        except Exception as e:
            print(f"Error loading icon: {e}")

        # -------- instance variables --------
        self.data = None
        self.active_channels = {}
        self.channel_names = []
        self.fig = None
        self.canvas_widget = None
        self.filter_frame_vars = None
        self.channel_conditioning = {}
        self.current_file_path = None
        self.undo_stack = []
        self.redo_stack = []
        self.show_sidebar_var = tk.BooleanVar(value=True)
        self.show_status_var = tk.BooleanVar(value=True)
        self.navigation_toolbar = None

        # build GUI
        self.create_menu_bar()
        self.create_layout()
        self.create_status_bar()  # Must be after create_layout to use grid
        self.window.mainloop()

    # ---- Fitting helpers ----
    @staticmethod
    def _compute_r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    def _build_fit_function(self, expr: str):
        # Build a python function f(x, *p) that evaluates the user's expression.
        # Variables available: x, p0, p1, p2, ... and numpy as np
        # Also support common math shorthands: sin, cos, tan, exp, log, ln, log10, sqrt, abs, pi, e, and hyperbolic
        def f(x, *params):
            local = {f"p{i}": params[i] for i in range(len(params))}
            local.update({
                "x": x,
                "np": np,
                # trig and hyperbolic
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
                "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
                # exponentials and logs
                "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
                # misc
                "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e
            })
            expr_pre = expr.replace('^', '**')
            return eval(expr_pre, {"__builtins__": {}}, local)
        return f

    def run_fit(self):
        if not hasattr(self, 'last_x') or not hasattr(self, 'last_y'):
            messagebox.showerror("Fitting", "Please generate a plot first.")
            return
        expr = self.fit_func_entry.get().strip()
        if not expr:
            messagebox.showerror("Fitting", "Please enter a function expression.")
            return
        guess_text = self.fit_guess_entry.get().strip()
        try:
            p0 = [float(s) for s in guess_text.split(',')] if guess_text else []
        except Exception:
            messagebox.showerror("Fitting", "Initial guess must be comma-separated numbers.")
            return

        target = self.fit_target_menu.get()
        x = np.asarray(self.last_x, dtype=float)
        if target.startswith("Primary"):
            y = np.asarray(self.last_y, dtype=float)
        else:
            if not hasattr(self, 'last_y2') or self.last_y2 is None:
                messagebox.showerror("Fitting", "No secondary (Y2) data available.")
                return
            y = np.asarray(self.last_y2, dtype=float)

        try:
            func = self._build_fit_function(expr)
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=20000)
            y_fit = func(x, *popt)
            r2 = self._compute_r2(y, y_fit)
        except Exception as e:
            messagebox.showerror("Fitting", f"Fit failed: {e}")
            return

        # Overlay fit on current plot
        try:
            ax = plt.gcf().get_axes()[0]
            ax.plot(x, y_fit, color="#AA00FF", linewidth=2, linestyle="-", label="Fit")
            if self.show_legend_var.get() if hasattr(self, 'show_legend_var') else True:
                ax.legend(loc='upper right')
            plt.gcf().canvas.draw_idle()
        except Exception:
            pass

        # Show R^2 and params
        params_text = ", ".join(f"p{i}={v:.3g}" for i, v in enumerate(popt))
        self.fit_result_label.configure(text=f"R^2: {r2:.4f}   ({params_text})")

    def auto_fit(self):
        if not hasattr(self, 'last_x') or not hasattr(self, 'last_y'):
            messagebox.showerror("Auto Fit", "Please generate a plot first.")
            return
        target = self.fit_target_menu.get()
        x = np.asarray(self.last_x, dtype=float)
        if target.startswith("Primary"):
            y = np.asarray(self.last_y, dtype=float)
        else:
            if not hasattr(self, 'last_y2') or self.last_y2 is None:
                messagebox.showerror("Auto Fit", "No secondary (Y2) data available.")
                return
            y = np.asarray(self.last_y2, dtype=float)

        # Candidate models: (name, expression string template, initial guess function)
        candidates = []
        # Linear: p0*x + p1
        candidates.append(("Linear", "p0*x + p1", lambda x,y: [1.0, float(np.median(y))]))
        # Quadratic: p0*x**2 + p1*x + p2
        candidates.append(("Quadratic", "p0*x**2 + p1*x + p2", lambda x,y: [1.0, 1.0, float(np.median(y))]))
        # Cubic
        candidates.append(("Cubic", "p0*x**3 + p1*x**2 + p2*x + p3", lambda x,y: [1.0, 1.0, 1.0, float(np.median(y))]))
        # Exponential: p0 * exp(p1*x) + p2
        candidates.append(("Exponential", "p0*np.exp(p1*x) + p2", lambda x,y: [y.max()-y.min()+1e-6, -1.0/(x.ptp()+1e-9), float(y.min())]))
        # Power: p0 * x**p1 + p2 (avoid x<=0)
        if np.all(x > 0):
            candidates.append(("Power", "p0 * x**p1 + p2", lambda x,y: [1.0, 1.0, float(np.median(y))]))
        # Logarithmic: p0 * log(x) + p1 (avoid x<=0)
        if np.all(x > 0):
            candidates.append(("Log", "p0*np.log(x) + p1", lambda x,y: [1.0, float(np.median(y))]))

        # Sine: p0 * sin(2*pi*p1*x + p2) + p3 (amplitude, frequency[Hz], phase, offset)
        def _guess_sine(x_arr, y_arr):
            y0 = y_arr - np.mean(y_arr)
            n = len(x_arr)
            if n < 4:
                return [1.0, 1.0/(x_arr.ptp()+1e-9), 0.0, float(np.mean(y_arr))]
            # estimate frequency via FFT
            dt = np.median(np.diff(x_arr)) if len(x_arr) > 1 else 1.0
            fs = 1.0 / max(dt, 1e-12)
            # resample not needed; use uneven spacing approx via mean dt
            fft_vals = np.fft.rfft(y0)
            freqs = np.fft.rfftfreq(len(y0), d=1/fs)
            k = np.argmax(np.abs(fft_vals[1:])) + 1 if len(fft_vals) > 1 else 1
            f0 = max(freqs[k], 1.0/(x_arr.ptp()+1e-9)) if k < len(freqs) else 1.0/(x_arr.ptp()+1e-9)
            amp = 0.5 * (np.max(y_arr) - np.min(y_arr))
            phase = 0.0
            offset = float(np.mean(y_arr))
            return [amp if amp != 0 else 1.0, f0, phase, offset]
        candidates.append(("Sine", "p0*np.sin(2*pi*p1*x + p2) + p3", _guess_sine))

        # Damped Sine: p0 * exp(-p1*x) * sin(2*pi*p2*x + p3) + p4
        def _guess_damped_sine(x_arr, y_arr):
            a, f, ph, c = _guess_sine(x_arr, y_arr)
            return [a, 0.0, f, ph, c]
        candidates.append(("Damped Sine", "p0*np.exp(-p1*x)*np.sin(2*pi*p2*x + p3) + p4", _guess_damped_sine))

        # Rational: p0 / (1 + p1*x) + p2
        candidates.append(("Rational", "p0 / (1 + p1*x) + p2", lambda x,y: [1.0, 1.0/(np.mean(np.abs(x))+1e-9), float(np.median(y))]))

        best = None  # (r2, name, expr, popt, yfit)
        for name, expr, guess_fn in candidates:
            try:
                p0 = guess_fn(x, y)
                func = self._build_fit_function(expr)
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
                y_fit = func(x, *popt)
                r2 = self._compute_r2(y, y_fit)
                if best is None or r2 > best[0]:
                    best = (r2, name, expr, popt, y_fit)
            except Exception:
                continue

        if best is None:
            messagebox.showerror("Auto Fit", "No model could be fitted to the data.")
            return

        r2, name, expr, popt, y_fit = best
        # Update UI fields with the best model
        self.fit_func_entry.delete(0, 'end')
        self.fit_func_entry.insert(0, expr)
        self.fit_guess_entry.delete(0, 'end')
        self.fit_guess_entry.insert(0, ", ".join(f"{v:.6g}" for v in popt))

        # Overlay and update R^2
        try:
            ax = plt.gcf().get_axes()[0]
            ax.plot(x, y_fit, color="#AA00FF", linewidth=2, linestyle="-", label=f"Auto {name} fit")
            if self.show_legend_var.get() if hasattr(self, 'show_legend_var') else True:
                ax.legend(loc='upper right')
            plt.gcf().canvas.draw_idle()
        except Exception:
            pass

        params_text = ", ".join(f"p{i}={v:.3g}" for i, v in enumerate(popt))
        self.fit_result_label.configure(text=f"R^2: {r2:.4f}   ({name})  [{params_text}]")

    def create_menu_bar(self):
        """Create main menu bar with File, Edit, View, Help menus"""
        # Note: CustomTkinter doesn't have native menu support, so we use tkinter Menu
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open File...", command=self.load_file, accelerator="Ctrl+O")
        file_menu.add_separator()

        # Recent Files submenu
        self.recent_files_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self.recent_files_menu)
        self.update_recent_files()

        file_menu.add_separator()
        file_menu.add_command(label="Save Project...", command=self.save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Project...", command=self.load_project, accelerator="Ctrl+Shift+O")
        file_menu.add_separator()

        # Export submenu
        export_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export Plot...", command=self.export_plot_dialog)
        export_menu.add_command(label="Export Data...", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.quit, accelerator="Alt+F4")

        # Edit Menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        self.undo_menu_item = edit_menu.add_command(label="Undo", command=self.undo_action, accelerator="Ctrl+Z", state="disabled")
        self.redo_menu_item = edit_menu.add_command(label="Redo", command=self.redo_action, accelerator="Ctrl+Y", state="disabled")
        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences...", command=self.open_preferences)

        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Sidebar", variable=self.show_sidebar_var, command=self.toggle_sidebar)
        view_menu.add_checkbutton(label="Show Status Bar", variable=self.show_status_var, command=self.toggle_status_bar)
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset View", command=self.reset_plot_view, accelerator="Ctrl+0")
        view_menu.add_separator()
        view_menu.add_command(label="Data Browser", command=self.open_data_browser, accelerator="Ctrl+B")

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

        # Bind keyboard shortcuts
        self.window.bind("<Control-o>", lambda e: self.load_file())
        self.window.bind("<Control-O>", lambda e: self.load_file())
        self.window.bind("<Control-s>", lambda e: self.save_project())
        self.window.bind("<Control-S>", lambda e: self.save_project())
        self.window.bind("<Control-z>", lambda e: self.undo_action())
        self.window.bind("<Control-Z>", lambda e: self.undo_action())
        self.window.bind("<Control-y>", lambda e: self.redo_action())
        self.window.bind("<Control-Y>", lambda e: self.redo_action())
        self.window.bind("<Control-plus>", lambda e: self.zoom_in())
        self.window.bind("<Control-equal>", lambda e: self.zoom_in())
        self.window.bind("<Control-minus>", lambda e: self.zoom_out())
        self.window.bind("<Control-0>", lambda e: self.reset_plot_view())
        self.window.bind("<Control-b>", lambda e: self.open_data_browser())
        self.window.bind("<Control-B>", lambda e: self.open_data_browser())

    def create_status_bar(self):
        """Create status bar at the bottom of the window"""
        self.status_frame = ctk.CTkFrame(self.window, height=25, corner_radius=0)
        # Use grid instead of pack to be compatible with main window grid layout
        self.status_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.status_frame.grid_propagate(False)

        # Configure grid columns for status bar
        self.status_frame.grid_columnconfigure(1, weight=1)  # Middle column expands

        # Left side - Main status
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            anchor="w",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=2, sticky="w")

        # Middle - Data info
        self.data_info_label = ctk.CTkLabel(
            self.status_frame,
            text="No data loaded",
            anchor="w",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray70")
        )
        self.data_info_label.grid(row=0, column=1, padx=10, pady=2, sticky="ew")

        # Right side - Memory/Progress
        self.memory_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            anchor="e",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray70")
        )
        self.memory_label.grid(row=0, column=2, padx=10, pady=2, sticky="e")

        # Progress bar (hidden by default)
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=200)
        self.progress_bar.grid(row=0, column=3, padx=10, pady=2, sticky="e")
        self.progress_bar.grid_remove()

        # Start memory monitoring
        self.update_memory_usage()

    # ========== Menu and Status Bar Helper Functions ==========

    def update_status(self, message, data_info=None):
        """Update status bar message"""
        self.status_label.configure(text=message)
        if data_info:
            self.data_info_label.configure(text=data_info)

    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.configure(text=f"Memory: {memory_mb:.1f} MB")
        except ImportError:
            self.memory_label.configure(text="")
        except Exception:
            pass
        # Update every 5 seconds
        self.window.after(5000, self.update_memory_usage)

    def show_progress(self, value):
        """Show progress bar"""
        self.progress_bar.grid(row=0, column=3, padx=10, pady=2, sticky="e")
        self.progress_bar.set(value)
        self.window.update_idletasks()

    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.grid_remove()

    def toggle_sidebar(self):
        """Toggle sidebar visibility"""
        if self.show_sidebar_var.get():
            self.sidebar_frame.grid()
        else:
            self.sidebar_frame.grid_remove()

    def toggle_status_bar(self):
        """Toggle status bar visibility"""
        if self.show_status_var.get():
            self.status_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        else:
            self.status_frame.grid_remove()

    def zoom_in(self):
        """Zoom in on plot"""
        if self.fig and hasattr(self, 'navigation_toolbar') and self.navigation_toolbar:
            # Navigation toolbar handles this
            pass

    def zoom_out(self):
        """Zoom out on plot"""
        if self.fig and hasattr(self, 'navigation_toolbar') and self.navigation_toolbar:
            # Navigation toolbar handles this
            pass

    def reset_plot_view(self):
        """Reset plot view to default"""
        if self.fig:
            for ax in self.fig.get_axes():
                ax.relim()
                ax.autoscale()
            if hasattr(self, 'canvas_widget') and self.canvas_widget:
                self.canvas_widget.draw()

    def undo_action(self):
        """Undo last action"""
        if self.undo_stack:
            action = self.undo_stack.pop()
            self.redo_stack.append(action)
            if self.undo_stack:
                self.undo_menu_item.config(state="normal")
            else:
                self.undo_menu_item.config(state="disabled")
            if self.redo_stack:
                self.redo_menu_item.config(state="normal")

    def redo_action(self):
        """Redo last undone action"""
        if self.redo_stack:
            action = self.redo_stack.pop()
            self.undo_stack.append(action)
            if self.undo_stack:
                self.undo_menu_item.config(state="normal")
            if self.redo_stack:
                self.redo_menu_item.config(state="normal")
            else:
                self.redo_menu_item.config(state="disabled")

    def open_preferences(self):
        """Open preferences dialog"""
        messagebox.showinfo("Preferences", "Preferences dialog coming soon!")

    def show_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        shortcuts_text = """Keyboard Shortcuts:

File Operations:
  Ctrl+O          Open File
  Ctrl+S          Save Project
  Ctrl+Shift+O    Load Project
  Alt+F4          Exit

Edit Operations:
  Ctrl+Z          Undo
  Ctrl+Y          Redo

View Operations:
  Ctrl++          Zoom In
  Ctrl+-          Zoom Out
  Ctrl+0          Reset View
  Ctrl+B          Data Browser"""

        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Keyboard Shortcuts")
        dialog.geometry("400x300")
        dialog.transient(self.window)

        text_widget = tk.Text(dialog, wrap="word", font=("Courier", 10), padx=10, pady=10)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", shortcuts_text)
        text_widget.config(state="disabled")

        close_btn = ctk.CTkButton(dialog, text="Close", command=dialog.destroy)
        close_btn.pack(pady=10)

    def show_about(self):
        """Show about dialog"""
        about_text = """Data Analyzer Pro v1.0.0

A comprehensive GUI tool for analyzing experimental data.
Developed as part of M.Sc. research in Applied Physics at Bar-Ilan University.

Features:
• Automated signal processing
• FFT analysis
• Curve fitting
• Advanced calculations

Copyright (c) 2025 Yehonatan Smaja
License: GNU GPL v3"""

        messagebox.showinfo("About Data Analyzer Pro", about_text)

    # ========== Recent Files Functions ==========

    def get_recent_files_path(self):
        """Get path to recent files storage"""
        recent_dir = Path.home() / ".data_analyzer_pro"
        recent_dir.mkdir(parents=True, exist_ok=True)
        return recent_dir / "recent_files.json"

    def load_recent_files(self):
        """Load list of recent files"""
        recent_path = self.get_recent_files_path()
        if recent_path.exists():
            try:
                import json
                with open(recent_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_recent_files(self, files_list):
        """Save list of recent files"""
        import json
        recent_path = self.get_recent_files_path()
        with open(recent_path, 'w') as f:
            json.dump(files_list[:10], f)  # Keep only 10 most recent

    def add_to_recent_files(self, file_path):
        """Add file to recent files list"""
        recent = self.load_recent_files()
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        self.save_recent_files(recent)
        self.update_recent_files()

    def update_recent_files(self):
        """Update recent files menu"""
        self.recent_files_menu.delete(0, tk.END)
        recent = self.load_recent_files()

        for file_path in recent:
            if Path(file_path).exists():
                filename = Path(file_path).name
                self.recent_files_menu.add_command(
                    label=filename,
                    command=lambda fp=file_path: self.load_file_from_path(fp)
                )

        if not recent:
            self.recent_files_menu.add_command(label="No recent files", state="disabled")

    def load_file_from_path(self, file_path):
        """Load file from given path"""
        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".csv":
                self.data = _read_table_auto_encoding(file_path)
            elif ext == ".xlsx":
                self.data = pd.read_excel(file_path)
            elif ext == ".txt":
                with open(file_path, "rb") as f:
                    first_line = b""
                    while first_line.strip() == b"":
                        first_line = f.readline()

                if b"\t" in first_line:
                    delim = "\t"
                elif b"," in first_line:
                    delim = ","
                else:
                    delim = r"\s+"

                self.data = _read_table_auto_encoding(file_path, sep=delim, engine="python")

            self.current_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.configure(
                text=f"Loaded: {filename}",
                text_color=("green", "#2FA572"),
                font=ctk.CTkFont(size=12, weight="bold"),
            )

            self.time_column = "Time" if "Time" in self.data.columns else self.data.columns[0]
            self.channel_names = [c for c in self.data.columns if c != self.time_column]
            self.channel_conditioning = {}

            self.build_channel_checkboxes()
            self.refresh_dropdowns()
            self.update_step_status(0, True)
            self.add_to_recent_files(file_path)
            self.update_status(f"Loaded: {filename}", f"{len(self.data)} rows, {len(self.channel_names)} channels")

        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to load file: {e}")

    # ========== Project Save/Load Functions ==========

    def save_project(self):
        """Save project (settings + data)"""
        if self.data is None:
            messagebox.showwarning("No Data", "No data to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".dap",
            filetypes=[("Data Analyzer Project", "*.dap"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        try:
            import pickle
            project_data = {
                'data_file_path': self.current_file_path,
                'active_channels': {k: v.get() for k, v in self.active_channels.items()},
                'channel_names': self.channel_names,
                'time_column': self.time_column,
                'x_selector': self.x_selector.get() if hasattr(self, 'x_selector') else None,
                'primary_enabled': self.primary_enabled.get() if hasattr(self, 'primary_enabled') else False,
                'secondary_enabled': self.secondary_enabled.get() if hasattr(self, 'secondary_enabled') else False,
                'y_op': self.y_op_menu.get() if hasattr(self, 'y_op_menu') else None,
                'a_channel': self.a_menu.get() if hasattr(self, 'a_menu') else None,
                'b_channel': self.b_menu.get() if hasattr(self, 'b_menu') else None,
                'filter_type': self.filter_menu.get() if hasattr(self, 'filter_menu') else "None",
                'plot_settings': {
                    'x_units': self.x_units.get() if hasattr(self, 'x_units') else "None",
                    'y1_units': self.y1_units.get() if hasattr(self, 'y1_units') else "None",
                    'y2_units': self.y2_units.get() if hasattr(self, 'y2_units') else "None",
                }
            }

            with open(file_path, 'wb') as f:
                pickle.dump({
                    'project_data': project_data,
                    'data': self.data
                }, f)

            self.add_to_recent_files(file_path)
            self.update_status(f"Project saved: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Project saved successfully to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save project: {str(e)}")

    def load_project(self):
        """Load project"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Data Analyzer Project", "*.dap"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        try:
            import pickle
            with open(file_path, 'rb') as f:
                project = pickle.load(f)

            self.data = project['data']
            project_data = project['project_data']

            self.active_channels = {}
            self.channel_names = project_data.get('channel_names', [])
            self.time_column = project_data.get('time_column', 'Time')
            self.current_file_path = project_data.get('data_file_path')

            # Restore UI state
            self.build_channel_checkboxes()

            # Restore channel selections
            active_dict = project_data.get('active_channels', {})
            for name, var in self.active_channels.items():
                if name in active_dict:
                    var.set(active_dict[name])

            self.refresh_dropdowns()

            # Restore plot settings if available
            if 'x_selector' in project_data and project_data['x_selector']:
                self.x_selector.set(project_data['x_selector'])

            self.add_to_recent_files(file_path)
            self.update_status(f"Project loaded: {os.path.basename(file_path)}",
                             f"{len(self.data)} rows, {len(self.channel_names)} channels")
            messagebox.showinfo("Success", "Project loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project: {str(e)}")

    # ========== Export Functions ==========

    def export_plot_dialog(self):
        """Export plot with high-quality options"""
        if self.fig is None:
            messagebox.showwarning("No Plot", "No plot to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG (High Quality)", "*.png"),
                ("PDF (Vector)", "*.pdf"),
                ("SVG (Vector)", "*.svg"),
                ("EPS (Publication)", "*.eps"),
                ("TIFF (High Quality)", "*.tiff")
            ]
        )

        if not file_path:
            return

        try:
            # Set high DPI for raster formats
            dpi = 300 if file_path.endswith(('.png', '.tiff')) else None

            self.fig.savefig(
                file_path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format=Path(file_path).suffix[1:] if Path(file_path).suffix else 'png'
            )

            self.update_status(f"Plot exported to: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Plot exported successfully to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

    # ========== Data Browser Functions ==========

    def open_data_browser(self):
        """Open data browser window"""
        if self.data is None:
            messagebox.showinfo("No Data", "Please load data first.")
            return

        browser = ctk.CTkToplevel(self.window)
        browser.title("Data Browser")
        browser.geometry("900x600")

        # Create scrollable frame
        frame = ctk.CTkScrollableFrame(browser)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create text widget for data display
        text_widget = tk.Text(frame, wrap="none", font=("Courier", 9))
        text_widget.pack(fill="both", expand=True)

        # Display data (limit to first 10000 rows for performance)
        display_data = self.data.head(10000) if len(self.data) > 10000 else self.data
        text_widget.insert("1.0", display_data.to_string())
        if len(self.data) > 10000:
            text_widget.insert("end", f"\n\n... (showing first 10000 of {len(self.data)} rows)")
        text_widget.config(state="disabled")

        # Statistics button
        stats_btn = ctk.CTkButton(
            browser,
            text="Show Column Statistics",
            command=lambda: self.show_column_statistics(browser)
        )
        stats_btn.pack(pady=10)

    def show_column_statistics(self, parent):
        """Show column statistics dialog"""
        if self.data is None:
            return

        stats_window = ctk.CTkToplevel(parent)
        stats_window.title("Column Statistics")
        stats_window.geometry("500x600")

        frame = ctk.CTkScrollableFrame(stats_window)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        for col in self.data.columns:
            if self.data[col].dtype in [np.float64, np.int64, float, int]:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    stats_text = f"""
{col}:
  Count:  {len(col_data)}
  Mean:   {np.mean(col_data):.6f}
  Std:    {np.std(col_data):.6f}
  Min:    {np.min(col_data):.6f}
  Max:    {np.max(col_data):.6f}
  Median: {np.median(col_data):.6f}
  Range:  {np.max(col_data) - np.min(col_data):.6f}
"""
                    label = ctk.CTkLabel(frame, text=stats_text, anchor="w", justify="left", font=ctk.CTkFont(family="Courier", size=11))
                    label.pack(fill="x", padx=10, pady=5)

        close_btn = ctk.CTkButton(stats_window, text="Close", command=stats_window.destroy)
        close_btn.pack(pady=10)

    def create_layout(self):
        # Configure window grid
        self.window.grid_columnconfigure(0, weight=1)  # Sidebar
        self.window.grid_columnconfigure(1, weight=3)  # Controls (center)
        self.window.grid_columnconfigure(2, weight=5)  # Plot (right)
        self.window.grid_rowconfigure(0, weight=1)  # Main content row
        self.window.grid_rowconfigure(1, weight=0)  # Status bar row (fixed height)

        # ---- 1. Sidebar (left) ----
        self.sidebar_frame = ctk.CTkFrame(self.window, corner_radius=0, fg_color=("gray90", "gray16"))
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Data\nAnalyzer",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.pack(pady=(20, 15))

        # Signal Conditioning Frame
        self.signal_cond_frame = ctk.CTkFrame(self.sidebar_frame, fg_color=("gray85", "gray18"))
        self.signal_cond_frame.pack(fill="x", padx=10, pady=(0, 15))

        self.cond_label = ctk.CTkLabel(
            self.signal_cond_frame,
            text="Signal Conditioning",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.cond_label.pack(anchor="w", padx=10, pady=(5, 0))

        self.channel_cond_container = ctk.CTkScrollableFrame(
            self.signal_cond_frame,
            height=100,
            fg_color="transparent"
        )
        self.channel_cond_container.pack(fill="x", padx=5, pady=5)

        self.channel_conditioning = {}

        self.no_channels_cond_label = ctk.CTkLabel(
            self.channel_cond_container,
            text="Load data to see channels",
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color=("gray50", "gray70")
        )
        self.no_channels_cond_label.pack(pady=2)

        # Steps Tracker
        self.steps = [
            ("Load Data", False),
            ("Select Channels", False),
            ("Configure Plot", False),
            ("Analyze Results", False)
        ]
        self.step_labels = []

        for i, (label, _) in enumerate(self.steps):
            step_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
            step_frame.pack(fill="x", pady=5, padx=10)

            # Step number indicator (circle)
            step_indicator = ctk.CTkLabel(
                step_frame,
                text=f"{i + 1}",
                font=ctk.CTkFont(size=14, weight="bold"),
                width=30,
                height=30,
                corner_radius=15,
                fg_color=("gray75", "gray35"),
                text_color=("gray20", "gray90")
            )
            step_indicator.pack(side="left", padx=(5, 10))

            # Step label
            step_lbl = ctk.CTkLabel(
                step_frame,
                text=label,
                font=ctk.CTkFont(size=14),
                anchor="w"
            )
            step_lbl.pack(side="left", fill="x", expand=True)

            self.step_labels.append((step_indicator, step_lbl))

        # Bottom part of sidebar (theme + version)
        self.theme_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance:", anchor="w")
        self.theme_label.pack(pady=(30, 0), padx=20, anchor="w")

        self.theme_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["System", "Dark", "Light"],
            command=self.change_appearance_mode
        )
        self.theme_menu.pack(pady=10, padx=20)

        version_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="v1.0.0",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        version_label.pack(side="bottom", pady=10)

        # ---- 2. Center column (controls, with full scroll) ----
        self.controls_frame = ctk.CTkFrame(self.window)
        self.controls_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.controls_frame.grid_rowconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(self.controls_frame, highlightthickness=0, bg="#ededed")
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scrollbar = ctk.CTkScrollbar(self.controls_frame, command=self.controls_canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.h_scrollbar = ctk.CTkScrollbar(self.controls_frame, command=self.controls_canvas.xview,
                                            orientation="horizontal")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.controls_canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )

        # All controls go in this frame:
        self.content_frame = ctk.CTkFrame(self.controls_canvas, fg_color="transparent")
        self.canvas_window = self.controls_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        def update_scrollregion(event):
            self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

        self.content_frame.bind("<Configure>", update_scrollregion)

        def on_canvas_configure(event):
            min_width = 950
            self.controls_canvas.itemconfig(self.canvas_window, width=max(event.width, min_width))

        self.controls_canvas.bind("<Configure>", on_canvas_configure)

        # Mousewheel vertical scroll
        def on_mousewheel(event):
            self.controls_canvas.yview_scroll(-1 * int(event.delta / 120), "units")
            return "break"

        self.controls_canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Shift+mousewheel for horizontal scroll
        def on_shift_mousewheel(event):
            self.controls_canvas.xview_scroll(-1 * int(event.delta / 120), "units")
            return "break"

        self.controls_canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)
        # Bind to both the canvas and the content frame for better coverage
        self.controls_canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.controls_canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)

        # Create sections in the content frame
        self.create_file_section()
        self.create_channel_section()
        self.create_plot_section()

        # 3. Right column (plot area)
        self.plot_frame = ctk.CTkFrame(self.window)
        self.plot_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.plot_frame.grid_rowconfigure(1, weight=1)  # Make plot area expandable
        self.plot_frame.grid_columnconfigure(0, weight=1)

        # Plot title
        self.plot_title = ctk.CTkLabel(
            self.plot_frame,
            text="Results Visualization",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.plot_title.grid(row=0, column=0, pady=(15, 0), sticky="ew")

        # Plot info
        self.plot_info = ctk.CTkLabel(
            self.plot_frame,
            text="Load data and configure plot settings to visualize results",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        self.plot_info.grid(row=0, column=0, pady=(35, 15), sticky="ew")

        # Placeholder for plot - use regular frame, not scrollable
        self.plot_placeholder = ctk.CTkFrame(
            self.plot_frame,
            fg_color=("gray90", "gray20"),
            corner_radius=6,
            border_width=1,
            border_color=("gray70", "gray30")
        )
        # Use grid for plot placeholder - configure properly for matplotlib
        self.plot_placeholder.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        # Configure grid weights - row 0 for toolbar, row 1 for canvas
        self.plot_placeholder.grid_rowconfigure(1, weight=1)
        self.plot_placeholder.grid_columnconfigure(0, weight=1)

        # Placeholder text
        self.plot_placeholder_text = ctk.CTkLabel(
            self.plot_placeholder,
            text="No data visualization available\nComplete steps 1-3 to generate a plot",
            font=ctk.CTkFont(size=14)
        )
        self.plot_placeholder_text.place(relx=0.5, rely=0.5, anchor="center")

        # Bottom toolbar for plot controls
        self.plot_toolbar = ctk.CTkFrame(self.plot_frame, fg_color="transparent")
        self.plot_toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 10), padx=20)

        # Export button
        self.export_button = ctk.CTkButton(
            self.plot_toolbar,
            text="Export Results",
            command=self.save_results,
            state="disabled",
            font=ctk.CTkFont(weight="bold")
        )
        self.export_button.pack(side="right", padx=5)

        # Reset button
        self.reset_button = ctk.CTkButton(
            self.plot_toolbar,
            text="Reset",
            command=self.reset_app,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40")
        )
        self.reset_button.pack(side="right", padx=5)

    def create_file_section(self):
        # File section
        file_section = ctk.CTkFrame(self.content_frame)
        file_section.pack(fill="x", padx=10, pady=10, expand=False)

        # Section header
        section_label = ctk.CTkLabel(
            file_section,
            text="Step 1: Load Data File",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label.pack(anchor="w", padx=15, pady=(15, 5))

        section_desc = ctk.CTkLabel(
            file_section,
            text="Select a CSV or Excel file containing your measurement data",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        section_desc.pack(anchor="w", padx=15, pady=(0, 15))

        # File controls
        file_controls = ctk.CTkFrame(file_section, fg_color="transparent")
        file_controls.pack(fill="x", padx=15, pady=(0, 15))

        self.load_button = ctk.CTkButton(
            file_controls,
            text="Select File",
            command=self.load_file,
            width=120
        )
        self.load_button.pack(side="left", padx=(0, 15))

        self.file_label = ctk.CTkLabel(
            file_controls,
            text="No file loaded",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        self.file_label.pack(side="left", fill="x")

    def create_channel_section(self):
        # Channel section
        channel_section = ctk.CTkFrame(self.content_frame)
        channel_section.pack(fill="x", padx=10, pady=10, expand=False)

        # Section header
        section_label = ctk.CTkLabel(
            channel_section,
            text="Step 2: Select Channels",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label.pack(anchor="w", padx=15, pady=(15, 5))

        section_desc = ctk.CTkLabel(
            channel_section,
            text="Select the data channels you want to include in your analysis",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        section_desc.pack(anchor="w", padx=15, pady=(0, 15))

        # Channel checkboxes container
        self.checkbox_frame = ctk.CTkFrame(channel_section, fg_color="transparent")
        self.checkbox_frame.pack(fill="x", padx=15, pady=(0, 15))

        # Initial message when no file is loaded
        self.no_channels_label = ctk.CTkLabel(
            self.checkbox_frame,
            text="Load a data file to see available channels",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color=("gray50", "gray70")
        )
        self.no_channels_label.pack(pady=10)

    def create_plot_section(self):
        # --------------------------------------------------------
        # Step 3: Configure Plot (Main Section)
        # --------------------------------------------------------
        plot_section = ctk.CTkFrame(self.content_frame)
        plot_section.pack(fill="x", padx=10, pady=10, expand=False)

        section_label = ctk.CTkLabel(
            plot_section,
            text="Step 3: Configure Plot",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label.pack(anchor="w", padx=15, pady=(15, 5))

        section_desc = ctk.CTkLabel(
            plot_section,
            text="Select variables and operations for your analysis",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        section_desc.pack(anchor="w", padx=15, pady=(0, 15))

        self.graph_frame = ctk.CTkFrame(plot_section, fg_color="transparent")
        self.graph_frame.pack(fill="x", padx=15, pady=(0, 15))

        # --- X-Axis Selection ---
        x_frame = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        x_frame.pack(fill="x", pady=5)

        self.x_label = ctk.CTkLabel(x_frame, text="X-axis:", width=100, anchor="e")
        self.x_label.pack(side="left", padx=(0, 10))

        self.x_selector = ctk.CTkOptionMenu(x_frame, values=["Time"], width=200)
        self.x_selector.pack(side="left")

        # X-Units
        x_units_label = ctk.CTkLabel(x_frame, text="Units:", width=50, anchor="e")
        x_units_label.pack(side="left", padx=(15, 5))

        self.x_units = ctk.CTkOptionMenu(
            x_frame,
            values=["None", "s", "ms", "μs", "ns", "Hz", "kHz", "MHz"],
            width=70
        )
        self.x_units.set("s")
        self.x_units.pack(side="left")

        # --- X Range Controls ---
        xrange_frame = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        xrange_frame.pack(fill="x", pady=5)

        xrange_label = ctk.CTkLabel(xrange_frame, text="X Range:", width=100, anchor="e")
        xrange_label.pack(side="left", padx=(0, 10))

        self.x_min_entry = ctk.CTkEntry(xrange_frame, placeholder_text="X min", width=100)
        self.x_min_entry.pack(side="left", padx=5)

        self.x_max_entry = ctk.CTkEntry(xrange_frame, placeholder_text="X max", width=100)
        self.x_max_entry.pack(side="left", padx=5)

        self.reset_xrange_button = ctk.CTkButton(
            xrange_frame,
            text="Auto",
            width=60,
            command=self.reset_x_range
        )
        self.reset_xrange_button.pack(side="left", padx=10)

        self.apply_xrange_to_both = tk.BooleanVar(value=True)
        self.xrange_both_checkbox = ctk.CTkCheckBox(
            xrange_frame,
            text="Apply to both plots",
            variable=self.apply_xrange_to_both
        )
        self.xrange_both_checkbox.pack(side="left", padx=6)

        # --- Y Range Controls ---
        yrange_frame = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        yrange_frame.pack(fill="x", pady=5)

        yrange_label = ctk.CTkLabel(yrange_frame, text="Y Range:", width=100, anchor="e")
        yrange_label.pack(side="left", padx=(0, 10))

        self.y_min_entry = ctk.CTkEntry(yrange_frame, placeholder_text="Y min", width=100)
        self.y_min_entry.pack(side="left", padx=5)

        self.y_max_entry = ctk.CTkEntry(yrange_frame, placeholder_text="Y max", width=100)
        self.y_max_entry.pack(side="left", padx=5)

        self.reset_yrange_button = ctk.CTkButton(
            yrange_frame,
            text="Auto",
            width=60,
            command=self.reset_y_range
        )
        self.reset_yrange_button.pack(side="left", padx=10)

        # --- Center Zero Checkbox ---
        self.center_zero = tk.BooleanVar(value=False)
        self.center_zero_check = ctk.CTkCheckBox(
            yrange_frame,
            text="Center 0",
            variable=self.center_zero,
            width=80,
            text_color="#AA00FF"
        )
        self.center_zero_check.pack(side="left", padx=(10, 0))

        # --- RESTORED: Apply to Both Checkbox ---
        self.apply_yrange_to_both = tk.BooleanVar(value=True)
        self.yrange_both_checkbox = ctk.CTkCheckBox(
            yrange_frame,
            text="Apply Both",
            variable=self.apply_yrange_to_both,
            width=80
        )
        self.yrange_both_checkbox.pack(side="left", padx=(10, 0))
        # ----------------------------------------

        # --- Edit Labels Section ---
        edit_labels_frame = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        edit_labels_frame.pack(fill="x", pady=5)

        self.custom_plot_title = ""
        self.custom_x_label = ""
        self.custom_y1_label = ""
        self.custom_y2_label = ""

        self.label_status = ctk.CTkLabel(
            edit_labels_frame,
            text="No custom labels",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70"),
            anchor="e",
            width=100
        )
        self.label_status.pack(side="left", padx=(0, 10))

        edit_labels_btn = ctk.CTkButton(
            edit_labels_frame,
            text="Edit Plot Labels",
            command=self.edit_plot_labels,
            width=120,
            height=28
        )
        edit_labels_btn.pack(side="left")

        # --- Plot Type Selection ---
        plot_type_frame = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        plot_type_frame.pack(fill="x", pady=5)

        plot_type_label = ctk.CTkLabel(plot_type_frame, text="Plot Type:", width=100, anchor="e")
        plot_type_label.pack(side="left", padx=(0, 10))

        self.plot_type_menu = ctk.CTkOptionMenu(
            plot_type_frame,
            values=["Line", "Scatter", "Line+Scatter", "Bar", "Histogram", "Step"],
            width=200
        )
        self.plot_type_menu.set("Line")
        self.plot_type_menu.pack(side="left")

        # Peak detection button
        self.peak_detection_btn = ctk.CTkButton(
            plot_type_frame,
            text="Detect Peaks",
            command=self.detect_and_plot_peaks,
            width=120,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40")
        )
        self.peak_detection_btn.pack(side="left", padx=(10, 0))

        # --- Tab Selection (Primary/Secondary) ---
        tab_frame = ctk.CTkFrame(self.graph_frame)
        tab_frame.pack(fill="x", pady=10)

        self.tab_var = ctk.IntVar(value=0)

        tab_button_frame = ctk.CTkFrame(tab_frame, fg_color="transparent")
        tab_button_frame.pack(fill="x", pady=(0, 5))

        primary_tab = ctk.CTkRadioButton(
            tab_button_frame,
            text="Primary Y-Axis (Left)",
            variable=self.tab_var,
            value=0,
            command=self.switch_y_axis_tab,
            font=ctk.CTkFont(weight="bold")
        )
        primary_tab.pack(side="left", padx=(5, 15))

        secondary_tab = ctk.CTkRadioButton(
            tab_button_frame,
            text="Secondary Y-Axis (Right)",
            variable=self.tab_var,
            value=1,
            command=self.switch_y_axis_tab,
            font=ctk.CTkFont(weight="bold")
        )
        secondary_tab.pack(side="left", padx=5)

        self.tab_content = ctk.CTkFrame(tab_frame, fg_color=("gray90", "gray17"))
        self.tab_content.pack(fill="x", padx=5, pady=5)

        self.create_primary_y_controls()
        self.create_secondary_y_controls()
        self.switch_y_axis_tab()

        # --- Filter Section ---
        filter_frame = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        filter_frame.pack(fill="x", pady=5)

        self.filter_label = ctk.CTkLabel(filter_frame, text="Filter Type:", width=100, anchor="e")
        self.filter_label.pack(side="left", padx=(0, 10))

        self.filter_menu = ctk.CTkOptionMenu(
            filter_frame,
            values=["None", "Low-Pass", "High-Pass", "Band-Pass", "Band-Stop", "Moving Average"],
            command=self.update_filter_params,
            width=200
        )
        self.filter_menu.pack(side="left")

        self.filter_type_indicator = ctk.CTkLabel(
            filter_frame,
            text="[NONE]",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray50", "#2FA572"),
            width=70
        )
        self.filter_type_indicator.pack(side="left", padx=(10, 0))

        self.filter_params_container = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        self.filter_params_container.pack(fill="x", pady=5)
        self.initialize_filter_params()

        # ============================================================
        # Plot style options - REORGANIZED INTO 3 ROWS
        # ============================================================

        # --- Row 1: Primary Y-Axis Style ---
        style_row1 = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        style_row1.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(style_row1, text="Primary Axis:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 5))

        ctk.CTkLabel(style_row1, text="Color:").pack(side="left", padx=(5, 3))
        self.line_color1_menu = ctk.CTkOptionMenu(
            style_row1,
            values=["#1f77b4", "blue", "red", "green", "black", "orange", "purple", "gray"],
            width=80
        )
        self.line_color1_menu.set("blue")
        self.line_color1_menu.pack(side="left", padx=3)

        ctk.CTkLabel(style_row1, text="Marker:").pack(side="left", padx=(10, 3))
        self.marker1_menu = ctk.CTkOptionMenu(
            style_row1,
            values=["o", "o (hollow)", "s", "D", "*", ".", "+", "x", "^", "v", "None"],
            width=80
        )
        self.marker1_menu.set("o")
        self.marker1_menu.pack(side="left", padx=3)

        # --- Row 2: Secondary Y-Axis Style ---
        style_row2 = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        style_row2.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(style_row2, text="Second Axis:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 8))

        ctk.CTkLabel(style_row2, text="Color:").pack(side="left", padx=(5, 3))
        self.line_color2_menu = ctk.CTkOptionMenu(
            style_row2,
            values=["orange", "red", "green", "blue", "black", "purple", "gray"],
            width=80
        )
        self.line_color2_menu.set("orange")
        self.line_color2_menu.pack(side="left", padx=3)

        ctk.CTkLabel(style_row2, text="Marker:").pack(side="left", padx=(10, 3))
        self.marker2_menu = ctk.CTkOptionMenu(
            style_row2,
            values=["s", "o", "o (hollow)", "D", "*", ".", "+", "x", "^", "v", "None"],
            width=80
        )
        self.marker2_menu.set("None")
        self.marker2_menu.pack(side="left", padx=3)

        # --- Row 3: General Settings (Axis Color, Ticks, Legend) ---
        style_row3 = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        style_row3.pack(fill="x", pady=5)

        ctk.CTkLabel(style_row3, text="General:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 28))

        # Axis Color
        ctk.CTkLabel(style_row3, text="Axis Color:").pack(side="left", padx=(0, 3))
        self.axis_color_menu = ctk.CTkOptionMenu(
            style_row3,
            values=["black", "#333333", "#666666", "gray", "blue", "red", "green", "orange"],
            width=80
        )
        self.axis_color_menu.set("black")
        self.axis_color_menu.pack(side="left", padx=3)

        # Tick Size
        ctk.CTkLabel(style_row3, text="Tick Size:").pack(side="left", padx=(10, 3))
        self.tick_fontsize_menu = ctk.CTkOptionMenu(
            style_row3,
            values=["8", "9", "10", "11", "12", "14", "16", "18", "20", "22", "24"],
            width=60
        )
        self.tick_fontsize_menu.set("12")
        self.tick_fontsize_menu.pack(side="left", padx=3)

        # Legend Checkbox
        self.show_legend_var = tk.BooleanVar(value=True)
        self.legend_checkbox = ctk.CTkCheckBox(
            style_row3,
            text="Legend",
            width=60,
            variable=self.show_legend_var
        )
        self.legend_checkbox.pack(side="left", padx=(15, 0))

        # --- Generate Button ---
        self.plot_button = ctk.CTkButton(
            self.graph_frame,
            text="Generate Plot",
            command=self.plot_graph,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.plot_button.pack(side="bottom", anchor="w", padx=10, pady=15)

        # --------------------------------------------------------
        # Step 4: Fitting Section
        # --------------------------------------------------------
        fit_section = ctk.CTkFrame(self.content_frame)
        fit_section.pack(fill="x", padx=10, pady=10, expand=False)

        fit_label = ctk.CTkLabel(
            fit_section,
            text="Step 4: Fitting",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        fit_label.pack(anchor="w", padx=15, pady=(15, 5))

        fit_desc = ctk.CTkLabel(
            fit_section,
            text="Provide a Python expression f(x, p1, p2, ...) and initial guesses. Example: p0*np.exp(-p1*x)+p2",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        fit_desc.pack(anchor="w", padx=15, pady=(0, 10))

        fit_controls = ctk.CTkFrame(fit_section, fg_color="transparent")
        fit_controls.pack(fill="x", padx=15, pady=(0, 10))

        # Target series selector
        ctk.CTkLabel(fit_controls, text="Target Axis:", width=100, anchor="e").pack(side="left", padx=(0, 10))
        self.fit_target_menu = ctk.CTkOptionMenu(
            fit_controls,
            values=["Primary (Y1)", "Secondary (Y2)"],
            width=140
        )
        self.fit_target_menu.set("Primary (Y1)")
        self.fit_target_menu.pack(side="left")

        # Function entry
        func_frame = ctk.CTkFrame(fit_section, fg_color="transparent")
        func_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(func_frame, text="f(x, p...)=", width=100, anchor="e").pack(side="left", padx=(0, 10))
        self.fit_func_entry = ctk.CTkEntry(func_frame, width=500, placeholder_text="p0*np.exp(-p1*x)+p2")
        self.fit_func_entry.pack(side="left")

        # Initial guesses
        guess_frame = ctk.CTkFrame(fit_section, fg_color="transparent")
        guess_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(guess_frame, text="Initial guess (comma-separated):", width=220, anchor="e").pack(side="left",
                                                                                                       padx=(0, 10))
        self.fit_guess_entry = ctk.CTkEntry(guess_frame, width=300, placeholder_text="1, 1, 0")
        self.fit_guess_entry.pack(side="left")

        # Run button and results
        run_frame = ctk.CTkFrame(fit_section, fg_color="transparent")
        run_frame.pack(fill="x", padx=15, pady=5)
        self.fit_button = ctk.CTkButton(run_frame, text="Run Fit", command=self.run_fit)
        self.fit_button.pack(side="left")
        self.auto_fit_button = ctk.CTkButton(run_frame, text="Auto Fit", command=self.auto_fit)
        self.auto_fit_button.pack(side="left", padx=10)

        self.auto_fit_after_plot = tk.BooleanVar(value=False)
        self.auto_fit_checkbox = ctk.CTkCheckBox(run_frame, text="Auto Fit after Plot",
                                                 variable=self.auto_fit_after_plot)
        self.auto_fit_checkbox.pack(side="left", padx=10)

        self.fit_result_label = ctk.CTkLabel(run_frame, text="R^2: -", font=ctk.CTkFont(size=12, weight="bold"))
        self.fit_result_label.pack(side="left", padx=15)

    def reset_x_range(self):
        self.x_min_entry.delete(0, "end")
        self.x_max_entry.delete(0, "end")

    def reset_y_range(self):
        self.y_min_entry.delete(0, "end")
        self.y_max_entry.delete(0, "end")

    def edit_plot_labels(self):
        """Open a dialog to edit all plot labels at once"""
        # Create a toplevel window for the dialog
        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Edit Plot Labels")
        dialog.geometry("400x300")
        dialog.transient(self.window)  # Set to be on top of the main window
        dialog.grab_set()  # Make the dialog modal

        # Add padding
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_columnconfigure(1, weight=2)

        # Plot title
        title_label = ctk.CTkLabel(dialog, text="Plot Title:", anchor="e")
        title_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        title_entry = ctk.CTkEntry(dialog, width=250)
        title_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        title_entry.insert(0, self.custom_plot_title)

        # X-axis label
        x_label = ctk.CTkLabel(dialog, text="X-Axis Label:", anchor="e")
        x_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        x_entry = ctk.CTkEntry(dialog, width=250)
        x_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        x_entry.insert(0, self.custom_x_label)

        # Y1-axis label
        y1_label = ctk.CTkLabel(dialog, text="Y1-Axis Label:", anchor="e")
        y1_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        y1_entry = ctk.CTkEntry(dialog, width=250)
        y1_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        y1_entry.insert(0, self.custom_y1_label)

        # Y2-axis label
        y2_label = ctk.CTkLabel(dialog, text="Y2-Axis Label:", anchor="e")
        y2_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")

        y2_entry = ctk.CTkEntry(dialog, width=250)
        y2_entry.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
        y2_entry.insert(0, self.custom_y2_label)

        # Helper text
        help_text = ctk.CTkLabel(
            dialog,
            text="Leave blank to use default labels. Custom labels will\noverride automatically generated labels.",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray70")
        )
        help_text.grid(row=4, column=0, columnspan=2, padx=10, pady=(10, 20))

        # Buttons
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        def apply_labels():
            self.custom_plot_title = title_entry.get()
            self.custom_x_label = x_entry.get()
            self.custom_y1_label = y1_entry.get()
            self.custom_y2_label = y2_entry.get()

            # Update status indicator
            custom_count = sum(1 for label in [self.custom_plot_title, self.custom_x_label,
                                               self.custom_y1_label, self.custom_y2_label] if label)

            if custom_count > 0:
                self.label_status.configure(
                    text=f"{custom_count} custom labels",
                    text_color=("green", "#2FA572")
                )
            else:
                self.label_status.configure(
                    text="No custom labels",
                    text_color=("gray50", "gray70")
                )

            dialog.destroy()

        apply_btn = ctk.CTkButton(button_frame, text="Apply", command=apply_labels)
        apply_btn.pack(side="left", padx=10)

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40")
        )
        cancel_btn.pack(side="left", padx=10)

    def create_primary_y_controls(self):
        """Create controls for the primary Y-axis (left side)"""
        self.primary_frame = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        self.primary_frame.pack(fill="x", padx=10, pady=10)

        # Operation selection
        y_op_frame = ctk.CTkFrame(self.primary_frame, fg_color="transparent")
        y_op_frame.pack(fill="x", pady=5)

        self.y_op_label = ctk.CTkLabel(y_op_frame, text="Y Operation:", width=100, anchor="e")
        self.y_op_label.pack(side="left", padx=(0, 10))

        self.y_operations = [
            "Single", "A + B", "A - B", "A * B", "A / B",
            "dA/dt", "∫A dt", "d(A - B)/dt", "∫(A - B) dt",
            "A * dB/dt", "A / dB/dt", "dA/dt / B", "∫A dt / B",
            "FFT of A", "FFT of B", "FFT of (A-B)", "Custom Expression"
        ]
        self.y_op_menu = ctk.CTkOptionMenu(
            y_op_frame,
            values=self.y_operations,
            command=self.update_y_label,
            width=200
        )
        self.y_op_menu.pack(side="left")

        self.custom_expr_y1 = ""  # Store custom formula for Y1

        custom_expr_btn = ctk.CTkButton(
            y_op_frame, text="Edit Custom Formula",
            command=self.edit_custom_formula_y1,
            width=150, height=28
        )
        custom_expr_btn.pack(side="left", padx=10)

        self.y_label_display = ctk.CTkLabel(
            y_op_frame,
            text="Y = A",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color=("gray50", "gray70")
        )
        self.y_label_display.pack(side="left", padx=(15, 0))

        # Units for primary Y
        y1_units_frame = ctk.CTkFrame(self.primary_frame, fg_color="transparent")
        y1_units_frame.pack(fill="x", pady=5)

        y1_units_label = ctk.CTkLabel(y1_units_frame, text="Units:", width=100, anchor="e")
        y1_units_label.pack(side="left", padx=(0, 10))

        self.y1_units = ctk.CTkOptionMenu(
            y1_units_frame,
            values=["None", "V", "mV", "A", "kA", "MA", "Ω", "mΩ", "H", "mH", "μH", "T", "mT", "Hz", "kHz", "MHz",
                    "Vs"],
            width=90
        )
        self.y1_units.pack(side="left")

        # --- NEW: DC Correction Checkbox ---
        self.dc_correct1 = ctk.BooleanVar(value=False)
        self.dc_correct_check1 = ctk.CTkCheckBox(
            y1_units_frame,
            text="Auto DC Corr",
            variable=self.dc_correct1,
            width=100,
            text_color="#1F77B4"
        )
        self.dc_correct_check1.pack(side="left", padx=(15, 0))
        try:
            ToolTip(self.dc_correct_check1, "Subtracts median baseline (first 10%) before calculation")
        except NameError:
            pass
        # -----------------------------------

        # Enable checkbox
        self.primary_enabled = ctk.BooleanVar(value=True)
        primary_check = ctk.CTkCheckBox(
            y1_units_frame,
            text="Enable Primary Y-Axis",
            variable=self.primary_enabled,
            font=ctk.CTkFont(size=12),
            onvalue=True,
            offvalue=False
        )
        primary_check.pack(side="left", padx=(20, 0))

        # --- FFT params frame ---
        self.fft_params_frame1 = ctk.CTkFrame(self.primary_frame, fg_color=("gray95", "gray15"))

        # FFT window selection
        fft_window_frame = ctk.CTkFrame(self.fft_params_frame1, fg_color="transparent")
        fft_window_frame.pack(fill="x", pady=5, padx=10)

        fft_window_label = ctk.CTkLabel(fft_window_frame, text="FFT Window:", width=100, anchor="w")
        fft_window_label.pack(side="left", padx=(0, 10))

        self.fft_window1 = ctk.CTkOptionMenu(
            fft_window_frame,
            values=["None (Rectangular)", "Hanning", "Hamming", "Blackman", "Bartlett"],
            width=150
        )
        self.fft_window1.set("None (Rectangular)")
        self.fft_window1.pack(side="left")

        # FFT display
        fft_display_frame = ctk.CTkFrame(self.fft_params_frame1, fg_color="transparent")
        fft_display_frame.pack(fill="x", pady=5, padx=10)

        fft_display_label = ctk.CTkLabel(fft_display_frame, text="FFT Display:", width=100, anchor="w")
        fft_display_label.pack(side="left", padx=(0, 10))

        self.fft_display1 = ctk.CTkOptionMenu(
            fft_display_frame,
            values=["Magnitude", "Power (Magnitude²)", "dB (20*log10)"],
            width=150
        )
        self.fft_display1.set("Magnitude")
        self.fft_display1.pack(side="left")

        # Zero-padding
        fft_padding_frame = ctk.CTkFrame(self.fft_params_frame1, fg_color="transparent")
        fft_padding_frame.pack(fill="x", pady=5, padx=10)

        fft_padding_label = ctk.CTkLabel(fft_padding_frame, text="Zero Padding:", width=100, anchor="w")
        fft_padding_label.pack(side="left", padx=(0, 10))

        self.fft_padding1 = ctk.CTkOptionMenu(
            fft_padding_frame,
            values=["None", "2x", "4x", "8x", "16x"],
            width=150
        )
        self.fft_padding1.set("None")
        self.fft_padding1.pack(side="left")

        # Channel A selection
        a_frame = ctk.CTkFrame(self.primary_frame, fg_color="transparent")
        a_frame.pack(fill="x", pady=5)
        self.a_label = ctk.CTkLabel(a_frame, text="Channel A:", width=100, anchor="e")
        self.a_label.pack(side="left", padx=(0, 10))
        self.a_menu = ctk.CTkOptionMenu(
            a_frame,
            values=["None"],
            command=self.update_y_label,
            width=200
        )
        self.a_menu.pack(side="left")

        # Channel B selection
        b_frame = ctk.CTkFrame(self.primary_frame, fg_color="transparent")
        b_frame.pack(fill="x", pady=5)
        self.b_label = ctk.CTkLabel(b_frame, text="Channel B:", width=100, anchor="e")
        self.b_label.pack(side="left", padx=(0, 10))
        self.b_menu = ctk.CTkOptionMenu(
            b_frame,
            values=["None"],
            command=self.update_y_label,
            width=200
        )
        self.b_menu.pack(side="left")

    def edit_custom_formula_y1(self):
        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Custom Y Expression (Primary)")
        dialog.geometry("400x200")
        dialog.transient(self.window)
        dialog.grab_set()

        label = ctk.CTkLabel(dialog, text="Enter formula for Y1 using A, B, constants:\nExample: A - B*0.001")
        label.pack(pady=10)
        entry = ctk.CTkEntry(dialog, width=350)
        entry.insert(0, self.custom_expr_y1)
        entry.pack(pady=10)

        def apply_expr():
            self.custom_expr_y1 = entry.get()
            self.y_op_menu.set("Custom Expression")
            dialog.destroy()
            self.plot_graph()

        ok_btn = ctk.CTkButton(dialog, text="Apply", command=apply_expr)
        ok_btn.pack(pady=10)

    def create_secondary_y_controls(self):
        """Create controls for the secondary Y-axis (right side)"""
        self.secondary_frame = ctk.CTkFrame(self.tab_content, fg_color="transparent")

        # Y2 Operation selection
        y2_op_frame = ctk.CTkFrame(self.secondary_frame, fg_color="transparent")
        y2_op_frame.pack(fill="x", pady=5)

        y2_op_label = ctk.CTkLabel(y2_op_frame, text="Y2 Operation:", width=100, anchor="e")
        y2_op_label.pack(side="left", padx=(0, 10))

        self.y2_operations = [
            "Single", "C + D", "C - D", "C * D", "C / D",
            "dC/dt", "∫C dt", "d(C - D)/dt", "∫(C - D) dt",
            "C * dD/dt", "C / dD/dt", "dC/dt / D", "∫C dt / D",
            "FFT of C", "FFT of D", "FFT of (C-D)", "Custom Expression"
        ]
        self.y2_op_menu = ctk.CTkOptionMenu(
            y2_op_frame,
            values=self.y2_operations,
            command=self.update_y2_label,
            width=200
        )
        self.y2_op_menu.pack(side="left")

        self.custom_expr_y2 = ""
        custom_expr_btn = ctk.CTkButton(y2_op_frame, text="Edit Custom Formula", command=self.edit_custom_formula_y2,
                                        width=150, height=28)
        custom_expr_btn.pack(side="left", padx=10)

        self.y2_label_display = ctk.CTkLabel(y2_op_frame, text="Y2 = C", font=ctk.CTkFont(size=12, slant="italic"),
                                             text_color=("gray50", "gray70"))
        self.y2_label_display.pack(side="left", padx=(15, 0))

        # Units for secondary Y
        y2_units_frame = ctk.CTkFrame(self.secondary_frame, fg_color="transparent")
        y2_units_frame.pack(fill="x", pady=5)

        y2_units_label = ctk.CTkLabel(y2_units_frame, text="Units:", width=100, anchor="e")
        y2_units_label.pack(side="left", padx=(0, 10))

        self.y2_units = ctk.CTkOptionMenu(
            y2_units_frame,
            values=["None", "V", "mV", "A", "kA", "MA", "Ω", "mΩ", "H", "mH", "μH", "T", "mT", "Hz", "kHz", "MHz",
                    "Vs"],
            width=90
        )
        self.y2_units.pack(side="left")

        # --- NEW: DC Correction Checkbox ---
        self.dc_correct2 = ctk.BooleanVar(value=False)
        self.dc_correct_check2 = ctk.CTkCheckBox(
            y2_units_frame,
            text="Auto DC Corr",
            variable=self.dc_correct2,
            width=100,
            text_color="orange"
        )
        self.dc_correct_check2.pack(side="left", padx=(15, 0))
        try:
            ToolTip(self.dc_correct_check2, "Subtracts median baseline (first 10%) before calculation")
        except NameError:
            pass
        # -----------------------------------

        # Enable checkbox
        self.secondary_enabled = ctk.BooleanVar(value=False)
        secondary_check = ctk.CTkCheckBox(y2_units_frame, text="Enable Secondary Y-Axis",
                                          variable=self.secondary_enabled, font=ctk.CTkFont(size=12), onvalue=True,
                                          offvalue=False)
        secondary_check.pack(side="left", padx=(20, 0))

        # FFT params for secondary axis
        self.fft_params_frame2 = ctk.CTkFrame(self.secondary_frame, fg_color=("gray95", "gray15"))

        fft_window_frame = ctk.CTkFrame(self.fft_params_frame2, fg_color="transparent")
        fft_window_frame.pack(fill="x", pady=5, padx=10)
        fft_window_label = ctk.CTkLabel(fft_window_frame, text="FFT Window:", width=100, anchor="w")
        fft_window_label.pack(side="left", padx=(0, 10))
        self.fft_window2 = ctk.CTkOptionMenu(fft_window_frame,
                                             values=["None (Rectangular)", "Hanning", "Hamming", "Blackman",
                                                     "Bartlett"], width=150)
        self.fft_window2.set("None (Rectangular)")
        self.fft_window2.pack(side="left")

        fft_display_frame = ctk.CTkFrame(self.fft_params_frame2, fg_color="transparent")
        fft_display_frame.pack(fill="x", pady=5, padx=10)
        fft_display_label = ctk.CTkLabel(fft_display_frame, text="FFT Display:", width=100, anchor="w")
        fft_display_label.pack(side="left", padx=(0, 10))
        self.fft_display2 = ctk.CTkOptionMenu(fft_display_frame,
                                              values=["Magnitude", "Power (Magnitude²)", "dB (20*log10)"], width=150)
        self.fft_display2.set("Magnitude")
        self.fft_display2.pack(side="left")

        fft_padding_frame = ctk.CTkFrame(self.fft_params_frame2, fg_color="transparent")
        fft_padding_frame.pack(fill="x", pady=5, padx=10)
        fft_padding_label = ctk.CTkLabel(fft_padding_frame, text="Zero Padding:", width=100, anchor="w")
        fft_padding_label.pack(side="left", padx=(0, 10))
        self.fft_padding2 = ctk.CTkOptionMenu(fft_padding_frame, values=["None", "2x", "4x", "8x", "16x"], width=150)
        self.fft_padding2.set("None")
        self.fft_padding2.pack(side="left")

        # Channel C selection
        c_frame = ctk.CTkFrame(self.secondary_frame, fg_color="transparent")
        c_frame.pack(fill="x", pady=5)
        c_label = ctk.CTkLabel(c_frame, text="Channel C:", width=100, anchor="e")
        c_label.pack(side="left", padx=(0, 10))
        self.c_menu = ctk.CTkOptionMenu(c_frame, values=["None"], command=self.update_y2_label, width=200)
        self.c_menu.pack(side="left")

        # Channel D selection
        d_frame = ctk.CTkFrame(self.secondary_frame, fg_color="transparent")
        d_frame.pack(fill="x", pady=5)
        d_label = ctk.CTkLabel(d_frame, text="Channel D:", width=100, anchor="e")
        d_label.pack(side="left", padx=(0, 10))
        self.d_menu = ctk.CTkOptionMenu(d_frame, values=["None"], command=self.update_y2_label, width=200)
        self.d_menu.pack(side="left")

    def edit_custom_formula_y2(self):
        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Custom Y2 Expression (Secondary)")
        dialog.geometry("400x200")
        dialog.transient(self.window)
        dialog.grab_set()

        label = ctk.CTkLabel(dialog, text="Enter formula for Y2 using C, D, constants:\nExample: C - D*0.001")
        label.pack(pady=10)
        entry = ctk.CTkEntry(dialog, width=350)
        entry.insert(0, self.custom_expr_y2)
        entry.pack(pady=10)

        def apply_expr():
            self.custom_expr_y2 = entry.get()
            self.y2_op_menu.set("Custom Expression")
            dialog.destroy()
            self.plot_graph()

        ok_btn = ctk.CTkButton(dialog, text="Apply", command=apply_expr)
        ok_btn.pack(pady=10)

    def switch_y_axis_tab(self, *args):
        """Switch between primary and secondary y-axis tabs"""
        tab_index = self.tab_var.get()

        # Hide all frames first
        for widget in self.tab_content.winfo_children():
            widget.pack_forget()

        # Show the selected tab
        if tab_index == 0:
            self.primary_frame.pack(fill="x", padx=10, pady=10)
        else:
            self.secondary_frame.pack(fill="x", padx=10, pady=10)

    def initialize_filter_params(self):
        """Initialize the filter parameters section with empty content"""
        # Clear any existing content
        for widget in self.filter_params_container.winfo_children():
            widget.destroy()

        # Create a label indicating no parameters needed
        no_params_label = ctk.CTkLabel(
            self.filter_params_container,
            text="No filter parameters needed",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color=("gray50", "gray70")
        )
        no_params_label.pack(pady=10, padx=(100, 0), anchor="w")

        # Initialize filter variables dictionary
        self.filter_frame_vars = {}

    def update_filter_params(self, filter_type=None):
        """Update filter parameters based on selected filter type"""
        if filter_type is None:
            filter_type = self.filter_menu.get()

        # Update the filter type indicator
        if filter_type == "None":
            self.filter_type_indicator.configure(text="[NONE]")
        elif filter_type == "Low-Pass":
            self.filter_type_indicator.configure(text="[LP]")
        elif filter_type == "High-Pass":
            self.filter_type_indicator.configure(text="[HP]")
        elif filter_type == "Band-Pass":
            self.filter_type_indicator.configure(text="[BP]")
        elif filter_type == "Band-Stop":
            self.filter_type_indicator.configure(text="[BS]")
        elif filter_type == "Moving Average":
            self.filter_type_indicator.configure(text="[MA]")

        # Clear existing filter parameters
        for widget in self.filter_params_container.winfo_children():
            widget.destroy()

        # Reset filter variables
        self.filter_frame_vars = {}

        # Create filter parameters based on the selected filter type
        if filter_type == "None":
            # No parameters needed
            no_params_label = ctk.CTkLabel(
                self.filter_params_container,
                text="No filter parameters needed",
                font=ctk.CTkFont(size=12, slant="italic"),
                text_color=("gray50", "gray70")
            )
            no_params_label.pack(pady=10, padx=(100, 0), anchor="w")

        elif filter_type == "Moving Average":
            # Moving average needs window size
            param_frame = ctk.CTkFrame(self.filter_params_container, fg_color="transparent")
            param_frame.pack(fill="x", pady=5)

            param_label = ctk.CTkLabel(param_frame, text="Window Size:", width=100, anchor="e")
            param_label.pack(side="left", padx=(0, 10))

            window_entry = ctk.CTkEntry(
                param_frame,
                placeholder_text="Enter window size (e.g., 15)",
                width=200
            )
            window_entry.pack(side="left")

            # Store entry field in variables dictionary
            self.filter_frame_vars["window_size"] = window_entry

            # Add hint
            hint_label = ctk.CTkLabel(
                self.filter_params_container,
                text="Larger window size = smoother curve, but may lose important features",
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color=("gray40", "gray60")
            )
            hint_label.pack(anchor="w", padx=(100, 0), pady=(0, 5))

        else:  # Frequency-based filters (LP, HP, BP, BS)
            # Create filter order field
            order_frame = ctk.CTkFrame(self.filter_params_container, fg_color="transparent")
            order_frame.pack(fill="x", pady=5)

            order_label = ctk.CTkLabel(order_frame, text="Filter Order:", width=100, anchor="e")
            order_label.pack(side="left", padx=(0, 10))

            order_entry = ctk.CTkEntry(
                order_frame,
                placeholder_text="Order (2-10, default: 4)",
                width=200
            )
            order_entry.insert(0, "4")  # Default order
            order_entry.pack(side="left")

            # Store entry field in variables dictionary
            self.filter_frame_vars["order"] = order_entry

            # Create filter type selection
            filter_design_frame = ctk.CTkFrame(self.filter_params_container, fg_color="transparent")
            filter_design_frame.pack(fill="x", pady=5)

            filter_design_label = ctk.CTkLabel(filter_design_frame, text="Filter Design:", width=100, anchor="e")
            filter_design_label.pack(side="left", padx=(0, 10))

            filter_design_menu = ctk.CTkOptionMenu(
                filter_design_frame,
                values=["Butterworth", "Chebyshev"],
                width=200
            )
            filter_design_menu.pack(side="left")

            # Store option menu in variables dictionary
            self.filter_frame_vars["design"] = filter_design_menu

            # Create cutoff frequency fields
            if filter_type in ["Low-Pass", "High-Pass"]:
                # Single cutoff frequency for LP and HP
                cutoff_frame = ctk.CTkFrame(self.filter_params_container, fg_color="transparent")
                cutoff_frame.pack(fill="x", pady=5)

                cutoff_label = ctk.CTkLabel(cutoff_frame, text="Cutoff Freq (Hz):", width=100, anchor="e")
                cutoff_label.pack(side="left", padx=(0, 10))

                cutoff_entry = ctk.CTkEntry(
                    cutoff_frame,
                    placeholder_text="Cutoff frequency in Hertz",
                    width=200
                )
                cutoff_entry.pack(side="left")

                # --- NEW: Cutoff suggestion label ---
                self.cutoff_suggestion_label = ctk.CTkLabel(
                    cutoff_frame,
                    text="",
                    font=("Arial", 10, "italic"),
                    text_color="gray"
                )
                self.cutoff_suggestion_label.pack(side="left", padx=(10, 0))

                # Store entry field in variables dictionary
                self.filter_frame_vars["cutoff"] = cutoff_entry

            elif filter_type in ["Band-Pass", "Band-Stop"]:
                # Two cutoff frequencies for BP and BS
                lowcut_frame = ctk.CTkFrame(self.filter_params_container, fg_color="transparent")
                lowcut_frame.pack(fill="x", pady=5)

                lowcut_label = ctk.CTkLabel(lowcut_frame, text="Low Cutoff (Hz):", width=100, anchor="e")
                lowcut_label.pack(side="left", padx=(0, 10))

                lowcut_entry = ctk.CTkEntry(
                    lowcut_frame,
                    placeholder_text="Lower cutoff frequency in Hertz",
                    width=200
                )
                lowcut_entry.pack(side="left")

                # Store entry field in variables dictionary
                self.filter_frame_vars["lowcut"] = lowcut_entry

                highcut_frame = ctk.CTkFrame(self.filter_params_container, fg_color="transparent")
                highcut_frame.pack(fill="x", pady=5)

                highcut_label = ctk.CTkLabel(highcut_frame, text="High Cutoff (Hz):", width=100, anchor="e")
                highcut_label.pack(side="left", padx=(0, 10))

                highcut_entry = ctk.CTkEntry(
                    highcut_frame,
                    placeholder_text="Higher cutoff frequency in Hertz",
                    width=200
                )
                highcut_entry.pack(side="left")

                # Store entry field in variables dictionary
                self.filter_frame_vars["highcut"] = highcut_entry

            # Add hint for frequency-based filters
            if filter_type != "None":
                hint_label = ctk.CTkLabel(
                    self.filter_params_container,
                    text="Frequencies should be in Hertz (Hz). Sampling rate will be calculated from time data.",
                    font=ctk.CTkFont(size=11, slant="italic"),
                    text_color=("gray40", "gray60")
                )
                hint_label.pack(anchor="w", padx=(100, 0), pady=(0, 5))

    def change_appearance_mode(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode.lower())

    def update_step_status(self, index, success):
        indicator, label = self.step_labels[index]

        if success:
            indicator.configure(
                fg_color=("green", "#2FA572"),
                text="✓"
            )
            label.configure(font=ctk.CTkFont(size=14, weight="bold"))
        else:
            indicator.configure(
                fg_color=("gray75", "gray35"),
                text=f"{index + 1}"
            )
            label.configure(font=ctk.CTkFont(size=14, weight="normal"))

    def ask_skip_header_rows(self):
        """Ask user if they want to skip header rows"""
        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Skip Header Rows")
        dialog.geometry("400x200")
        dialog.transient(self.window)
        dialog.grab_set()

        label = ctk.CTkLabel(
            dialog,
            text="Does your file have header rows to skip?\nEnter number of rows to skip (0 for none):"
        )
        label.pack(pady=20)

        entry = ctk.CTkEntry(dialog, width=100)
        entry.insert(0, "0")
        entry.pack(pady=10)
        entry.focus()

        result = [0]  # Use list to modify from nested function

        def ok_clicked():
            try:
                result[0] = int(entry.get())
                if result[0] < 0:
                    result[0] = 0
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number")

        def cancel_clicked():
            result[0] = None
            dialog.destroy()

        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=20)

        ok_btn = ctk.CTkButton(button_frame, text="OK", command=ok_clicked, width=100)
        ok_btn.pack(side="left", padx=10)

        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=cancel_clicked, width=100)
        cancel_btn.pack(side="left", padx=10)

        # Bind Enter key
        entry.bind("<Return>", lambda e: ok_clicked())

        dialog.wait_window()
        return result[0]

    def load_file(self):
        # ── 1 File‑open dialog ─────────────────────────────────────────────
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx"),
                ("Text", "*.txt"),
            ]
        )
        if not file_path:  # user cancelled
            return

        # Ask user if they want to skip header rows
        skip_rows = self.ask_skip_header_rows()
        if skip_rows is None:  # User cancelled
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()

            # ── 2 Load according to extension ──────────────────────────────
            if ext == ".csv":
                # robust encoding auto‑detection
                self.data = _read_table_auto_encoding(file_path, skiprows=skip_rows)

            elif ext == ".xlsx":
                self.data = pd.read_excel(file_path, skiprows=skip_rows)  # Excel is binary → no encoding issue

            elif ext == ".txt":
                # quick delimiter sniff (first non‑empty text line)
                with open(file_path, "rb") as f:
                    first_line = b""
                    while first_line.strip() == b"":
                        first_line = f.readline()

                if b"\t" in first_line:
                    delim = "\t"
                elif b"," in first_line:
                    delim = ","
                else:
                    delim = r"\s+"  # any whitespace

                # read with auto‑encoding + chosen delimiter
                self.data = _read_table_auto_encoding(
                    file_path, sep=delim, engine="python", skiprows=skip_rows
                )

                # convert to CSV so the rest of the app sees a uniform format
                csv_path = os.path.splitext(file_path)[0] + ".csv"
                self.data.to_csv(csv_path, index=False)
                file_path = csv_path
                messagebox.showinfo(
                    "TXT imported",
                    f"TXT loaded with delimiter '{delim}'.\n"
                    f"Converted copy saved as:\n{csv_path}",
                )
            else:
                raise ValueError("Unsupported file type")

            # ── 3 House‑keeping after successful load ──────────────────────
            self.current_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.configure(
                text=f"Loaded: {filename}",
                text_color=("green", "#2FA572"),
                font=ctk.CTkFont(size=12, weight="bold"),
            )

            # time column detection
            self.time_column = "Time" if "Time" in self.data.columns else self.data.columns[0]

            # channel bookkeeping
            self.channel_names = [c for c in self.data.columns if c != self.time_column]
            self.channel_conditioning = {}

            self.build_channel_checkboxes()
            self.refresh_dropdowns()
            self.update_step_status(0, True)

            # Add to recent files and update status
            self.add_to_recent_files(file_path)
            self.update_status(f"Loaded: {filename}", f"{len(self.data)} rows, {len(self.channel_names)} channels")

            messagebox.showinfo(
                "File Loaded",
                f"Successfully loaded {filename} with {len(self.channel_names)} data channels"
            )

        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to load file: {e}")
            self.update_step_status(0, False)

    def build_channel_checkboxes(self):
        # Clear existing widgets
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()

        # Clear existing active channels
        self.active_channels = {}

        if not self.channel_names:
            self.no_channels_label = ctk.CTkLabel(
                self.checkbox_frame,
                text="No channels found in the data file",
                font=ctk.CTkFont(size=12, slant="italic"),
                text_color=("gray50", "gray70")
            )
            self.no_channels_label.pack(pady=10)
            return

        # Create a select all checkbox
        select_all_var = ctk.BooleanVar(value=True)
        select_all_chk = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Select All",
            variable=select_all_var,
            command=lambda: self.toggle_all_channels(select_all_var.get()),
            font=ctk.CTkFont(weight="bold")
        )
        select_all_chk.pack(anchor="w", pady=(0, 10))

        # Create a grid layout for checkboxes
        channels_grid = ctk.CTkFrame(self.checkbox_frame, fg_color="transparent")
        channels_grid.pack(fill="x")

        # Calculate number of columns based on number of channels
        num_columns = min(3, len(self.channel_names))

        # Create checkboxes in a grid layout
        for idx, name in enumerate(self.channel_names):
            var = ctk.BooleanVar(value=True)

            # Create a frame for each checkbox for better alignment
            chk_frame = ctk.CTkFrame(channels_grid, fg_color="transparent")
            row = idx // num_columns
            col = idx % num_columns
            chk_frame.grid(row=row, column=col, sticky="w", padx=5, pady=5)

            chk = ctk.CTkCheckBox(
                chk_frame,
                text=name,
                variable=var,
                command=self.refresh_dropdowns
            )
            chk.pack(anchor="w")

            self.active_channels[name] = var

        # Update the signal conditioning options in the sidebar
        self.setup_signal_conditioning()

        self.update_step_status(1, True)

    def toggle_all_channels(self, select_all):
        for name, var in self.active_channels.items():
            var.set(select_all)
        self.refresh_dropdowns()

    def setup_signal_conditioning(self):
        """Setup signal conditioning options for active channels"""
        # Clear existing widgets
        for widget in self.channel_cond_container.winfo_children():
            widget.destroy()

        # Get active channels
        active_channels = [name for name, var in self.active_channels.items() if var.get()]

        if not active_channels:
            # No active channels, show message
            self.no_channels_cond_label = ctk.CTkLabel(
                self.channel_cond_container,
                text="No active channels",
                font=ctk.CTkFont(size=10, slant="italic"),
                text_color=("gray50", "gray70")
            )
            self.no_channels_cond_label.pack(pady=2)
            return

        # Create compact UI for each active channel
        for channel in active_channels:
            channel_frame = ctk.CTkFrame(self.channel_cond_container, fg_color="transparent", height=20)
            channel_frame.pack(fill="x", pady=1)

            # Get display name - use custom name if available
            display_name = channel
            if hasattr(self, 'channel_custom_names') and channel in self.channel_custom_names:
                display_name = self.channel_custom_names[channel]

            # Abbreviate channel name if too long
            if len(display_name) > 12:
                display_name = display_name[:10] + ".."

            # Channel name
            channel_label = ctk.CTkLabel(
                channel_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
                width=75,
                anchor="w"
            )
            channel_label.pack(side="left", padx=(2, 5))

            # Conditioning type dropdown
            cond_var = ctk.StringVar(value="None")
            if channel in self.channel_conditioning:
                cond_var = self.channel_conditioning[channel]
            else:
                self.channel_conditioning[channel] = cond_var

            cond_menu = ctk.CTkOptionMenu(
                channel_frame,
                values=["None", "LeCroy (x20)", "LeCroy (x200)", "Rogowski (x5000)", "Shant (x1000)", "X50", "X500", "X10"],
                variable=cond_var,
                width=120,
                height=20,
                font=ctk.CTkFont(size=10),
                dropdown_font=ctk.CTkFont(size=10),
                anchor="center"
            )
            cond_menu.pack(side="right", padx=2)

    def refresh_dropdowns(self):
        active = [name for name, var in self.active_channels.items() if var.get()]

        # Update X-axis dropdown
        self.x_selector.configure(values=[self.time_column] + active)
        self.x_selector.set(self.time_column)

        # Update Channel A and B dropdowns for primary Y-axis
        if active:
            self.a_menu.configure(values=active)
            self.a_menu.set(active[0])

            self.b_menu.configure(values=active)
            self.b_menu.set(active[0] if len(active) == 1 else active[1] if len(active) > 1 else active[0])

            # Update Channel C and D dropdowns for secondary Y-axis
            self.c_menu.configure(values=active)
            self.c_menu.set(active[0])

            self.d_menu.configure(values=active)
            self.d_menu.set(active[0] if len(active) == 1 else active[1] if len(active) > 1 else active[0])
        else:
            self.a_menu.configure(values=["None"])
            self.a_menu.set("None")

            self.b_menu.configure(values=["None"])
            self.b_menu.set("None")

            self.c_menu.configure(values=["None"])
            self.c_menu.set("None")

            self.d_menu.configure(values=["None"])
            self.d_menu.set("None")

        self.update_y_label()
        self.update_y2_label()
        self.update_filter_params()  # Initialize the filter indicator

        # Update signal conditioning display (some channels may have been deselected)
        self.setup_signal_conditioning()

    def update_y_label(self, *args):
        op = self.y_op_menu.get()
        a = self.a_menu.get()
        b = self.b_menu.get()

        if op == "Custom Expression...":
            label = f"Y = {self.custom_expr if self.custom_expr else '[expression]'}"

        # Check if this is an FFT operation
        is_fft = op.startswith("FFT")
        if is_fft:
            # Show FFT parameters
            self.fft_params_frame1.pack(fill="x", pady=5, after=self.y_label_display.winfo_parent())
        else:
            # Hide FFT parameters
            self.fft_params_frame1.pack_forget()

        if op == "Single":
            label = f"Y = {a}"
        elif op in ["A + B", "A - B", "A * B", "A / B"]:
            label = f"Y = {op.replace('A', a).replace('B', b)}"
        elif op in ["dA/dt", "∫A dt"]:
            label = f"Y = {op.replace('A', a)}"
        elif op in ["d(A - B)/dt", "∫(A - B) dt"]:
            label = f"Y = {op.replace('A', a).replace('B', b)}"
        elif op in ["A * dB/dt", "A / dB/dt"]:
            label = f"Y = {op.replace('A', a).replace('B', b)}"
        elif op in ["dA/dt / B", "∫A dt / B"]:
            label = f"Y = {op.replace('A', a).replace('B', b)}"
        elif op == "FFT of A":
            label = f"Y = FFT({a})"
        elif op == "FFT of B":
            label = f"Y = FFT({b})"
        elif op == "FFT of (A-B)":
            label = f"Y = FFT({a}-{b})"
        else:
            label = f"Y = {op}"

        self.y_label_display.configure(text=label)

    def update_y2_label(self, *args):
        op = self.y2_op_menu.get()
        c = self.c_menu.get()
        d = self.d_menu.get()

        # Check if this is an FFT operation
        is_fft = op.startswith("FFT")
        if is_fft:
            # Show FFT parameters
            self.fft_params_frame2.pack(fill="x", pady=5, after=self.y2_label_display.winfo_parent())
        else:
            # Hide FFT parameters
            self.fft_params_frame2.pack_forget()

        if op == "Single":
            label = f"Y2 = {c}"
        elif op in ["C + D", "C - D", "C * D", "C / D"]:
            label = f"Y2 = {op.replace('C', c).replace('D', d)}"
        elif op in ["dC/dt", "∫C dt"]:
            label = f"Y2 = {op.replace('C', c)}"
        elif op in ["d(C - D)/dt", "∫(C - D) dt"]:
            label = f"Y2 = {op.replace('C', c).replace('D', d)}"
        elif op in ["C * dD/dt", "C / dD/dt"]:
            label = f"Y2 = {op.replace('C', c).replace('D', d)}"
        elif op in ["dC/dt / D", "∫C dt / D"]:
            label = f"Y2 = {op.replace('C', c).replace('D', d)}"
        elif op == "FFT of C":
            label = f"Y2 = FFT({c})"
        elif op == "FFT of D":
            label = f"Y2 = FFT({d})"
        elif op == "FFT of (C-D)":
            label = f"Y2 = FFT({c}-{d})"
        else:
            label = f"Y2 = {op}"

        self.y2_label_display.configure(text=label)

    def get_series(self, name):
        """Get a data series with applied signal conditioning if any"""
        if name in self.data.columns:
            series = self.data[name].values

            # Apply any signal conditioning multiplier if configured
            if name in self.channel_conditioning:
                cond_type = self.channel_conditioning[name].get()
                if cond_type == "LeCroy (x20)":
                    series = series * 20.0
                elif cond_type == "LeCroy (x200)":
                    series = series * 200.0
                elif cond_type == "Rogowski (x5000)":
                    series = series * 5000.0
                elif cond_type == "Shant (x1000)":
                    series = series * 1000.0
                elif cond_type == "X50":
                    series = series * 50.0
                elif cond_type == "X500":
                    series = series * 500.0
                elif cond_type == "X10":
                    series = series * 10.0

            return series
        return None

    def calculate_sampling_rate(self, time_data):
        """Calculate sampling rate from time data"""
        # Calculate the average time step
        time_steps = np.diff(time_data)
        avg_time_step = np.mean(time_steps)
        # Calculate sampling rate (samples per second)
        fs = 1.0 / avg_time_step
        return fs

    def apply_filter(self, data, filter_type, time_data, filter_params):
        """Apply selected filter to the data"""
        if filter_type == "None" or not filter_params:
            return data

        # Calculate sampling rate from time data
        fs = self.calculate_sampling_rate(time_data)

        if filter_type == "Moving Average":
            # Get window size
            try:
                window_size = int(filter_params.get("window_size", "15"))
                if window_size < 2:
                    raise ValueError("Window size must be at least 2")
                return self.moving_average(data, window_size)
            except (ValueError, TypeError) as e:
                messagebox.showerror("Filter Error", f"Invalid window size: {str(e)}")
                return data

        # Get filter design and order for frequency-based filters
        filter_design = filter_params.get("design", "Butterworth")

        try:
            filter_order = int(filter_params.get("order", "4"))
            if filter_order < 1 or filter_order > 20:
                raise ValueError("Filter order must be between 1 and 20")
        except (ValueError, TypeError) as e:
            messagebox.showerror("Filter Error", f"Invalid filter order: {str(e)}")
            return data

        # Apply the appropriate filter based on type
        try:
            if filter_type == "Low-Pass":
                cutoff = float(filter_params.get("cutoff", "0"))
                if cutoff <= 0:
                    raise ValueError("Cutoff frequency must be greater than 0")
                return self.apply_lowpass(data, cutoff, fs, filter_order, filter_design)

            elif filter_type == "High-Pass":
                cutoff = float(filter_params.get("cutoff", "0"))
                if cutoff <= 0:
                    raise ValueError("Cutoff frequency must be greater than 0")
                return self.apply_highpass(data, cutoff, fs, filter_order, filter_design)

            elif filter_type in ["Band-Pass", "Band-Stop"]:
                lowcut = float(filter_params.get("lowcut", "0"))
                highcut = float(filter_params.get("highcut", "0"))

                if lowcut <= 0 or highcut <= 0:
                    raise ValueError("Cutoff frequencies must be greater than 0")
                if lowcut >= highcut:
                    raise ValueError("High cutoff must be greater than low cutoff")

                if filter_type == "Band-Pass":
                    return self.apply_bandpass(data, lowcut, highcut, fs, filter_order, filter_design)
                else:  # Band-Stop
                    return self.apply_bandstop(data, lowcut, highcut, fs, filter_order, filter_design)
        except (ValueError, TypeError) as e:
            messagebox.showerror("Filter Error", f"Invalid filter parameters: {str(e)}")
            return data

        return data

    def plot_graph(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded. Please load a data file first.")
            return

        # Show progress for long operations
        self.update_status("Generating plot...")
        self.show_progress(0.1)

        x_name = self.x_selector.get()

        # ------------- gather GUI selections --------------------
        primary_enabled = self.primary_enabled.get()
        secondary_enabled = self.secondary_enabled.get()

        y1_op = self.y_op_menu.get() if primary_enabled else None
        a_name = self.a_menu.get() if primary_enabled else None
        b_name = self.b_menu.get() if primary_enabled else None

        y2_op = self.y2_op_menu.get() if secondary_enabled else None
        c_name = self.c_menu.get() if secondary_enabled else None
        d_name = self.d_menu.get() if secondary_enabled else None

        # Gather DC Correction settings
        do_dc1 = self.dc_correct1.get() if hasattr(self, 'dc_correct1') else False
        do_dc2 = self.dc_correct2.get() if hasattr(self, 'dc_correct2') else False

        # Check range options
        center_zero_active = self.center_zero.get() if hasattr(self, 'center_zero') else False
        apply_both_active = self.apply_yrange_to_both.get() if hasattr(self, 'apply_yrange_to_both') else False

        # ------------- basic validation -------------------------
        if not primary_enabled and not secondary_enabled:
            messagebox.showerror("Error", "At least one Y‑axis must be enabled.")
            self.hide_progress()
            return

        try:
            self.show_progress(0.2)
            x = self.get_series(x_name)

            # ----------- PRIMARY Y -------------------------
            y1 = y1_label = None
            if primary_enabled:
                if y1_op == "Custom Expression":
                    expr = self.custom_expr_y1
                    if not expr.strip():
                        messagebox.showerror("Error", "Custom Y1 expression is empty.")
                        return
                    y1 = self.evaluate_math_expression(expr)
                    y1_label = f"Y1: {expr}"
                else:
                    a = self.get_series(a_name)
                    b = self.get_series(b_name) if b_name != "None" else None
                    # PASS THE DC CORRECTION FLAG
                    y1 = self.compute_y(y1_op, a, b, x, use_dc_correction=do_dc1)
                    y1_label = self.y_label_display.cget("text")
                    if do_dc1:
                        y1_label += " (DC Corr)"

            # ----------- SECONDARY Y -------------------------
            y2 = y2_label = None
            if secondary_enabled:
                if y2_op == "Custom Expression":
                    expr2 = self.custom_expr_y2
                    if not expr2.strip():
                        messagebox.showerror("Error", "Custom Y2 expression is empty.")
                        return
                    y2 = self.evaluate_math_expression(expr2)
                    y2_label = f"Y2: {expr2}"
                else:
                    c = self.get_series(c_name)
                    d = self.get_series(d_name) if d_name != "None" else None
                    # PASS THE DC CORRECTION FLAG
                    y2 = self.compute_y(y2_op, c, d, x, use_dc_correction=do_dc2)
                    y2_label = self.y2_label_display.cget("text")
                    if do_dc2:
                        y2_label += " (DC Corr)"

            # --------- FFT / X axis / Range logic -------------
            if primary_enabled and "FFT" in (y1_op or ""):
                x = self.fft_freqs
                x_name = "Frequency (Hz)"
            elif secondary_enabled and "FFT" in (y2_op or ""):
                x = self.fft_freqs
                x_name = "Frequency (Hz)"

            # --------- X Range masking -------------
            x_min = self.x_min_entry.get()
            x_max = self.x_max_entry.get()
            x_min_val = float(x_min) if x_min.strip() else np.min(x)
            x_max_val = float(x_max) if x_max.strip() else np.max(x)
            mask = (x >= x_min_val) & (x <= x_max_val)
            x = x[mask]
            if y1 is not None: y1 = y1[mask]
            if y2 is not None: y2 = y2[mask]

            # --------- Y Range Filtering (Data Masking) -------------
            y_min = self.y_min_entry.get()
            y_max = self.y_max_entry.get()

            # Only apply data cropping if NOT centering zero (because center zero needs full data to find max)
            # OR if user explicitly wants to crop.
            # Generally, limits in plot should just be view limits, not data removal.
            # But original code removed data points outside range. Keeping consistent unless Center Zero is on.

            if not center_zero_active:
                if y1 is not None:
                    if y_min.strip() or y_max.strip():
                        y1_min_val = float(y_min) if y_min.strip() else np.min(y1)
                        y1_max_val = float(y_max) if y_max.strip() else np.max(y1)
                        y_mask = (y1 >= y1_min_val) & (y1 <= y1_max_val)
                        y1 = y1[y_mask]
                        x = x[y_mask]
                        if y2 is not None: y2 = y2[y_mask]
                elif y2 is not None:
                    if y_min.strip() or y_max.strip():
                        y2_min_val = float(y_min) if y_min.strip() else np.min(y2)
                        y2_max_val = float(y_max) if y_max.strip() else np.max(y2)
                        y2_mask = (y2 >= y2_min_val) & (y2 <= y2_max_val)
                        y2 = y2[y2_mask]
                        x = x[y2_mask]

            # ----------- Filters --------------
            filter_type = self.filter_menu.get()
            if filter_type != "None":
                params = {k: w.get() if isinstance(w, ctk.CTkEntry) else w.get()
                          for k, w in (self.filter_frame_vars or {}).items()}
                if y1 is not None: y1 = self.apply_filter(y1, filter_type, x, params)
                if y2 is not None: y2 = self.apply_filter(y2, filter_type, x, params)

            self.last_x = x
            self.last_y = y1 if y1 is not None else y2
            self.last_x_name = x_name
            self.last_y_name = y1_label if y1 is not None else y2_label
            self.last_y2 = y2
            self.last_y2_name = y2_label

            # --------- Gather style & Plot ---------
            line_color1 = self.line_color1_menu.get()
            marker1 = self.marker1_menu.get()
            line_color2 = self.line_color2_menu.get()
            marker2 = self.marker2_menu.get()

            if marker1 == "o (hollow)":
                markerfacecolor1 = "none"
                markeredgecolor1 = line_color1
                marker1 = "o"
            else:
                markerfacecolor1 = line_color1
                markeredgecolor1 = line_color1
                if marker1 == "None": marker1 = None

            if marker2 == "o (hollow)":
                markerfacecolor2 = "none"
                markeredgecolor2 = line_color2
                marker2 = "o"
            else:
                markerfacecolor2 = line_color2
                markeredgecolor2 = line_color2
                if marker2 == "None": marker2 = None

            self.create_plot(
                x, y1, x_name, y1_label, y2, y2_label,
                custom_x_label=self.custom_x_label,
                custom_plot_title=self.custom_plot_title,
                custom_y1_label=self.custom_y1_label,
                custom_y2_label=self.custom_y2_label,
                line_color1=line_color1, marker1=marker1,
                markerfacecolor1=markerfacecolor1, markeredgecolor1=markeredgecolor1,
                line_color2=line_color2, marker2=marker2,
                markerfacecolor2=markerfacecolor2, markeredgecolor2=markeredgecolor2
            )

            # Axis limits & FFT Peak detection
            axes = plt.gcf().get_axes()
            ax_primary = axes[0] if axes else None
            ax_secondary = axes[1] if len(axes) > 1 else None

            # --- APPLY Y-LIMITS LOGIC ---

            if center_zero_active:
                # Symmetric logic
                if ax_primary and y1 is not None:
                    max_abs = max(abs(np.min(y1)), abs(np.max(y1)))
                    if max_abs == 0: max_abs = 1.0
                    limit = max_abs * 1.1
                    ax_primary.set_ylim(-limit, limit)

                if ax_secondary and y2 is not None:
                    max_abs2 = max(abs(np.min(y2)), abs(np.max(y2)))
                    if max_abs2 == 0: max_abs2 = 1.0
                    limit2 = max_abs2 * 1.1
                    ax_secondary.set_ylim(-limit2, limit2)
            else:
                # Manual logic
                y1_min_set = self.y_min_entry.get().strip() != ""
                y1_max_set = self.y_max_entry.get().strip() != ""

                if ax_primary and (y1_min_set or y1_max_set):
                    y_lo = float(self.y_min_entry.get()) if y1_min_set else None
                    y_hi = float(self.y_max_entry.get()) if y1_max_set else None
                    ax_primary.set_ylim(bottom=y_lo, top=y_hi)

                    # If Apply Both is checked, apply Y1 limits to Y2 as well
                    if apply_both_active and ax_secondary:
                        ax_secondary.set_ylim(bottom=y_lo, top=y_hi)

                # Secondary can have its own logic if apply_both is OFF,
                # but currently we don't have separate input boxes for Y2 range.
                # So if Apply Both is OFF, secondary just autoscales.

            # FFT Peaks
            if primary_enabled and "FFT" in (y1_op or "") and y1 is not None:
                peaks = self.detect_fft_peaks(self.fft_freqs, y1)
                self._annotate_peaks(ax_primary, peaks, "FFT peaks (primary)")
                cutoff_freq, suggestion = self.suggest_cutoff_from_fft(y1, self.fft_freqs)
                if hasattr(self, 'cutoff_suggestion_label'): self.cutoff_suggestion_label.configure(text=suggestion)

            if secondary_enabled and "FFT" in (y2_op or "") and y2 is not None:
                peaks = self.detect_fft_peaks(self.fft_freqs, y2)
                self._annotate_peaks(ax_secondary, peaks, "FFT peaks (secondary)")
                if hasattr(self, "cutoff_suggestion_label2"):
                    cutoff_freq2, suggestion2 = self.suggest_cutoff_from_fft(y2, self.fft_freqs)
                    self.cutoff_suggestion_label2.configure(text=suggestion2)

            # Stats & K Factor
            if y2 is not None and y1 is not None:
                stats_text = (
                    f"Statistics Primary ({line_color1.capitalize()}):\n"
                    f"Mean: {np.mean(y1):.4g} | Min: {np.min(y1):.4g} | Max: {np.max(y1):.4g}\n"
                    f"Range: {np.max(y1) - np.min(y1):.4g}\n"
                    f"\n"
                    f"Statistics Secondary ({line_color2.capitalize()}):\n"
                    f"Mean: {np.mean(y2):.4g} | Min: {np.min(y2):.4g} | Max: {np.max(y2):.4g}\n"
                    f"Range: {np.max(y2) - np.min(y2):.4g}\n"
                )
                try:
                    stats_text += f"\n----- Analysis -----\n"
                    valid = np.isfinite(y1) & np.isfinite(y2)
                    if np.sum(valid) > 10:
                        a_vec = y1[valid].reshape(-1, 1)
                        b_vec = y2[valid]
                        k_val = np.linalg.lstsq(a_vec, b_vec, rcond=None)[0][0]
                        self.last_k = k_val
                        stats_text += f"Scale Factor (K): {k_val:.5g}\n(Y2 ≈ K * Y1)\n"

                        dt = np.mean(np.diff(x[valid]))
                        if dt > 0:
                            y1_ac = y1[valid] - np.mean(y1[valid])
                            y2_ac = y2[valid] - np.mean(y2[valid])
                            fft1 = np.fft.rfft(y1_ac)
                            fft2 = np.fft.rfft(y2_ac)
                            freqs = np.fft.rfftfreq(len(y1_ac), d=dt)
                            idx = np.argmax(np.abs(fft1[1:])) + 1
                            dom_freq = freqs[idx]
                            if dom_freq > 0:
                                phase1 = np.angle(fft1[idx])
                                phase2 = np.angle(fft2[idx])
                                phase_diff_deg = np.degrees(phase2 - phase1)
                                phase_diff_deg = (phase_diff_deg + 180) % 360 - 180
                                stats_text += f"Phase Shift: {phase_diff_deg:.2f}°\n(at {dom_freq:.1f} Hz)"
                except Exception as e:
                    pass
                self.plot_info.configure(text=stats_text)
            elif y1 is not None:
                # Stats code for single axis...
                max_idx_y = np.argmax(y1)
                x_max_y = x[max_idx_y]
                y_max = y1[max_idx_y]
                stats_text = (
                    f"Statistics Primary ({line_color1.capitalize()}):\n"
                    f"Mean: {np.mean(y1):.4g}\n"
                    f"Min: {np.min(y1):.4g}\n"
                    f"Max: {np.max(y1):.4g}\n"
                    f"Range: {np.max(y1) - np.min(y1):.4g}\n"
                    f"Max at: (X={x_max_y:.4g}, Y={y_max:.4g})"
                )
                self.plot_info.configure(text=stats_text)
            elif y2 is not None:
                max_idx_y2 = np.argmax(y2)
                x_max_y2 = x[max_idx_y2]
                y2_max = y2[max_idx_y2]
                stats_text = (
                    f"Statistics Secondary ({line_color2.capitalize()}):\n"
                    f"Mean: {np.mean(y2):.4g}\n"
                    f"Min: {np.min(y2):.4g}\n"
                    f"Max: {np.max(y2):.4g}\n"
                    f"Range: {np.max(y2) - np.min(y2):.4g}\n"
                    f"Max at: (X={x_max_y2:.4g}, Y={y2_max:.4g})"
                )
                self.plot_info.configure(text=stats_text)
            else:
                self.plot_info.configure(text="No data to display")

            self.update_step_status(2, True)
            self.update_step_status(3, True)
            self.export_button.configure(state="normal")

            # Hide progress when done
            self.hide_progress()
            self.update_status("Plot generated successfully")

            if hasattr(self, 'auto_fit_after_plot') and self.auto_fit_after_plot.get():
                try:
                    self.auto_fit()
                except Exception:
                    pass

        except Exception as e:
            messagebox.showerror("Plot Error", f"Error generating plot: {e}")
            self.update_step_status(2, False)
            self.update_step_status(3, False)
            self.hide_progress()
            self.update_status("Error generating plot")

    def evaluate_math_expression(self, expr):
        context = {col: self.data[col].values for col in self.data.columns}
        context.update({
            "pi": np.pi,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "sqrt": np.sqrt, "log": np.log, "abs": np.abs,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        })
        try:
            y = ne.evaluate(expr, local_dict=context)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Expression Error", f"Could not evaluate expression:\n{e}")
            raise
        return y

    def create_plot(self, x, y, x_label, y_label, y2=None, y2_label=None,
                    custom_x_label=None, custom_plot_title=None, custom_y1_label=None, custom_y2_label=None,
                    line_color1="#1F77B4", marker1="o", markerfacecolor1="#1F77B4", markeredgecolor1="#1F77B4",
                    line_color2="orange", marker2=None, markerfacecolor2="orange", markeredgecolor2="orange"):
        """Create a plot with optional secondary y-axis and custom labels, style per user selection"""
        # Remove previous plot if exists
        for widget in self.plot_placeholder.winfo_children():
            widget.destroy()

        # Also remove placeholder text if it exists (it uses place, not pack/grid)
        if hasattr(self, 'plot_placeholder_text'):
            try:
                self.plot_placeholder_text.destroy()
            except:
                pass

        self.fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)

        # Axis appearance settings
        axis_color = self.axis_color_menu.get() if hasattr(self, "axis_color_menu") else "black"
        tick_fontsize = int(self.tick_fontsize_menu.get()) if hasattr(self, "tick_fontsize_menu") else 12
        show_legend = self.show_legend_var.get() if hasattr(self, "show_legend_var") else True

        x_units = self.x_units.get()
        y1_units = self.y1_units.get()
        y2_units = self.y2_units.get() if y2 is not None else None

        if custom_x_label and custom_x_label.strip():
            x_label_with_units = custom_x_label
        else:
            x_label_with_units = x_label
            if x_units != "None":
                x_label_with_units += f" ({x_units})"

        if custom_y1_label and custom_y1_label.strip():
            y1_label_with_units = custom_y1_label
        else:
            y1_label_with_units = y_label if y_label else ""
            if y1_units != "None" and y_label:
                if "(" not in y1_label_with_units:
                    y1_label_with_units += f" ({y1_units})"

        if custom_y2_label and custom_y2_label.strip():
            y2_label_with_units = custom_y2_label
        else:
            y2_label_with_units = y2_label if y2_label else ""
            if y2 is not None and y2_units != "None" and y2_label:
                if "(" not in y2_label_with_units:
                    y2_label_with_units += f" ({y2_units})"

        # Get plot type
        plot_type = self.plot_type_menu.get() if hasattr(self, 'plot_type_menu') else "Line"

        # Plot the primary y-axis data if available
        line1 = []
        if y is not None:
            if plot_type == "Scatter":
                line1 = ax1.scatter(
                    x, y,
                    s=30,
                    color=line_color1 if line_color1 else "#1F77B4",
                    marker=marker1 if marker1 and marker1 != "None" else "o",
                    label=y1_label_with_units,
                    alpha=0.7
                )
            elif plot_type == "Line+Scatter":
                line1 = ax1.plot(
                    x, y,
                    linewidth=2,
                    color=line_color1 if line_color1 else "#1F77B4",
                    marker=marker1 if marker1 and marker1 != "None" else "o",
                    markerfacecolor=markerfacecolor1 if markerfacecolor1 else line_color1,
                    markeredgecolor=markeredgecolor1 if markeredgecolor1 else line_color1,
                    markersize=6,
                    label=y1_label_with_units
                )
            elif plot_type == "Bar":
                line1 = ax1.bar(
                    x, y,
                    color=line_color1 if line_color1 else "#1F77B4",
                    alpha=0.7,
                    width=(np.max(x) - np.min(x)) / len(x) * 0.8 if len(x) > 1 else 0.1,
                    label=y1_label_with_units
                )
            elif plot_type == "Histogram":
                line1 = ax1.hist(
                    y,
                    bins=min(50, len(y) // 10) if len(y) > 10 else 10,
                    color=line_color1 if line_color1 else "#1F77B4",
                    alpha=0.7,
                    label=y1_label_with_units
                )
                ax1.set_xlabel(y1_label_with_units, fontsize=12)
                ax1.set_ylabel("Frequency", fontsize=12, color=axis_color)
            elif plot_type == "Step":
                line1 = ax1.step(
                    x, y,
                    linewidth=2,
                    color=line_color1 if line_color1 else "#1F77B4",
                    where='mid',
                    label=y1_label_with_units
                )
            else:  # Line (default)
                line1 = ax1.plot(
                    x, y,
                    linewidth=2,
                    color=line_color1 if line_color1 else "#1F77B4",
                    marker=marker1 if marker1 and marker1 != "None" else "",
                    markerfacecolor=markerfacecolor1 if markerfacecolor1 else line_color1,
                    markeredgecolor=markeredgecolor1 if markeredgecolor1 else line_color1,
                    linestyle="None" if marker1 and marker1 != "None" else "-",
                    label=y1_label_with_units
                )

            if plot_type != "Histogram":
                ax1.set_xlabel(x_label_with_units, fontsize=12)
                ax1.set_ylabel(y1_label_with_units, fontsize=12, color=axis_color)
            ax1.tick_params(axis='y', colors=axis_color)

        # Apply axis colors and tick sizes to primary axis
        for spine in ['bottom', 'left', 'top', 'right']:
            ax1.spines[spine].set_color(axis_color)
        ax1.tick_params(axis='x', colors=axis_color)
        ax1.xaxis.label.set_color(axis_color)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)

        # Add a secondary y-axis if needed
        line2 = []
        if y2 is not None:
            ax2 = ax1.twinx()
            if plot_type == "Scatter":
                line2 = ax2.scatter(
                    x, y2,
                    s=30,
                    color=line_color2 if line_color2 else "orange",
                    marker=marker2 if marker2 and marker2 != "None" else "s",
                    label=y2_label_with_units,
                    alpha=0.7
                )
            elif plot_type == "Line+Scatter":
                line2 = ax2.plot(
                    x, y2,
                    linewidth=2,
                    color=line_color2 if line_color2 else "orange",
                    marker=marker2 if marker2 and marker2 != "None" else "s",
                    markerfacecolor=markerfacecolor2 if markerfacecolor2 else line_color2,
                    markeredgecolor=markeredgecolor2 if markeredgecolor2 else line_color2,
                    markersize=6,
                    linestyle="--",
                    label=y2_label_with_units
                )
            elif plot_type == "Step":
                line2 = ax2.step(
                    x, y2,
                    linewidth=2,
                    color=line_color2 if line_color2 else "orange",
                    where='mid',
                    linestyle="--",
                    label=y2_label_with_units
                )
            else:  # Line, Bar, Histogram
                line2 = ax2.plot(
                    x, y2,
                    linewidth=2,
                    color=line_color2 if line_color2 else "orange",
                    marker=marker2 if marker2 and marker2 != "None" else "",
                    markerfacecolor=markerfacecolor2 if markerfacecolor2 else line_color2,
                    markeredgecolor=markeredgecolor2 if markeredgecolor2 else line_color2,
                    linestyle="None" if marker2 and marker2 != "None" else "--",
                    label=y2_label_with_units
                )
            ax2.set_ylabel(y2_label_with_units, fontsize=12, color=axis_color)
            ax2.tick_params(axis='y', colors=axis_color)
            # Apply axis colors and tick sizes to secondary axis
            for spine in ['right', 'top', 'bottom', 'left']:
                if spine in ax2.spines:
                    ax2.spines[spine].set_color(axis_color)
            ax2.tick_params(axis='both', labelsize=tick_fontsize)
            if y is not None and show_legend:
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right')
        elif y is not None and show_legend:
            ax1.legend(loc='upper right')

        if custom_plot_title and custom_plot_title.strip():
            ax1.set_title(custom_plot_title, fontsize=14)
        elif y2 is not None and y is not None:
            ax1.set_title(f"Data Analysis: Dual Y-Axis Plot", fontsize=14)
        elif y is not None:
            ax1.set_title(f"Data Analysis: {y1_label_with_units} vs {x_label_with_units}", fontsize=14)
        elif y2 is not None:
            ax1.set_title(f"Data Analysis: {y2_label_with_units} vs {x_label_with_units}", fontsize=14)

        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.spines['top'].set_visible(False)
        if y2 is None:
            ax1.spines['right'].set_visible(False)

        self.fig.tight_layout()

        # Configure plot_placeholder grid if not already configured
        self.plot_placeholder.grid_rowconfigure(1, weight=1)
        self.plot_placeholder.grid_columnconfigure(0, weight=1)

        # Add navigation toolbar for zoom/pan
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ctk.CTkFrame(self.plot_placeholder)
        toolbar_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Create canvas with interactive features
        canvas = FigureCanvasTkAgg(self.fig, master=self.plot_placeholder)
        canvas.draw()

        self.navigation_toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        self.navigation_toolbar.update()

        # Grid canvas after toolbar - ensure it's visible
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, sticky="nsew")
        self.canvas_widget = canvas

        # Force update to ensure visibility
        self.plot_placeholder.update_idletasks()
        self.window.update_idletasks()

        # Add data cursor functionality
        self.annot = None
        if y is not None or y2 is not None:
            # Create annotation for data cursor
            self.annot = ax1.annotate('', xy=(0,0), xytext=(20,20),
                                     textcoords="offset points",
                                     bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                                     arrowprops=dict(arrowstyle="->"))
            self.annot.set_visible(False)

            def on_hover(event):
                if event.inaxes == ax1 and self.annot and event.xdata is not None:
                    try:
                        # Find closest point
                        idx = np.argmin(np.abs(x - event.xdata))
                        x_val = x[idx]
                        y_val = y[idx] if y is not None else None
                        y2_val = y2[idx] if y2 is not None else None

                        # Update annotation
                        y_display = y_val if y_val is not None else (y2_val if y2_val is not None else 0)
                        self.annot.xy = (x_val, y_display)
                        text = f'X: {x_val:.4f}'
                        if y_val is not None:
                            text += f'\nY1: {y_val:.4f}'
                        if y2_val is not None:
                            text += f'\nY2: {y2_val:.4f}'
                        self.annot.set_text(text)
                        self.annot.set_visible(True)
                        canvas.draw_idle()
                    except Exception:
                        pass

            def on_leave(event):
                if self.annot:
                    self.annot.set_visible(False)
                    canvas.draw_idle()

            canvas.mpl_connect('motion_notify_event', on_hover)
            canvas.mpl_connect('axes_leave_event', on_leave)

        # Update window to ensure plot is visible
        self.window.update_idletasks()

    def detect_and_plot_peaks(self):
        """Detect and visualize peaks on the current plot"""
        if not hasattr(self, 'last_x') or not hasattr(self, 'last_y'):
            messagebox.showwarning("No Data", "Please generate a plot first.")
            return

        from scipy.signal import find_peaks

        # Ask user for peak detection parameters
        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Peak Detection Parameters")
        dialog.geometry("400x250")
        dialog.transient(self.window)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Peak Detection Parameters", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)

        height_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        height_frame.pack(pady=5)
        ctk.CTkLabel(height_frame, text="Min Height (% of max):", width=150).pack(side="left", padx=5)
        height_entry = ctk.CTkEntry(height_frame, width=100)
        height_entry.insert(0, "50")
        height_entry.pack(side="left", padx=5)

        distance_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        distance_frame.pack(pady=5)
        ctk.CTkLabel(distance_frame, text="Min Distance (points):", width=150).pack(side="left", padx=5)
        distance_entry = ctk.CTkEntry(distance_frame, width=100)
        distance_entry.insert(0, "10")
        distance_entry.pack(side="left", padx=5)

        prominence_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        prominence_frame.pack(pady=5)
        ctk.CTkLabel(prominence_frame, text="Prominence (% of max):", width=150).pack(side="left", padx=5)
        prominence_entry = ctk.CTkEntry(prominence_frame, width=100)
        prominence_entry.insert(0, "30")
        prominence_entry.pack(side="left", padx=5)

        result = [None]

        def detect():
            try:
                y_data = self.last_y
                height_pct = float(height_entry.get()) / 100.0
                distance = int(distance_entry.get())
                prominence_pct = float(prominence_entry.get()) / 100.0

                height = np.max(y_data) * height_pct
                prominence = (np.max(y_data) - np.min(y_data)) * prominence_pct

                peaks, properties = find_peaks(
                    y_data,
                    height=height,
                    distance=distance,
                    prominence=prominence
                )

                result[0] = peaks
                dialog.destroy()

                # Plot peaks on current figure
                if self.fig:
                    ax = self.fig.get_axes()[0]
                    x_peaks = self.last_x[peaks]
                    y_peaks = y_data[peaks]

                    # Mark peaks
                    ax.plot(x_peaks, y_peaks, 'ro', markersize=10, label=f'Peaks ({len(peaks)})', zorder=5)

                    # Annotate peaks
                    for i, (x_p, y_p) in enumerate(zip(x_peaks, y_peaks)):
                        ax.annotate(
                            f'P{i+1}\n({x_p:.2f}, {y_p:.2f})',
                            xy=(x_p, y_p),
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                            fontsize=8
                        )

                    ax.legend()
                    if hasattr(self, 'canvas_widget') and self.canvas_widget:
                        self.canvas_widget.draw()

                    messagebox.showinfo("Peak Detection", f"Found {len(peaks)} peaks!")
                    self.update_status(f"Detected {len(peaks)} peaks")

            except Exception as e:
                messagebox.showerror("Error", f"Peak detection failed: {e}")

        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=20)

        ctk.CTkButton(button_frame, text="Detect", command=detect, width=100).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy, width=100).pack(side="left", padx=10)

        dialog.wait_window()

    def edit_channel_names(self):
        """Open a dialog to edit channel names"""
        if not self.channel_names:
            messagebox.showinfo("No Channels", "Please load data first to see available channels.")
            return

        # Create a toplevel window for the dialog
        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Edit Channel Names")
        dialog.geometry("400x400")
        dialog.transient(self.window)  # Set to be on top of the main window
        dialog.grab_set()  # Make the dialog modal

        # Add padding
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_columnconfigure(1, weight=2)

        # Create a scrollable frame to contain all channel entries
        scroll_frame = ctk.CTkScrollableFrame(dialog, width=380, height=300)
        scroll_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Dictionary to store entry widgets
        channel_entries = {}

        # Add header
        header_label = ctk.CTkLabel(
            scroll_frame,
            text="Original Name → Custom Name",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        header_label.pack(pady=(5, 10), anchor="w")

        # Initialize the channel_custom_names dictionary if it doesn't exist
        if not hasattr(self, 'channel_custom_names'):
            self.channel_custom_names = {}

        # Create an entry for each channel
        for i, channel in enumerate(self.channel_names):
            channel_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            channel_frame.pack(fill="x", pady=5)

            # Original channel name
            original_label = ctk.CTkLabel(
                channel_frame,
                text=f"{channel} →",
                width=150,
                anchor="e"
            )
            original_label.pack(side="left", padx=(0, 10))

            # Entry field for custom name
            custom_entry = ctk.CTkEntry(channel_frame, width=200)
            if channel in self.channel_custom_names:
                custom_entry.insert(0, self.channel_custom_names[channel])
            custom_entry.pack(side="left")

            # Store the entry widget reference
            channel_entries[channel] = custom_entry

        # Helper text
        help_text = ctk.CTkLabel(
            dialog,
            text="Leave blank to use original channel names.",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray70")
        )
        help_text.grid(row=1, column=0, columnspan=2, padx=10, pady=(5, 10))

        # Buttons
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        def apply_names():
            # Store custom names in a class variable
            for channel, entry in channel_entries.items():
                custom_name = entry.get().strip()
                if custom_name:
                    self.channel_custom_names[channel] = custom_name
                elif channel in self.channel_custom_names:
                    # Remove the entry if the field is blank
                    del self.channel_custom_names[channel]

            # Refresh dropdowns to display new names
            self.refresh_dropdowns()

            # Close the dialog
            dialog.destroy()

        apply_btn = ctk.CTkButton(button_frame, text="Apply", command=apply_names)
        apply_btn.pack(side="left", padx=10)

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40")
        )
        cancel_btn.pack(side="left", padx=10)

        ### 2. Modify the create_layout method to add the channel naming button

        # Add this right after the logo_label.pack() line in create_layout:
        # Channel naming button
        self.channel_names_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="Edit Channel Names",
            command=self.edit_channel_names,
            height=30
        )
        self.channel_names_btn.pack(pady=(5, 15))

    ### 3. Modify the get_series method to use custom channel names

    def get_series(self, name):
        """Get a data series with applied signal conditioning if any"""
        # Find the original column name if this is a custom name
        original_name = name
        if hasattr(self, 'channel_custom_names'):
            for orig, custom in self.channel_custom_names.items():
                if custom == name:
                    original_name = orig
                    break

        if original_name in self.data.columns:
            series = self.data[original_name].values

            # Apply any signal conditioning multiplier if configured
            if original_name in self.channel_conditioning:
                cond_type = self.channel_conditioning[original_name].get()
                if cond_type == "LeCroy (x20)":
                    series = series * 20.0
                elif cond_type == "LeCroy (x200)":
                    series = series * 200.0
                elif cond_type == "Rogowski (x5000)":
                    series = series * 5000.0
                elif cond_type == "Shant (x1000)":
                    series = series * 1000.0
                elif cond_type == "X50":
                    series = series * 50.0
                elif cond_type == "X500":
                    series = series * 500.0
                elif cond_type == "X10":
                    series = series * 10.0

            return series
        return None

    ### 4. Modify the refresh_dropdowns method to display custom channel names

    def refresh_dropdowns(self):
        original_active = [name for name, var in self.active_channels.items() if var.get()]

        # Create a list with custom names if available
        active = []
        for name in original_active:
            if hasattr(self, 'channel_custom_names') and name in self.channel_custom_names:
                active.append(self.channel_custom_names[name])
            else:
                active.append(name)

        # Update X-axis dropdown
        self.x_selector.configure(values=[self.time_column] + active)
        self.x_selector.set(self.time_column)

        # Update Channel A and B dropdowns for primary Y-axis
        if active:
            self.a_menu.configure(values=active)
            self.a_menu.set(active[0])

            self.b_menu.configure(values=active)
            self.b_menu.set(active[0] if len(active) == 1 else active[1] if len(active) > 1 else active[0])

            # Update Channel C and D dropdowns for secondary Y-axis
            self.c_menu.configure(values=active)
            self.c_menu.set(active[0])

            self.d_menu.configure(values=active)
            self.d_menu.set(active[0] if len(active) == 1 else active[1] if len(active) > 1 else active[0])
        else:
            self.a_menu.configure(values=["None"])
            self.a_menu.set("None")

            self.b_menu.configure(values=["None"])
            self.b_menu.set("None")

            self.c_menu.configure(values=["None"])
            self.c_menu.set("None")

            self.d_menu.configure(values=["None"])
            self.d_menu.set("None")

        self.update_y_label()
        self.update_y2_label()
        self.update_filter_params()  # Initialize the filter indicator

        # Update signal conditioning display (some channels may have been deselected)
        self.setup_signal_conditioning()

    def save_results(self):
        if not hasattr(self, 'last_x') or not hasattr(self, 'last_y'):
            messagebox.showerror("Error", "No plot data available to save.")
            return

        # 1. Prepare list of available "Sources"
        available_sources = []

        # Add Calculated Results first
        available_sources.append("Calculated Y1")
        if hasattr(self, 'last_y2') and self.last_y2 is not None:
            available_sources.append("Calculated Y2")

        # Add Raw Channels
        available_sources.extend(self.channel_names)

        # Check if we have a K factor from the last plot analysis
        k_val = getattr(self, 'last_k', None)

        # 2. Define the callback for when user clicks "Export"
        def perform_export(export_config):
            if not export_config:
                return

            file_path = filedialog.asksaveasfilename(
                title="Save Results As",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file_path: return

            base_path = os.path.splitext(file_path)[0]
            plot_path = f"{base_path}_plot.png"

            data_dict = {}

            # --- MASTER AXIS ---
            # We use the X-axis from the last plot as the absolute reference
            master_x = self.last_x

            x_col_name = self.last_x_name
            if self.x_units.get() != "None" and "(" not in x_col_name:
                x_col_name += f" ({self.x_units.get()})"
            data_dict[x_col_name] = master_x

            # Check if we are in FFT mode (to prevent mixing Time/Freq domains)
            is_fft = "Frequency" in self.last_x_name

            # Process each selected column
            for item in export_config:
                source = item['source']
                target_name = item['target_name']
                scale = item['scale']

                series_data = None

                # A. Handle Virtual Calculated Channels (Already matching Master X)
                if source == "Calculated Y1":
                    series_data = self.last_y
                elif source == "Calculated Y2":
                    series_data = self.last_y2

                # B. Handle Raw Channels (Need Alignment)
                elif source in self.channel_names:
                    if is_fft:
                        print(f"Skipping {source} - cannot export time data on Frequency axis")
                        continue

                    # Get raw data and its original full time axis
                    raw_y = self.get_series(source)
                    if raw_y is None: continue

                    # Get the full original X axis (Time)
                    x_raw_name = self.x_selector.get()
                    raw_x = self.get_series(x_raw_name)

                    # --- THE FIX: INTERPOLATION ---
                    # Instead of masking/cropping, we interpolate the raw data
                    # to match the exact points of master_x.
                    try:
                        # np.interp(x_new, x_original, y_original)
                        series_data = np.interp(master_x, raw_x, raw_y)

                        # Optional: Apply Filter if active (so export matches visual "WYSIWYG")
                        filter_type = self.filter_menu.get()
                        if filter_type != "None":
                            params = {k: w.get() if isinstance(w, ctk.CTkEntry) else w.get()
                                      for k, w in (self.filter_frame_vars or {}).items()}
                            # Note: applying filter on the interpolated chunk
                            series_data = self.apply_filter(series_data, filter_type, master_x, params)

                    except Exception as e:
                        print(f"Error processing {source}: {e}")
                        continue

                # C. Apply Scaling & Add to Dictionary
                if series_data is not None:
                    # Final safety check
                    if len(series_data) == len(master_x):
                        data_dict[target_name] = series_data * scale
                    else:
                        print(f"Length mismatch for {target_name}, skipping.")

            # Save to disk
            try:
                df_out = pd.DataFrame(data_dict)
                df_out.to_csv(file_path, index=False)

                # Save Plot image as well
                if self.fig:
                    self.fig.savefig(plot_path, dpi=300, bbox_inches="tight")

                messagebox.showinfo("Success", f"Data saved to:\n{file_path}\n\nPlot saved to:\n{plot_path}")

            except Exception as e:
                messagebox.showerror("Export Error", str(e))

        # 3. Open the Dialog
        AdvancedExportDialog(self.window, available_sources, k_factor=k_val, on_export=perform_export)

    def reset_app(self):
        """Reset the application to its initial state"""
        # Confirm reset
        if messagebox.askyesno("Confirm Reset",
                               "Are you sure you want to reset the application? This will clear all data and settings."):
            # Clear data
            self.data = None
            self.active_channels = {}
            self.channel_names = {}
            self.channel_conditioning = {}

            # Reset file section
            self.file_label.configure(
                text="No file loaded",
                text_color=("gray50", "gray70"),
                font=ctk.CTkFont(size=12)
            )

            # Clear channel checkboxes
            for widget in self.checkbox_frame.winfo_children():
                widget.destroy()

            self.no_channels_label = ctk.CTkLabel(
                self.checkbox_frame,
                text="Load a data file to see available channels",
                font=ctk.CTkFont(size=12, slant="italic"),
                text_color=("gray50", "gray70")
            )
            self.no_channels_label.pack(pady=10)

            # Clear channel conditioning options
            for widget in self.channel_cond_container.winfo_children():
                widget.destroy()

            self.no_channels_cond_label = ctk.CTkLabel(
                self.channel_cond_container,
                text="Load data to see channels",
                font=ctk.CTkFont(size=10, slant="italic"),
                text_color=("gray50", "gray70")
            )
            self.no_channels_cond_label.pack(pady=2)

            # Reset dropdowns
            self.x_selector.configure(values=["Time"])
            self.a_menu.configure(values=["None"])
            self.b_menu.configure(values=["None"])
            self.a_menu.set("None")
            self.b_menu.set("None")

            # Reset secondary Y-axis dropdowns
            self.c_menu.configure(values=["None"])
            self.d_menu.configure(values=["None"])
            self.c_menu.set("None")
            self.d_menu.set("None")

            # Reset units
            self.x_units.set("s")
            self.y1_units.set("None")
            self.y2_units.set("None")

            # Reset X/Y range entries
            self.reset_x_range()
            self.reset_y_range()

            # Reset axis states
            self.primary_enabled.set(True)
            self.secondary_enabled.set(False)

            # Reset to primary tab
            self.tab_var.set(0)
            self.switch_y_axis_tab()

            # Reset filter
            self.filter_menu.set("None")
            self.update_filter_params("None")

            # Clear plot
            for widget in self.plot_placeholder.winfo_children():
                widget.destroy()

            self.plot_placeholder_text = ctk.CTkLabel(
                self.plot_placeholder,
                text="No data visualization available\nComplete steps 1-3 to generate a plot",
                font=ctk.CTkFont(size=14)
            )
            self.plot_placeholder_text.place(relx=0.5, rely=0.5, anchor="center")

            # Reset plot info
            self.plot_title.configure(text="Results Visualization")
            self.plot_info.configure(
                text="Load data and configure plot settings to visualize results",
                font=ctk.CTkFont(size=12),
                text_color=("gray50", "gray70")
            )

            # Disable export button
            self.export_button.configure(state="disabled")

            # Reset step indicators
            for i in range(len(self.steps)):
                self.update_step_status(i, False)

            # If we have a figure, close it
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
                self.fig = None

    # Signal Processing Methods
    @staticmethod
    def symmetric_derivative(y, x):
        """Calculate a symmetric derivative with protection against division by zero"""
        y = np.asarray(y)
        x = np.asarray(x)
        dy = np.zeros_like(y)

        # Center derivative
        dx_center = x[2:] - x[:-2]
        # Avoid division by zero
        dx_center[dx_center == 0] = np.finfo(float).eps
        dy[1:-1] = (y[2:] - y[:-2]) / dx_center

        # Forward/backward at endpoints
        dx_start = x[1] - x[0]
        if dx_start == 0:
            dx_start = np.finfo(float).eps

        dx_end = x[-1] - x[-2]
        if dx_end == 0:
            dx_end = np.finfo(float).eps

        dy[0] = (y[1] - y[0]) / dx_start
        dy[-1] = (y[-1] - y[-2]) / dx_end

        return dy

    @staticmethod
    def integrate_trapezoidal(y, x):
        """Calculate the integral using the trapezoidal rule"""
        integral = np.zeros_like(y)
        for i in range(1, len(y)):
            dx = x[i] - x[i - 1]
            integral[i] = integral[i - 1] + 0.5 * dx * (y[i] + y[i - 1])
        return integral

    @staticmethod
    def moving_average(data, window_size):
        """Apply a moving average filter"""
        # Use numpy's convolve for efficiency
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')

    @staticmethod
    def design_butter_sos(cutoff, fs, order, btype):
        """Design a Butterworth filter as second-order sections for better numerical stability"""
        nyq = 0.5 * fs
        if isinstance(cutoff, (list, tuple)):
            normalized_cutoff = [c / nyq for c in cutoff]
        else:
            normalized_cutoff = cutoff / nyq

        # Create a second-order sections (SOS) filter for better numerical stability
        sos = butter(order, normalized_cutoff, btype=btype, analog=False, output='sos')
        return sos

    @staticmethod
    def design_cheby_sos(cutoff, fs, order, btype, rp=1.0):
        """Design a Chebyshev Type I filter as second-order sections"""
        nyq = 0.5 * fs
        if isinstance(cutoff, (list, tuple)):
            normalized_cutoff = [c / nyq for c in cutoff]
        else:
            normalized_cutoff = cutoff / nyq

        # Create a second-order sections (SOS) filter
        sos = cheby1(order, rp, normalized_cutoff, btype=btype, analog=False, output='sos')
        return sos

    def apply_lowpass(self, data, cutoff, fs, order=4, design="Butterworth"):
        """Apply a low-pass filter to the data"""
        if design == "Butterworth":
            sos = self.design_butter_sos(cutoff, fs, order, 'lowpass')
        else:  # Chebyshev
            sos = self.design_cheby_sos(cutoff, fs, order, 'lowpass')

        # Apply the filter
        filtered_data = sosfilt(sos, data)
        return filtered_data

    def apply_highpass(self, data, cutoff, fs, order=4, design="Butterworth"):
        """Apply a high-pass filter to the data"""
        if design == "Butterworth":
            sos = self.design_butter_sos(cutoff, fs, order, 'highpass')
        else:  # Chebyshev
            sos = self.design_cheby_sos(cutoff, fs, order, 'highpass')

        # Apply the filter
        filtered_data = sosfilt(sos, data)
        return filtered_data

    def apply_bandpass(self, data, lowcut, highcut, fs, order=4, design="Butterworth"):
        """Apply a band-pass filter to the data"""
        if design == "Butterworth":
            sos = self.design_butter_sos([lowcut, highcut], fs, order, 'bandpass')
        else:  # Chebyshev
            sos = self.design_cheby_sos([lowcut, highcut], fs, order, 'bandpass')

        # Apply the filter
        filtered_data = sosfilt(sos, data)
        return filtered_data

    def apply_bandstop(self, data, lowcut, highcut, fs, order=4, design="Butterworth"):
        """Apply a band-stop filter to the data"""
        if design == "Butterworth":
            sos = self.design_butter_sos([lowcut, highcut], fs, order, 'bandstop')
        else:  # Chebyshev
            sos = self.design_cheby_sos([lowcut, highcut], fs, order, 'bandstop')

        # Apply the filter
        filtered_data = sosfilt(sos, data)
        return filtered_data

    def compute_y(self, op, a, b, x, use_dc_correction=False):
        """
        Compute y values based on selected operation.
        'Inductance L' logic updated: Uses INTERPOLATION to fill gaps where dI/dt is too small.
        """
        from scipy.signal import savgol_filter
        import numpy as np

        # --- Helper: Get Baseline ---
        def get_baseline(arr):
            if arr is None: return 0.0
            n_samples = max(10, int(len(arr) * 0.10))
            if n_samples > len(arr): n_samples = len(arr)
            return np.median(arr[:n_samples])

        # --- Helper: Force End to Zero ---
        def force_end_to_zero(y_integrated, x_time):
            if len(y_integrated) < 2: return y_integrated
            y_start = y_integrated[0]
            y_end = y_integrated[-1]
            t_start = x_time[0]
            t_end = x_time[-1]
            drift_slope = (y_end - y_start) / (t_end - t_start)
            correction_ramp = drift_slope * (x_time - t_start)
            return y_integrated - correction_ramp

        is_integral = "∫" in op

        # --- 1. PRE-PROCESSING (DC Correction) ---
        if use_dc_correction:
            if a is not None: a = a - get_baseline(a)
            if b is not None: b = b - get_baseline(b)

        # --- 2. PERFORM OPERATION ---

        if op == "Custom Expression...":
            expr = self.custom_expr
            local_dict = {}
            for col in self.data.columns:
                series = self.get_series(col)
                if use_dc_correction and series is not None:
                    series = series - get_baseline(series)
                local_dict[col] = series
            try:
                local_dict['np'] = np
                y = eval(expr, {"__builtins__": {}}, local_dict)
                return y
            except Exception as e:
                return np.zeros_like(x)

        if op == "Single":
            return a
        elif op == "A + B":
            return a + b
        elif op == "A - B":
            return a - b
        elif op == "A * B":
            return a * b
        elif op == "A / B":
            with np.errstate(divide='ignore', invalid='ignore'):
                res = a / b
                res[np.abs(b) < 1e-9] = 0  # Simple fallback for basic division
            return res

        # --- INDUCTANCE CALCULATION (With Interpolation) ---
        elif "Inductance L" in op:
            # L = V / (dI/dt)

            # Step A: Determine dI/dt
            if "Direct dI/dt" in op:
                dI_dt = b
            else:
                try:
                    dI_dt = savgol_filter(b, window_length=11, polyorder=3, deriv=1, delta=x[1] - x[0])
                except:
                    dI_dt = self.symmetric_derivative(b, x)

            # Step B: Smart Interpolation
            # Define threshold for "bad data" (too close to zero)
            threshold = np.max(np.abs(dI_dt)) * 0.02

            # Identify good points
            valid_mask = np.abs(dI_dt) > threshold

            # Initialize result array
            L = np.zeros_like(a)

            # 1. Calculate real physics where possible
            L[valid_mask] = a[valid_mask] / dI_dt[valid_mask]

            # 2. Fill the gaps using interpolation
            # If we have valid points, we use them to interpolate the invalid ones (gaps)
            if np.sum(valid_mask) > 1:
                # x values of valid points
                x_valid = x[valid_mask]
                # L values at valid points
                L_valid = L[valid_mask]

                # x values where we need to fill in the blanks
                x_invalid = x[~valid_mask]

                # Interpolate!
                L[~valid_mask] = np.interp(x_invalid, x_valid, L_valid)

            return L

        # --- Standard Operations ---
        elif op == "dA/dt":
            return self.symmetric_derivative(a, x)

        elif op == "∫A dt":
            res = self.integrate_trapezoidal(a, x)
            if use_dc_correction: res = force_end_to_zero(res, x)
            return res

        elif op == "d(A - B)/dt":
            return self.symmetric_derivative(a - b, x)

        elif op == "∫(A - B) dt":
            diff = a - b
            res = self.integrate_trapezoidal(diff, x)
            if use_dc_correction: res = force_end_to_zero(res, x)
            return res

        elif op == "A * dB/dt":
            return a * self.symmetric_derivative(b, x)

        elif op == "A / dB/dt":
            db_dt = self.symmetric_derivative(b, x)
            with np.errstate(divide='ignore', invalid='ignore'):
                res = a / db_dt
                res[np.abs(db_dt) < 1e-9] = 0
            return res

        elif op == "dA/dt / B":
            da_dt = self.symmetric_derivative(a, x)
            with np.errstate(divide='ignore', invalid='ignore'):
                res = da_dt / b
                res[np.abs(b) < 1e-9] = 0
            return res

        elif op == "∫A dt / B":
            int_a = self.integrate_trapezoidal(a, x)
            if use_dc_correction: int_a = force_end_to_zero(int_a, x)
            with np.errstate(divide='ignore', invalid='ignore'):
                res = int_a / b
                res[np.abs(b) < 1e-9] = 0
            return res

        elif op.startswith("FFT of"):
            if "A" in op:
                fft_data, freqs = self.compute_fft(a, x, is_primary=True)
            elif "B" in op:
                fft_data, freqs = self.compute_fft(b, x, is_primary=True)
            elif "C" in op:
                fft_data, freqs = self.compute_fft(a, x, is_primary=False)
            elif "D" in op:
                fft_data, freqs = self.compute_fft(b, x, is_primary=False)
            elif "(A-B)" in op:
                fft_data, freqs = self.compute_fft(a - b, x, is_primary=True)
            elif "(C-D)" in op:
                fft_data, freqs = self.compute_fft(a - b, x, is_primary=False)
            self.fft_freqs = freqs
            return fft_data

        return a

    def apply_window(self, data, window_type):
        """Apply a window function to the data"""
        n = len(data)
        if window_type == "None (Rectangular)":
            return data
        elif window_type == "Hanning":
            window = np.hanning(n)
        elif window_type == "Hamming":
            window = np.hamming(n)
        elif window_type == "Blackman":
            window = np.blackman(n)
        elif window_type == "Bartlett":
            window = np.bartlett(n)
        else:
            return data
        return data * window

    def compute_fft(self, data, time, is_primary=True):
        """
        Compute a single‑sided FFT and return:
            1) the processed spectrum
            2) the corresponding frequency axis
        """
        # --- GUI parameters -------------------------------------------------
        if is_primary:
            window_type = self.fft_window1.get()
            display_type = self.fft_display1.get()
            padding_opt = self.fft_padding1.get()
        else:
            window_type = self.fft_window2.get()
            display_type = self.fft_display2.get()
            padding_opt = self.fft_padding2.get()

        # --- Window ---------------------------------------------------------
        y = np.asarray(data, dtype=float) - np.mean(data)
        n = len(y)
        win = {
            "None (Rectangular)": np.ones(n),
            "Hanning": np.hanning(n),
            "Hamming": np.hamming(n),
            "Blackman": np.blackman(n),
            "Bartlett": np.bartlett(n),
        }.get(window_type, np.ones(n))
        y_win = y * win
        win_gain = np.sum(win) / n  # amplitude loss

        # --- Zero‑padding ---------------------------------------------------
        pad_factor = {"None": 1, "2x": 2, "4x": 4, "8x": 8, "16x": 16}[padding_opt]
        n_fft = n * pad_factor  # length after padding

        # --- FFT ------------------------------------------------------------
        fs = 1.0 / np.mean(np.diff(time))  # sampling rate
        fft_vals = np.fft.rfft(y_win, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
        amp = 2.0 * np.abs(fft_vals) / (n * win_gain)

        if display_type == "Magnitude":
            spec = amp
        elif display_type == "Power (Magnitude²)":
            spec = amp ** 2
        else:  # dB
            spec = 20 * np.log10(np.maximum(amp, 1e-12))

        # keep a copy for the X‑axis swap that plot_graph performs
        self.fft_freqs = freqs
        return spec, freqs

    def detect_fft_peaks(self, freqs, spectrum,
                         top_n=5, threshold_ratio=0.1):
        """
        החזרת רשימת (freq, amp) של השיאים הגדולים בספקטרום.
        threshold_ratio  – אחוז מגובה השיא הראשי שמתחתיו מתעלמים.
        """
        amp = np.asarray(spectrum, float)
        # מציאת מקסימום כללי וסף
        max_amp = amp.max()
        if max_amp == 0:
            return []
        threshold = max_amp * threshold_ratio

        # מציאת מקס' מקומיים גסים: כל נקודה גדולה משכנותיה ומהסף
        gt_prev = amp[1:-1] > amp[:-2]
        gt_next = amp[1:-1] > amp[2:]
        candidates = np.where(gt_prev & gt_next)[0] + 1
        strong = candidates[amp[candidates] >= threshold]

        # אם אין, קח את השיא היחיד
        if len(strong) == 0:
            strong = [np.argmax(amp)]

        # מיון ולבחירת N הגדולים
        idx_sorted = strong[np.argsort(amp[strong])[::-1]]
        idx_top = idx_sorted[:top_n]

        return [(freqs[i], amp[i]) for i in idx_top]

    def _annotate_peaks(self, axis, peaks, title):
        """מצייר נקודות + תוויות ומציג חלונית Info; מתעלם אם axis==None."""
        import tkinter.messagebox as mb
        if axis is None or not peaks:
            return
        for f, a in peaks:
            axis.plot(f, a, "ro")
            axis.text(f, a, f"{f:.0f} Hz", color="red",
                      fontsize=8, ha="center", va="bottom")
        info = "\n".join(f"{f:.0f} Hz – {a:.3g}" for f, a in peaks)
        mb.showinfo(title, info)

    @staticmethod
    def suggest_cutoff_from_fft(fft_data, freqs, percent=0.95):
        """
        Suggest a cutoff frequency such that the given percentage (default 95%) of the spectral energy
        is below this frequency.

        Parameters:
            fft_data: 1D array of the FFT magnitude (or power) values (use only positive frequencies).
            freqs: 1D array of frequency values corresponding to fft_data.
            percent: Fraction of total energy to include (default 0.95 = 95%).

        Returns:
            cutoff_freq: The recommended cutoff frequency (Hz).
            suggestion_text: A string message with a practical recommendation.
        """
        # Compute the spectral energy (use squared magnitude for power)
        energy = np.abs(fft_data) ** 2
        cumsum_energy = np.cumsum(energy)
        total_energy = cumsum_energy[-1]
        norm_cumsum = cumsum_energy / total_energy
        idx = np.searchsorted(norm_cumsum, percent)
        cutoff_freq = freqs[idx]

        # Recommendation text for user (add ~10% margin and round to nearest 100Hz for safety)
        suggestion_text = (
            f"More than {int(percent * 100)}% of the signal energy is below {cutoff_freq:.0f} Hz. "
            f"Suggested cutoff: {int(round(cutoff_freq * 1.1, -2))} Hz."
        )
        return cutoff_freq, suggestion_text


class AdvancedExportDialog(ctk.CTkToplevel):
    def __init__(self, parent, channel_names, k_factor=None, on_export=None):
        super().__init__(parent)
        self.title("Advanced Export Options")
        self.geometry("600x500")
        self.transient(parent)
        self.grab_set()

        self.on_export = on_export
        self.k_factor = k_factor
        self.channel_rows = []

        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header_frame, text="Include", width=60, font=("Arial", 12, "bold")).pack(side="left")
        ctk.CTkLabel(header_frame, text="Original Name", width=120, anchor="w", font=("Arial", 12, "bold")).pack(
            side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Export Name (Rename)", width=150, anchor="w", font=("Arial", 12, "bold")).pack(
            side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Scale Factor (x)", width=100, anchor="w", font=("Arial", 12, "bold")).pack(
            side="left", padx=5)

        # Scrollable list of channels
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # K-Factor Helper Text
        if self.k_factor:
            k_info = ctk.CTkLabel(self, text=f"Detected K-Factor: {self.k_factor:.6g}", text_color="orange")
            k_info.pack(pady=(0, 5))

        # Generate rows
        for name in channel_names:
            row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)

            # 1. Checkbox (Include?)
            var_include = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(row_frame, text="", variable=var_include, width=40)
            chk.pack(side="left", padx=(10, 5))

            # 2. Original Name
            lbl = ctk.CTkLabel(row_frame, text=name, width=120, anchor="w")
            lbl.pack(side="left", padx=5)

            # 3. New Name Entry
            ent_name = ctk.CTkEntry(row_frame, width=150)
            ent_name.insert(0, name)  # Default to original name
            ent_name.pack(side="left", padx=5)

            # 4. Scale Factor Entry
            ent_scale = ctk.CTkEntry(row_frame, width=80)
            ent_scale.insert(0, "1.0")
            ent_scale.pack(side="left", padx=5)

            # Helper button to apply K to this specific row
            if self.k_factor:
                btn_k = ctk.CTkButton(row_frame, text="Use K", width=40,
                                      command=lambda e=ent_scale: self._set_scale(e, self.k_factor))
                btn_k.pack(side="left", padx=2)

            self.channel_rows.append({
                "orig_name": name,
                "var_include": var_include,
                "ent_name": ent_name,
                "ent_scale": ent_scale
            })

        # Footer Actions
        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.pack(fill="x", padx=10, pady=10)

        btn_cancel = ctk.CTkButton(footer_frame, text="Cancel", command=self.destroy, fg_color="gray")
        btn_cancel.pack(side="right", padx=10)

        btn_save = ctk.CTkButton(footer_frame, text="Export Selected", command=self._finish)
        btn_save.pack(side="right", padx=10)

    def _set_scale(self, entry_widget, value):
        entry_widget.delete(0, "end")
        entry_widget.insert(0, f"{value:.6g}")

    def _finish(self):
        export_config = []
        for row in self.channel_rows:
            if row["var_include"].get():
                try:
                    scale = float(row["ent_scale"].get())
                except ValueError:
                    scale = 1.0

                export_config.append({
                    "source": row["orig_name"],
                    "target_name": row["ent_name"].get(),
                    "scale": scale
                })

        if self.on_export:
            self.on_export(export_config)
        self.destroy()

if __name__ == "__main__":
    DataAnalyzerApp()