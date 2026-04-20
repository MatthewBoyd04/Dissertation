import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import os
import sys
import subprocess
import threading
import shutil
import re
import time

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _PLOTS_AVAILABLE = True
except Exception as _plot_err:
    _PLOTS_AVAILABLE = False
    # Write the real error and interpreter path to a log file so it is visible
    import traceback
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_import_error.log"), "w") as _f:
        _f.write(f"interpreter: {sys.executable}\n")
        _f.write(traceback.format_exc())

# Renderer lives one level up (repo root), next to UI/
_UI_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_UI_DIR)
sys.path.insert(0, _REPO_ROOT)
try:
    from renderer import GridRenderer
    _RENDERER_AVAILABLE = True
except ImportError:
    _RENDERER_AVAILABLE = False

# ── ANSI colour parsing ────────────────────────────────────────────────────────
_ANSI_TAG = {
    "31": "ansi_red",
    "32": "ansi_green",
    "33": "ansi_yellow",
    "34": "ansi_blue",
    "36": "ansi_cyan",
    "37": "ansi_white",
}
_ANSI_SPLIT = re.compile(r"(\x1b\[\d+m)")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "training_config.json")

# ── Default reward weights ─────────────────────────────────────────────────────
IPPO_DEFAULT_WEIGHTS = {
    "tileDiscovered": 0.5,
    "rewardFound":    200.0,
    "HazardHit":      -100.0,
    "Steps":          -0.1,
    "approachReward": 5.0,
}
MAPPO_DEFAULT_WEIGHTS = {
    "tileDiscovered": 0.5,
    "rewardFound":    200.0,
    "HazardHit":      -100.0,
    "Steps":          -0.1,
    "approachReward": 5.0,
}
MAPPO_ONLY_KEYS = set()  # No MAPPO-specific reward keys — both algorithms use identical weights
WEIGHT_LABELS = {
    "tileDiscovered": "Tile Discovered",
    "rewardFound":    "Reward Found",
    "HazardHit":      "Hazard Hit",
    "Steps":          "Step Cost",
    "approachReward": "Approach Reward",
}

# ── Plot spec: (column, colour, y-label) ──────────────────────────────────────
_PLOT_SPEC = [
    ("Reward Found %",   "Reward Found %",   "#42a5f5", "%"),
    ("Avg Steps",        "Avg Steps",        "#ffa726", "Steps"),
    ("Avg Tiles",        "Avg Tiles",        "#66bb6a", "Tiles"),
    ("Avg Tiles Per Step","Avg Tiles/Step",  "#ef5350", "Tiles/Step"),
    ("Avg Steps to Reward","Steps to Reward","#ab47bc", "Steps"),
    ("Avg Score",        "Avg Score",        "#8d6e63", "Score"),
]
_MA_WINDOW  = 10
_BG         = "#1e1e1e"
_AX_BG      = "#252526"
_GRID_COL   = "#3a3a3a"
_TEXT_COL   = "#c0c0c0"
_MAP_NAMES  = ["15x15", "30x30", "45x45"]


class TrainingUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAR Training Control Panel")
        self.configure(bg="#2b2b2b")
        self.state("zoomed")

        self._process       = None
        self._reader_thread = None
        self._auto_refresh  = False
        self._timer_running = False
        self._elapsed_base  = 0.0   # seconds stored in progress file
        self._session_start = None  # time.time() when current session started

        # Simulation renderer state
        self._render_polling_active = False
        self._render_frames: list | None = None
        self._current_frame_idx: int = 0
        self._discovered_state: list | None = None
        self._grid_renderer = None   # set in _build_plots after canvas is created

        self._style_notebook()
        self._build_ui()
        self._on_algorithm_change()
        self._refresh_plots()
        self._load_timer_display()

    # ── Notebook dark style ────────────────────────────────────────────────────
    def _style_notebook(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",        background="#2b2b2b", borderwidth=0)
        s.configure("TNotebook.Tab",    background="#3c3c3c", foreground="#a0a0a0",
                                        padding=[14, 6], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", "#555555")],
              foreground=[("selected", "#ffffff")])
        s.configure("TPanedwindow",     background="#2b2b2b")

    # ── Top-level layout ───────────────────────────────────────────────────────
    def _build_ui(self):
        pane = tk.PanedWindow(self, orient="horizontal",
                              bg="#444444", sashwidth=5, sashrelief="flat")
        pane.pack(fill="both", expand=True)

        left = tk.Frame(pane, bg="#2b2b2b", width=400)
        left.pack_propagate(False)
        pane.add(left, minsize=360)

        right = tk.Frame(pane, bg="#2b2b2b")
        pane.add(right, stretch="always")

        self._build_controls(left)
        self._build_plots(right)

    # ── Left panel: controls ───────────────────────────────────────────────────
    def _build_controls(self, parent):
        parent.columnconfigure(0, weight=1)

        # Algorithm
        af = self._section(parent, "Algorithm", 0)
        self._algorithm = tk.StringVar(value="ippo_live")
        for text, val in [("IPPO  (Live Teammates)", "ippo_live"),
                           ("IPPO  (Isolated)",       "ippo_isolated"),
                           ("MAPPO",                 "mappo")]:
            tk.Radiobutton(af, text=text, variable=self._algorithm, value=val,
                           bg="#3c3c3c", fg="#e0e0e0", selectcolor="#555555",
                           activebackground="#3c3c3c", activeforeground="#ffffff",
                           font=("Segoe UI", 10),
                           command=self._on_algorithm_change).pack(anchor="w", padx=12, pady=2)

        # Drones
        df = self._section(parent, "Number of Drones", 1)
        row = tk.Frame(df, bg="#3c3c3c")
        row.pack(fill="x", padx=12, pady=4)
        self._num_drones = tk.IntVar(value=4)
        tk.Label(row, textvariable=self._num_drones, width=3,
                 bg="#3c3c3c", fg="#f0c040",
                 font=("Segoe UI", 12, "bold")).pack(side="right", padx=(4, 0))
        tk.Scale(row, from_=1, to=8, orient="horizontal",
                 variable=self._num_drones, showvalue=False,
                 bg="#3c3c3c", fg="#e0e0e0", troughcolor="#555555",
                 highlightthickness=0, sliderlength=20).pack(side="left", fill="x", expand=True)

        # Map mode
        mf = self._section(parent, "Map Mode", 2)
        self._map_mode = tk.StringVar(value="safe")
        mr = tk.Frame(mf, bg="#3c3c3c")
        mr.pack(fill="x", padx=12, pady=4)
        for text, val in [("Safe", "safe"), ("Unsafe", "unsafe"), ("Multiple Rewards", "multiple_rewards")]:
            tk.Radiobutton(mr, text=text, variable=self._map_mode, value=val,
                           bg="#3c3c3c", fg="#e0e0e0", selectcolor="#555555",
                           activebackground="#3c3c3c", activeforeground="#ffffff",
                           font=("Segoe UI", 10)).pack(side="left", padx=8)

        # Vision Range
        vf = self._section(parent, "Vision Range", 3)
        vrow = tk.Frame(vf, bg="#3c3c3c")
        vrow.pack(fill="x", padx=12, pady=4)
        self._vision_range = tk.IntVar(value=3)
        tk.Label(vrow, textvariable=self._vision_range, width=3,
                 bg="#3c3c3c", fg="#f0c040",
                 font=("Segoe UI", 12, "bold")).pack(side="right", padx=(4, 0))
        tk.Scale(vrow, from_=1, to=8, orient="horizontal",
                 variable=self._vision_range, showvalue=False,
                 bg="#3c3c3c", fg="#e0e0e0", troughcolor="#555555",
                 highlightthickness=0, sliderlength=20).pack(side="left", fill="x", expand=True)

        # Reward weights
        wf = self._section(parent, "Reward Weights", 4)
        self._weight_vars = {}
        self._weight_rows = {}
        all_keys = list(IPPO_DEFAULT_WEIGHTS.keys()) + [k for k in MAPPO_DEFAULT_WEIGHTS if k in MAPPO_ONLY_KEYS]
        for key in all_keys:
            default = MAPPO_DEFAULT_WEIGHTS.get(key, IPPO_DEFAULT_WEIGHTS.get(key, 0.0))
            var = tk.StringVar(value=str(default))
            self._weight_vars[key] = var
            wr = tk.Frame(wf, bg="#3c3c3c")
            wr.pack(fill="x", padx=12, pady=2)
            tk.Label(wr, text=WEIGHT_LABELS[key], width=30, anchor="w",
                     bg="#3c3c3c", fg="#c0c0c0",
                     font=("Segoe UI", 9)).pack(side="left")
            tk.Entry(wr, textvariable=var, width=10,
                     bg="#555555", fg="#ffffff", insertbackground="#ffffff",
                     relief="flat", font=("Segoe UI", 9)).pack(side="left", padx=4)
            self._weight_rows[key] = wr

        # Rendering
        rf = self._section(parent, "Rendering", 5)
        rrow = tk.Frame(rf, bg="#3c3c3c")
        rrow.pack(fill="x", padx=12, pady=4)
        self._render_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(rrow, text="Enable rendering during training",
                       variable=self._render_enabled,
                       bg="#3c3c3c", fg="#e0e0e0", selectcolor="#555555",
                       activebackground="#3c3c3c", activeforeground="#ffffff",
                       font=("Segoe UI", 10),
                       command=self._on_render_toggle).pack(side="left", padx=4)
        frow = tk.Frame(rf, bg="#3c3c3c")
        frow.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(frow, text="Every", bg="#3c3c3c", fg="#c0c0c0",
                 font=("Segoe UI", 9)).pack(side="left")
        self._render_every = tk.IntVar(value=50)
        self._render_every_spin = tk.Spinbox(frow, from_=1, to=1000,
            textvariable=self._render_every, width=6,
            bg="#555555", fg="#ffffff", insertbackground="#ffffff",
            disabledbackground="#444444", disabledforeground="#666666",
            relief="flat", font=("Segoe UI", 9), state="disabled")
        self._render_every_spin.pack(side="left", padx=4)
        tk.Label(frow, text="episodes", bg="#3c3c3c", fg="#c0c0c0",
                 font=("Segoe UI", 9)).pack(side="left")

        # Buttons
        bf = tk.Frame(parent, bg="#2b2b2b")
        bf.grid(row=6, column=0, padx=10, pady=(4, 4), sticky="ew")
        self._start_btn = tk.Button(bf, text="▶  Start Training",
                                    bg="#4caf50", fg="white", activebackground="#388e3c",
                                    font=("Segoe UI", 10, "bold"), relief="flat", padx=12, pady=5,
                                    command=self._start_training)
        self._start_btn.pack(side="left", padx=(0, 6))
        self._stop_btn = tk.Button(bf, text="■  Stop",
                                   bg="#f44336", fg="white", activebackground="#c62828",
                                   font=("Segoe UI", 10, "bold"), relief="flat", padx=12, pady=5,
                                   state="disabled", command=self._stop_training)
        self._stop_btn.pack(side="left", padx=(0, 6))
        self._wipe_btn = tk.Button(bf, text="🗑  Wipe Data",
                                   bg="#795548", fg="white", activebackground="#4e342e",
                                   font=("Segoe UI", 10, "bold"), relief="flat", padx=12, pady=5,
                                   command=self._wipe_training_data)
        self._wipe_btn.pack(side="left", padx=(0, 6))
        self._save_btn = tk.Button(bf, text="💾  Save Training Data",
                                   bg="#1565c0", fg="white", activebackground="#0d47a1",
                                   font=("Segoe UI", 10, "bold"), relief="flat", padx=12, pady=5,
                                   command=self._save_training_data)
        self._save_btn.pack(side="left")

        # Timer
        tf = tk.Frame(parent, bg="#2b2b2b")
        tf.grid(row=7, column=0, padx=10, pady=(0, 4), sticky="ew")
        tk.Label(tf, text="Time Elapsed:", bg="#2b2b2b", fg="#a0a0a0",
                 font=("Segoe UI", 9)).pack(side="left")
        self._timer_var = tk.StringVar(value="0d 00:00:00")
        tk.Label(tf, textvariable=self._timer_var, bg="#2b2b2b", fg="#f0c040",
                 font=("Consolas", 10, "bold")).pack(side="left", padx=(6, 0))

        # Log
        lf = self._section(parent, "Training Log", 8, expand=True)
        self._log = scrolledtext.ScrolledText(
            lf, bg=_BG, fg="#d4d4d4", insertbackground="#d4d4d4",
            font=("Consolas", 9), relief="flat", state="disabled")
        self._log.pack(padx=8, pady=(4, 8), fill="both", expand=True)
        self._log.tag_configure("ansi_green",  foreground="#66bb6a")
        self._log.tag_configure("ansi_yellow", foreground="#ffa726")
        self._log.tag_configure("ansi_red",    foreground="#ef5350")
        self._log.tag_configure("ansi_blue",   foreground="#42a5f5")
        self._log.tag_configure("ansi_cyan",   foreground="#26c6da")
        self._log.tag_configure("ansi_white",  foreground="#d4d4d4")

    def _section(self, parent, title, row, expand=False):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          bg="#3c3c3c", fg="#90caf9",
                          font=("Segoe UI", 10, "bold"), relief="flat", bd=1)
        f.grid(row=row, column=0, padx=10, pady=4,
               sticky="nsew" if expand else "ew")
        if expand:
            parent.rowconfigure(row, weight=1)
        return f

    # ── Right panel: plots ─────────────────────────────────────────────────────
    def _build_plots(self, parent):
        if not _PLOTS_AVAILABLE:
            tk.Label(parent, text="matplotlib / pandas not installed.\nPlots unavailable.",
                     bg="#2b2b2b", fg="#808080", font=("Segoe UI", 12)).pack(expand=True)
            return

        top = tk.Frame(parent, bg="#2b2b2b")
        top.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(top, text="Live Training Plots", bg="#2b2b2b", fg="#90caf9",
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(top, text="⟳  Refresh", bg="#455a64", fg="white",
                  activebackground="#37474f", font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=10, pady=3,
                  command=self._refresh_plots).pack(side="right")

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        self._plot_figures  = {}
        self._plot_canvases = {}

        for map_name in _MAP_NAMES:
            tab = tk.Frame(nb, bg=_BG)
            nb.add(tab, text=f"  Map {map_name}  ")

            fig = Figure(facecolor=_BG)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(fill="both", expand=True)

            self._plot_figures[map_name]  = fig
            self._plot_canvases[map_name] = canvas

        # ── Simulation Render tab ──────────────────────────────────────────────
        sim_render_tab = tk.Frame(nb, bg=_BG)
        nb.add(sim_render_tab, text="  Simulation Render  ")

        # Controls row at the top
        sr_ctrl = tk.Frame(sim_render_tab, bg="#3c3c3c")
        sr_ctrl.pack(fill="x", padx=8, pady=(8, 4))

        self._sim_render_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(sr_ctrl, text="Enable simulation rendering",
                       variable=self._sim_render_enabled,
                       bg="#3c3c3c", fg="#e0e0e0", selectcolor="#555555",
                       activebackground="#3c3c3c", activeforeground="#ffffff",
                       font=("Segoe UI", 10),
                       command=self._on_sim_render_toggle).pack(side="left", padx=(8, 16))

        tk.Label(sr_ctrl, text="Render every", bg="#3c3c3c", fg="#c0c0c0",
                 font=("Segoe UI", 9)).pack(side="left")
        self._sim_render_every = tk.IntVar(value=10)
        self._sim_render_every_spin = tk.Spinbox(
            sr_ctrl, from_=1, to=999, textvariable=self._sim_render_every,
            width=6, bg="#555555", fg="#ffffff", insertbackground="#ffffff",
            disabledbackground="#444444", disabledforeground="#666666",
            relief="flat", font=("Segoe UI", 9), state="disabled")
        self._sim_render_every_spin.pack(side="left", padx=4)
        tk.Label(sr_ctrl, text="simulation episodes", bg="#3c3c3c", fg="#c0c0c0",
                 font=("Segoe UI", 9)).pack(side="left")

        # Canvas — fills remaining space
        self._render_canvas = tk.Canvas(sim_render_tab, bg="#1a1a1a",
                                        highlightthickness=0)
        self._render_canvas.pack(fill="both", expand=True, padx=8, pady=(4, 4))

        # Status label at the bottom
        self._render_status_label = tk.Label(
            sim_render_tab,
            text="Waiting for training to start...",
            bg=_BG, fg="#808080", font=("Segoe UI", 9))
        self._render_status_label.pack(pady=(0, 6))

        # Instantiate renderer (attaches to canvas, no drawing yet)
        if _RENDERER_AVAILABLE:
            self._grid_renderer = GridRenderer(self._render_canvas)

    # ── Plot rendering ─────────────────────────────────────────────────────────
    def _refresh_plots(self):
        if not _PLOTS_AVAILABLE:
            return
        algo   = self._algorithm.get()
        folder = "IPPO" if algo in ("ippo_live", "ippo_isolated") else "MAPPO"
        for map_name in _MAP_NAMES:
            csv = os.path.join(ROOT_DIR, folder, f"map_{map_name}_analysis_Results.csv")
            self._draw_map_plot(map_name, csv)
        if self._auto_refresh:
            self.after(3000, self._refresh_plots)

    def _draw_map_plot(self, map_name, csv_path):
        fig = self._plot_figures[map_name]
        fig.clear()

        if not os.path.exists(csv_path):
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            ax.text(0.5, 0.5, "No data yet — training has not started",
                    ha="center", va="center", color="#606060",
                    fontsize=13, transform=ax.transAxes)
            ax.axis("off")
            self._plot_canvases[map_name].draw()
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return

        # Last column is either 'Avg Score' (safe mode) or 'Avg Drones Terminated' (hazard mode)
        if 'Avg Drones Terminated' in df.columns:
            last_entry = ("Avg Drones Terminated", "Avg Drones Terminated", "#e53935", "Drones")
        else:
            last_entry = ("Avg Score", "Avg Score", "#8d6e63", "Score")
        plot_spec = _PLOT_SPEC[:-1] + [last_entry]

        for col, _, _, _ in plot_spec:
            ma_col = f"MA_{col}"
            df[ma_col] = df[col].rolling(window=_MA_WINDOW, min_periods=1).mean()

        axes = fig.subplots(2, 3)
        fig.subplots_adjust(hspace=0.42, wspace=0.32, left=0.07, right=0.97,
                            top=0.92, bottom=0.09)
        fig.suptitle(f"Map {map_name}", color=_TEXT_COL,
                     fontsize=11, fontweight="bold")

        for idx, (col, title, colour, ylabel) in enumerate(plot_spec):
            ax = axes[idx // 3][idx % 3]
            ax.set_facecolor(_AX_BG)
            ax.tick_params(colors=_TEXT_COL, labelsize=7)
            ax.xaxis.label.set_color(_TEXT_COL)
            ax.yaxis.label.set_color(_TEXT_COL)
            ax.title.set_color(_TEXT_COL)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.grid(True, color=_GRID_COL, linewidth=0.6)

            ma_col = f"MA_{col}"
            ax.plot(df["TimeSteps"], df[col],
                    color=colour, alpha=0.25, linewidth=1)
            ax.plot(df["TimeSteps"], df[ma_col],
                    color=colour, linewidth=2,
                    label=f"MA({_MA_WINDOW})")
            ax.set_title(title, fontsize=9, pad=4)
            ax.set_xlabel("Timesteps", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.legend(fontsize=7, facecolor="#333333",
                      labelcolor=_TEXT_COL, edgecolor="#444444")

        self._plot_canvases[map_name].draw()

    # ── Dynamic behaviour ──────────────────────────────────────────────────────
    def _on_render_toggle(self):
        state = "normal" if self._render_enabled.get() else "disabled"
        self._render_every_spin.config(state=state)

    def _on_sim_render_toggle(self):
        state = "normal" if self._sim_render_enabled.get() else "disabled"
        self._sim_render_every_spin.config(state=state)

    # ── Simulation render polling ──────────────────────────────────────────────
    def _start_render_polling(self):
        if not hasattr(self, "_render_status_label"):
            return
        # Remove any stale queue file from a previous run
        stale = os.path.join(ROOT_DIR, "render_queue.json")
        try:
            os.remove(stale)
        except OSError:
            pass
        self._render_polling_active = True
        self._render_status_label.config(text="Waiting for simulation episode...")
        self._poll_render_queue()

    def _stop_render_polling(self):
        self._render_polling_active = False
        if hasattr(self, "_render_status_label"):
            self._render_status_label.config(text="Training stopped.")

    def _poll_render_queue(self):
        if not self._render_polling_active:
            return
        path = os.path.join(ROOT_DIR, "render_queue.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                os.remove(path)
                self._start_episode_playback(data)
                return  # playback chain resumes polling when done
            except (json.JSONDecodeError, OSError):
                pass  # partial write — retry next poll
        self.after(500, self._poll_render_queue)

    def _start_episode_playback(self, data: dict):
        gs = data["grid_size"]
        self._discovered_state = [[0] * gs for _ in range(gs)]
        for r, c in data.get("initial_discovered", []):
            self._discovered_state[r][c] = 1
        self._render_frames = data["frames"]
        self._current_frame_idx = 0
        if self._grid_renderer:
            self._grid_renderer.setup_grid(data["grid"])
        total = len(self._render_frames) - 1
        self._render_status_label.config(text=f"Playing step 0/{total}")
        self._play_next_frame()

    def _play_next_frame(self):
        if not self._render_frames or self._current_frame_idx >= len(self._render_frames):
            self._render_status_label.config(text="Episode complete. Waiting for next...")
            self.after(500, self._poll_render_queue)
            return
        frame = self._render_frames[self._current_frame_idx]
        for r, c in frame.get("new_discovered", []):
            self._discovered_state[r][c] = 1
        if self._grid_renderer:
            self._grid_renderer.render_frame(frame["agents"], self._discovered_state)
        total = len(self._render_frames) - 1
        self._render_status_label.config(text=f"Playing step {frame['step']}/{total}")
        self._current_frame_idx += 1
        self.after(100, self._play_next_frame)  # 10 fps

    def _on_algorithm_change(self):
        is_mappo = self._algorithm.get() == "mappo"
        for key in MAPPO_ONLY_KEYS:
            row = self._weight_rows[key]
            if is_mappo:
                row.pack(fill="x", padx=12, pady=2)
            else:
                row.pack_forget()
        defaults = MAPPO_DEFAULT_WEIGHTS if is_mappo else IPPO_DEFAULT_WEIGHTS
        for key, var in self._weight_vars.items():
            if key in defaults:
                var.set(str(defaults[key]))
        self._refresh_plots()
        if not self._timer_running:
            self._load_timer_display()

    # ── Training control ───────────────────────────────────────────────────────
    def _start_training(self):
        weights = {}
        for key, var in self._weight_vars.items():
            if key in MAPPO_ONLY_KEYS and self._algorithm.get() != "mappo":
                continue
            try:
                weights[key] = float(var.get())
            except ValueError:
                messagebox.showerror("Invalid Input",
                    f"Reward weight '{WEIGHT_LABELS[key]}' must be a number.")
                return

        algorithm = self._algorithm.get()
        config = {
            "algorithm":        algorithm,
            "num_drones":       self._num_drones.get(),
            "vision_range":     self._vision_range.get(),
            "map_mode":         self._map_mode.get(),
            "render_every":     self._render_every.get() if self._render_enabled.get() else None,
            "sim_render_every": self._sim_render_every.get() if self._sim_render_enabled.get() else 0,
            "reward_weights":   weights,
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

        main_script = os.path.join(
            ROOT_DIR,
            "IPPO" if algorithm in ("ippo_live", "ippo_isolated") else "MAPPO",
            "Main_IPPO.py" if algorithm in ("ippo_live", "ippo_isolated") else "Main_MAPPO.py",
        )

        self._log_clear()
        self._log_append(f"Config written to {CONFIG_PATH}\n")
        self._log_append(f"Launching: {main_script}\n")
        self._log_append("-" * 60 + "\n")

        self._process = subprocess.Popen(
            [sys.executable, main_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=ROOT_DIR,
            text=True,
            bufsize=1,
        )
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._auto_refresh = True
        self.after(3000, self._refresh_plots)
        self._start_timer()

        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        self._start_render_polling()

    def _stop_training(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log_append("\n[Training stopped by user]\n")
        self._on_training_finished()

    def _read_output(self):
        try:
            for line in self._process.stdout:
                self.after(0, self._log_append, line)
        except Exception:
            pass
        finally:
            self._process.wait()
            self.after(0, self._on_training_finished)

    def _on_training_finished(self):
        self._auto_refresh  = False
        self._timer_running = False
        self._stop_render_polling()
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        if self._process and self._process.returncode not in (None, 0, -15):
            self._log_append(f"\n[Process exited with code {self._process.returncode}]\n")
        else:
            self._log_append("\n[Training complete]\n")
        self._refresh_plots()

    # ── Timer helpers ──────────────────────────────────────────────────────────
    def _load_timer_display(self):
        """Show persisted elapsed time on startup (static, no tick)."""
        algorithm = self._algorithm.get()
        folder = "IPPO" if algorithm in ("ippo_live", "ippo_isolated") else "MAPPO"
        progress_path = os.path.join(ROOT_DIR, folder, "training_progress.json")
        elapsed = 0.0
        if os.path.exists(progress_path):
            try:
                with open(progress_path) as f:
                    elapsed = json.load(f).get("elapsed_seconds", 0.0)
            except Exception:
                pass
        total = int(elapsed)
        d, rem = divmod(total, 86400)
        h, rem = divmod(rem, 3600)
        m, s   = divmod(rem, 60)
        self._timer_var.set(f"{d}d {h:02}:{m:02}:{s:02}")

    def _start_timer(self):
        algorithm = self._algorithm.get()
        folder = "IPPO" if algorithm in ("ippo_live", "ippo_isolated") else "MAPPO"
        progress_path = os.path.join(ROOT_DIR, folder, "training_progress.json")
        self._elapsed_base = 0.0
        if os.path.exists(progress_path):
            try:
                with open(progress_path) as f:
                    self._elapsed_base = json.load(f).get("elapsed_seconds", 0.0)
            except Exception:
                pass
        self._session_start = time.time()
        self._timer_running = True
        self._tick_timer()

    def _tick_timer(self):
        if not self._timer_running:
            return
        total = int(self._elapsed_base + (time.time() - self._session_start))
        d, rem = divmod(total, 86400)
        h, rem = divmod(rem, 3600)
        m, s   = divmod(rem, 60)
        self._timer_var.set(f"{d}d {h:02}:{m:02}:{s:02}")
        self.after(1000, self._tick_timer)

    # ── Wipe training data ─────────────────────────────────────────────────────
    def _wipe_training_data(self):
        if self._process and self._process.poll() is None:
            messagebox.showwarning("Training Running", "Stop training before wiping data.")
            return

        algorithm = self._algorithm.get()
        target = "IPPO" if algorithm in ("ippo_live", "ippo_isolated") else "MAPPO"
        confirmed = messagebox.askokcancel(
            "Wipe Training Data",
            f"This will permanently delete all saved {target} models, CSV results,\n"
            f"and training progress. Tensorboard logs will be archived.\n\nContinue?",
            icon="warning",
        )
        if not confirmed:
            return

        self._log_clear()
        self._log_append(f"Wiping {target} training data...\n" + "-" * 60 + "\n")
        if target == "IPPO":
            self._wipe_ippo()
        else:
            self._wipe_mappo()
        self._log_append("-" * 60 + "\nDone. Ready for fresh training.\n")
        self._refresh_plots()

    def _wipe_ippo(self):
        ippo_dir = os.path.join(ROOT_DIR, "IPPO")
        files = ["training_progress.json",
                 "map_15x15_analysis_Results.csv",
                 "map_30x30_analysis_Results.csv",
                 "map_45x45_analysis_Results.csv"]
        for name in os.listdir(ippo_dir):
            if name.endswith("_ppo_model.zip"):
                files.append(name)
        for filename in files:
            path = os.path.join(ippo_dir, filename)
            if os.path.exists(path):
                os.remove(path)
                self._log_append(f"  Deleted: {filename}\n")
            else:
                self._log_append(f"  Skipped: {filename} (not found)\n")
        archive_base = os.path.join(ippo_dir, "__Drone_Data_Archive", "old_tensorboard_logs")
        os.makedirs(archive_base, exist_ok=True)
        for name in os.listdir(ippo_dir):
            if name.endswith("_tensorboard") and os.path.isdir(os.path.join(ippo_dir, name)):
                src = os.path.join(ippo_dir, name)
                run_id = len([d for d in os.listdir(archive_base) if d.startswith("run_")]) + 1
                dst = os.path.join(archive_base, f"run_{run_id}_{name}")
                shutil.move(src, dst)
                self._log_append(f"  Archived: {name} → run_{run_id}_{name}\n")

    def _wipe_mappo(self):
        mappo_dir = os.path.join(ROOT_DIR, "MAPPO")
        files = ["shared_mappo_model.zip", "vecnormalize.pkl",
                 "training_progress.json",
                 "map_15x15_analysis_Results.csv",
                 "map_30x30_analysis_Results.csv",
                 "map_45x45_analysis_Results.csv"]
        for filename in files:
            path = os.path.join(mappo_dir, filename)
            if os.path.exists(path):
                os.remove(path)
                self._log_append(f"  Deleted: {filename}\n")
            else:
                self._log_append(f"  Skipped: {filename} (not found)\n")
        tb_dir = os.path.join(mappo_dir, "mappo_tensorboard")
        if os.path.exists(tb_dir):
            archive_base = os.path.join(mappo_dir, "_droneDataArchive", "old_tensorboard_logs")
            os.makedirs(archive_base, exist_ok=True)
            run_id = len([d for d in os.listdir(archive_base) if d.startswith("run_")]) + 1
            dst = os.path.join(archive_base, f"run_{run_id}")
            shutil.move(tb_dir, dst)
            self._log_append(f"  Archived: mappo_tensorboard → _droneDataArchive/run_{run_id}\n")

    # ── Save training data ─────────────────────────────────────────────────────
    def _save_training_data(self):
        save_dir = filedialog.askdirectory(title="Choose Save Location for Training Data")
        if not save_dir:
            return

        algorithm = self._algorithm.get()
        folder    = "IPPO" if algorithm in ("ippo_live", "ippo_isolated") else "MAPPO"
        src_dir   = os.path.join(ROOT_DIR, folder)

        results_dir = os.path.join(save_dir, "results")
        models_dir  = os.path.join(save_dir, "models")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir,  exist_ok=True)

        self._log_clear()
        self._log_append(f"Saving {folder} training data to:\n  {save_dir}\n" + "-" * 60 + "\n")

        # CSVs
        for map_name in _MAP_NAMES:
            csv_name = f"map_{map_name}_analysis_Results.csv"
            src = os.path.join(src_dir, csv_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(results_dir, csv_name))
                self._log_append(f"  Saved:   {csv_name}\n")
            else:
                self._log_append(f"  Skipped: {csv_name} (not found)\n")

        # Plot PNGs
        if _PLOTS_AVAILABLE:
            # Refresh plots so they are up to date before saving
            algo_folder = "IPPO" if algorithm in ("ippo_live", "ippo_isolated") else "MAPPO"
            for map_name in _MAP_NAMES:
                csv = os.path.join(ROOT_DIR, algo_folder, f"map_{map_name}_analysis_Results.csv")
                self._draw_map_plot(map_name, csv)
            for map_name in _MAP_NAMES:
                fig = self._plot_figures.get(map_name)
                if fig:
                    png_path = os.path.join(results_dir, f"map_{map_name}_plot.png")
                    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=_BG)
                    self._log_append(f"  Saved:   map_{map_name}_plot.png\n")

        # Model data
        if folder == "IPPO":
            for name in os.listdir(src_dir):
                if name.endswith("_ppo_model.zip"):
                    shutil.copy2(os.path.join(src_dir, name), os.path.join(models_dir, name))
                    self._log_append(f"  Saved:   {name}\n")
            for name in os.listdir(src_dir):
                if name.endswith("_tensorboard") and os.path.isdir(os.path.join(src_dir, name)):
                    dst = os.path.join(models_dir, name)
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(os.path.join(src_dir, name), dst)
                    self._log_append(f"  Saved:   {name}/\n")
        else:
            for name in ["shared_mappo_model.zip", "vecnormalize.pkl"]:
                src = os.path.join(src_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(models_dir, name))
                    self._log_append(f"  Saved:   {name}\n")
                else:
                    self._log_append(f"  Skipped: {name} (not found)\n")
            tb_dir = os.path.join(src_dir, "mappo_tensorboard")
            if os.path.exists(tb_dir):
                dst = os.path.join(models_dir, "mappo_tensorboard")
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(tb_dir, dst)
                self._log_append(f"  Saved:   mappo_tensorboard/\n")

        # Elapsed time
        progress_path = os.path.join(src_dir, "training_progress.json")
        elapsed = 0.0
        if os.path.exists(progress_path):
            try:
                with open(progress_path) as f:
                    elapsed = json.load(f).get("elapsed_seconds", 0.0)
            except Exception:
                pass
        if self._timer_running and self._session_start is not None:
            elapsed = self._elapsed_base + (time.time() - self._session_start)
        total = int(elapsed)
        d, rem = divmod(total, 86400)
        h, rem = divmod(rem, 3600)
        m, s   = divmod(rem, 60)
        time_str = f"{d}d {h:02}:{m:02}:{s:02}"

        # Reward weights
        weights = {}
        for key, var in self._weight_vars.items():
            if key in MAPPO_ONLY_KEYS and algorithm != "mappo":
                continue
            try:
                weights[key] = float(var.get())
            except ValueError:
                weights[key] = var.get()

        lines = [
            f"Training Summary — {folder}",
            "=" * 44,
            f"Algorithm  : {algorithm}",
            f"Map Mode   : {self._map_mode.get()}",
            f"Num Drones : {self._num_drones.get()}",
            f"Vision Range: {self._vision_range.get()}",
            f"Time Taken : {time_str}",
            "",
            "Reward Weights:",
        ]
        for key, val in weights.items():
            label = WEIGHT_LABELS.get(key, key)
            lines.append(f"  {label:<32} {val}")

        summary_path = os.path.join(save_dir, "training_summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self._log_append(f"  Saved:   training_summary.txt\n")

        self._log_append("-" * 60 + f"\nSave complete → {save_dir}\n")

    # ── Log helpers ────────────────────────────────────────────────────────────
    def _log_append(self, text: str):
        self._log.config(state="normal")
        current_tag = "ansi_white"
        for segment in _ANSI_SPLIT.split(text):
            if not segment:
                continue
            m = re.fullmatch(r"\x1b\[(\d+)m", segment)
            if m:
                current_tag = _ANSI_TAG.get(m.group(1), current_tag)
            else:
                self._log.insert("end", segment, current_tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _log_clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")


if __name__ == "__main__":
    app = TrainingUI()
    app.mainloop()
