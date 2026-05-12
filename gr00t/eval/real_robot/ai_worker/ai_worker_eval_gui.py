#!/usr/bin/env python3
"""
PyQt5 GUI for robot policy evaluation with episode management.

All robot eval logic lives in ai_worker_eval.py; this file is purely
GUI layout + threading.

Usage:
    python gr00t/eval/real_robot/ai_worker/ai_worker_eval_gui.py
"""

import sys
import os
import time
import argparse
import importlib.util

import numpy as np
import cv2
import matplotlib.cm as mpl_cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QSlider, QSizePolicy,
    QInputDialog, QCheckBox, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# ROS2 is optional when running --dummy.
try:
    import rclpy
    _ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    _ROS2_AVAILABLE = False

from gr00t.eval.real_robot.ai_worker.ai_worker_eval import (
    AiWorkerObsCollector,
    AiWorkerCommandSender,
    AiWorkerAdapter,
    DummyCollector,
    DummySender,
    DIMS_BY_KEY,
    concat_obs_rgb,
    run_eval_loop,
    parse_checkpoint_config,
)
from gr00t.policy.server_client import PolicyClient


# ---------------------------------------------------------------------------
# Camera display helpers
# ---------------------------------------------------------------------------

_CAM_ORDER = ("left_wrist_view", "ego_view", "right_wrist_view")
_CAM_DISPLAY_NAMES = {
    "left_wrist_view": "Left Wrist",
    "ego_view": "Head",
    "right_wrist_view": "Right Wrist",
}


def _build_labeled_camera_row(rgb_by_key, model_keys, target_h=240):
    """Horizontal strip of all 3 cameras with MODEL/VIEW labels. Gray placeholder for missing."""
    imgs = []
    for cam_key in _CAM_ORDER:
        is_model = cam_key in model_keys
        name = _CAM_DISPLAY_NAMES.get(cam_key, cam_key)

        if cam_key in (rgb_by_key or {}) and rgb_by_key[cam_key] is not None:
            img = rgb_by_key[cam_key].copy()
        else:
            img = np.zeros((target_h, target_h * 4 // 3, 3), dtype=np.uint8)
            img[:] = (25, 25, 25)

        h, w = img.shape[:2]
        if h != target_h:
            img = cv2.resize(img, (max(1, int(w * target_h / h)), target_h))

        border_color = (0, 180, 0) if is_model else (200, 0, 0)  # green / red (RGB)
        img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border_color)

        tag = "[Model Input]" if is_model else "[Display Only - Not Used by Model]"
        text = f"{name}  {tag}"
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 230, 0) if is_model else (255, 80, 80), 1, cv2.LINE_AA)
        imgs.append(img)

    if not imgs:
        return None
    target_row_h = max(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        if h != target_row_h:
            im = cv2.resize(im, (max(1, int(w * target_row_h / h)), target_row_h))
        resized.append(im)
    return cv2.hconcat(resized)


def _build_ref_camera_row(ref_frames, model_keys, target_h=200):
    """Horizontal strip of reference frames matching _CAM_ORDER layout."""
    imgs = []
    for cam_key in _CAM_ORDER:
        is_model = cam_key in model_keys
        name = _CAM_DISPLAY_NAMES.get(cam_key, cam_key)

        if cam_key in (ref_frames or {}) and ref_frames[cam_key] is not None:
            img = ref_frames[cam_key].copy()
        else:
            img = np.zeros((target_h, target_h * 4 // 3, 3), dtype=np.uint8)
            img[:] = (15, 15, 35)

        h, w = img.shape[:2]
        if h != target_h:
            img = cv2.resize(img, (max(1, int(w * target_h / h)), target_h))

        border_color = (0, 100, 0) if is_model else (140, 0, 0)  # dark green / dark red (RGB)
        img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border_color)

        text = f"REF: {name}"
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        imgs.append(img)

    if not imgs:
        return None
    target_row_h = max(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        if h != target_row_h:
            im = cv2.resize(im, (max(1, int(w * target_row_h / h)), target_row_h))
        resized.append(im)
    return cv2.hconcat(resized)


# ---------------------------------------------------------------------------
# Action history visualization
# ---------------------------------------------------------------------------

class ActionHistoryViz:
    """
    Renders a rolling action-history plot as a numpy uint8 RGB image.

    One subplot per action key; each dimension of a key gets its own colored line.
    Raw model output is always stored (masking only affects visual style, not data).
    """

    WINDOW = 300
    FIG_W_IN = 12
    ROW_H_IN = 1.8
    DPI = 90
    _MASKED_BG = "#330000"
    _NORMAL_BG = "#1a1a1a"
    _DIMS = {
        "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1,
        "head": 2, "lift": 1, "base": 3,
    }

    def __init__(self, action_keys):
        self._action_keys = list(action_keys)
        self._dims = {k: self._DIMS.get(k, 1) for k in self._action_keys}
        self._history = []        # [(global_step_start, {key: np.ndarray (K, D)})]
        self._global_step = 0
        self._masked_keys = set()
        self._colors = None
        self._fig = None
        self._canvas = None
        self._axes = []

    def _ensure_fig(self):
        """Create the matplotlib figure on first use (called from worker thread)."""
        if self._fig is not None or not self._action_keys:
            return
        self._colors = list(mpl_cm.tab10.colors)
        n = len(self._action_keys)
        self._fig = Figure(figsize=(self.FIG_W_IN, self.ROW_H_IN * n), dpi=self.DPI)
        self._canvas = FigureCanvasAgg(self._fig)
        axes = self._fig.subplots(nrows=n)
        self._axes = [axes] if n == 1 else list(axes)
        self._fig.subplots_adjust(left=0.08, right=0.85, top=0.97, bottom=0.04, hspace=0.55)
        self._fig.patch.set_facecolor(self._NORMAL_BG)

    def update(self, action_seq, masked_keys=None):
        """Append raw model chunk, optionally update mask, return rendered RGB image."""
        if masked_keys is not None:
            self._masked_keys = set(masked_keys)
        if not action_seq:
            return None
        self._ensure_fig()
        if self._fig is None:
            return None

        K = len(action_seq)
        arrays = {}
        for key in self._action_keys:
            try:
                arrays[key] = np.stack(
                    [np.atleast_1d(np.asarray(s[key], dtype=np.float32)) for s in action_seq],
                    axis=0,
                )  # (K, D)
            except (KeyError, ValueError):
                arrays[key] = np.zeros((K, self._dims[key]), dtype=np.float32)

        self._history.append((self._global_step, arrays))
        self._global_step += K
        return self._render()

    def _render(self):
        total = self._global_step
        win_start = max(0, total - self.WINDOW)
        win_end = win_start + self.WINDOW
        last_idx = len(self._history) - 1

        for ax_idx, key in enumerate(self._action_keys):
            ax = self._axes[ax_idx]
            ax.cla()
            is_masked = key in self._masked_keys
            ax.set_facecolor(self._MASKED_BG if is_masked else self._NORMAL_BG)

            for c_idx, (g_start, arrays) in enumerate(self._history):
                chunk = arrays[key]          # (K, D)
                K = chunk.shape[0]
                if g_start + K <= win_start or g_start >= win_end:
                    continue
                is_last = (c_idx == last_idx)
                x = np.arange(g_start, g_start + K)

                if is_last:
                    ax.axvspan(g_start, g_start + K, alpha=0.20, color="#FFD700", zorder=0)
                if c_idx > 0:
                    ax.axvline(g_start, color="#888888", lw=0.8, alpha=0.65, zorder=1)

                alpha = 1.0 if is_last else 0.50
                lw = 2.0 if is_last else 1.2
                ls = "--" if is_masked else "-"
                for dim_i in range(self._dims[key]):
                    ax.plot(x, chunk[:, dim_i],
                            color=self._colors[dim_i % 10],
                            alpha=alpha, lw=lw, ls=ls, zorder=2)

            ax.set_xlim(win_start, win_end)
            ax.tick_params(labelsize=6, colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#555555")
            ax.spines["bottom"].set_color("#555555")

            label = f"{key}  [MASKED]" if is_masked else key
            ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=56, va="center",
                          color="#ff6666" if is_masked else "white")
            ax.yaxis.set_label_position("right")

        self._canvas.draw()
        w, h = self._canvas.get_width_height()
        buf = self._canvas.buffer_rgba()
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    def close(self):
        self._fig = None
        self._canvas = None


# ---------------------------------------------------------------------------
# Import init_pose_from_data by file path
# ---------------------------------------------------------------------------

def _load_init_pose_module():
    path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",
        "help_scripts", "deployment_process", "init_pose_from_data.py",
    ))
    spec = importlib.util.spec_from_file_location("init_pose_from_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Ros2SpinThread -- keeps ROS2 callbacks alive between episodes
# ---------------------------------------------------------------------------

class Ros2SpinThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, collector, sender, video_keys, ref_frames_getter=None):
        super().__init__()
        self._collector = collector
        self._sender = sender
        self._model_video_keys = set(video_keys or [])
        self._ref_frames_getter = ref_frames_getter  # callable returning {cam_key: rgb}
        self._stop = False
        self._paused = False
        self._init_pose_mod = None

    def run(self):
        last_frame_time = 0.0
        while not self._stop and rclpy.ok():
            if self._paused:
                time.sleep(0.01)
                continue

            rclpy.spin_once(self._collector, timeout_sec=0.01)
            rclpy.spin_once(self._sender, timeout_sec=0.0)

            now = time.time()
            if now - last_frame_time > 0.1 and self._collector.latest_rgb_by_key:
                live_by_key = self._collector.latest_rgb_by_key
                live_row = _build_labeled_camera_row(live_by_key, self._model_video_keys)
                if live_row is None:
                    continue

                ref_frames = self._ref_frames_getter() if self._ref_frames_getter else {}
                if ref_frames:
                    if self._init_pose_mod is None:
                        self._init_pose_mod = _load_init_pose_module()
                    lw = live_row.shape[1]
                    rows = [live_row]
                    for k in ("ego_view",):
                        if k not in ref_frames or k not in live_by_key:
                            continue
                        cmp_row = self._init_pose_mod.build_comparison_grid(
                            ref_frames[k], live_by_key[k]
                        )
                        if cmp_row is not None:
                            ch, cw = cmp_row.shape[:2]
                            if cw != lw:
                                cmp_row = cv2.resize(cmp_row, (lw, max(1, int(ch * lw / cw))))
                            rows.append(cmp_row)
                    if len(rows) > 1:
                        self.frame_ready.emit(np.vstack(rows))
                        last_frame_time = now
                        continue

                self.frame_ready.emit(live_row)
                last_frame_time = now

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stop = True


# ---------------------------------------------------------------------------
# EpisodeWorker -- runs the eval loop in a background thread via run_eval_loop
# ---------------------------------------------------------------------------

class EpisodeWorker(QThread):
    frame_ready = pyqtSignal(object)
    status_update = pyqtSignal(str)
    episode_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    action_viz_ready = pyqtSignal(object)   # numpy uint8 RGB of action history plot

    def __init__(
        self, collector, sender, adapter, horizon, lang, exec_sec=1.0, use_ros=True,
        model_video_keys=None, action_keys=None, get_action_mask=None,
    ):
        super().__init__()
        self._collector = collector
        self._sender = sender
        self._adapter = adapter
        self._horizon = horizon
        self._lang = lang
        self._exec_sec = exec_sec
        self._use_ros = use_ros
        self._model_video_keys = set(model_video_keys or [])
        self._stop_event = __import__("threading").Event()
        self._viz = ActionHistoryViz(action_keys) if action_keys else None
        self._get_action_mask = get_action_mask

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, val):
        self._horizon = val

    def stop(self):
        self._stop_event.set()

    def _on_action(self, action_seq):
        if self._viz is None:
            return
        try:
            masked = self._get_action_mask() if self._get_action_mask else set()
            rgb = self._viz.update(action_seq, masked_keys=masked)
            if rgb is not None:
                self.action_viz_ready.emit(rgb)
        except Exception:
            pass  # viz errors must not abort the episode

    def run(self):
        try:
            run_eval_loop(
                collector=self._collector,
                sender=self._sender,
                adapter=self._adapter,
                lang_instruction=self._lang,
                action_horizon=self._horizon,
                exec_sec=self._exec_sec,
                use_ros=self._use_ros,
                should_stop=self._stop_event.is_set,
                on_obs=lambda obs: self.frame_ready.emit(
                    _build_labeled_camera_row(
                        getattr(self._collector, "latest_rgb_by_key", None) or obs.get("rgb", {}),
                        self._model_video_keys,
                    )
                ),
                on_status=lambda msg: self.status_update.emit(msg),
                get_action_horizon=lambda: self._horizon,
                on_action=self._on_action,
                get_action_mask=self._get_action_mask,
            )
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self._viz is not None:
                self._viz.close()
            self.episode_finished.emit()


# ---------------------------------------------------------------------------
# InitPoseWorker -- resets robot to init pose after episode
# ---------------------------------------------------------------------------

class InitPoseWorker(QThread):
    status_update = pyqtSignal(str)
    lang_loaded = pyqtSignal(str)
    ref_frames_ready = pyqtSignal(object)  # dict {cam_key: rgb_ndarray}
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, sender, collector, dataset_path, video_keys=None, episode_idx=None):
        super().__init__()
        self._sender = sender
        self._collector = collector
        self._dataset_path = dataset_path
        self._video_keys = video_keys or []
        self._episode_idx = episode_idx

    def run(self):
        try:
            mod = _load_init_pose_module()

            self.status_update.emit("Loading init pose from dataset...")
            positions, ref_ep, lang = mod.compute_init_pose(
                self._dataset_path, episode_idx=self._episode_idx
            )
            if lang:
                self.lang_loaded.emit(lang)

            # Extract reference frames for comparison
            self.status_update.emit(f"Extracting ref frames for {self._video_keys}...")
            if self._video_keys:
                ref_frames = mod.extract_reference_frames(
                    self._dataset_path, ref_ep, self._video_keys
                )
                self.status_update.emit(f"Ref frames: {list(ref_frames.keys())}")
                if ref_frames:
                    self.ref_frames_ready.emit(ref_frames)
                else:
                    self.status_update.emit("WARNING: No reference video frames found in dataset")

            # Must have real joint states before building any trajectory
            import rclpy
            self.status_update.emit("Waiting for joint states...")
            for _ in range(500):  # ~5 s max
                rclpy.spin_once(self._collector, timeout_sec=0.01)
                if self._collector.joint_map:
                    break
            if not self._collector.joint_map:
                raise RuntimeError(
                    "No joint states received after 5 s — aborting init pose to avoid dangerous movement."
                )

            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                self.status_update.emit(
                    f"Sending robot to init pose (ep {ref_ep}, attempt {attempt})..."
                )
                mod.send_init_pose(
                    self._sender, self._collector.joint_map, positions
                )

                # Wait for trajectory execution
                time.sleep(3.0)

                # Spin to get fresh joint states
                import rclpy
                for _ in range(50):
                    rclpy.spin_once(self._collector, timeout_sec=0.01)

                reached, max_err, worst = mod.check_init_pose_reached(
                    self._collector.joint_map, positions
                )
                if reached:
                    self.status_update.emit(
                        f"Init pose reached (attempt {attempt}, err={max_err:.4f})"
                    )
                    break
                self.status_update.emit(
                    f"Not at init pose (err={max_err:.4f} @ {worst}), retrying..."
                )
            else:
                self.status_update.emit(
                    f"Init pose not reached after {max_attempts} attempts (err={max_err:.4f} @ {worst})"
                )
        except Exception as e:
            self.error_occurred.emit(f"Init pose error: {e}")
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class EvalGuiWindow(QMainWindow):
    def __init__(self, checkpoint="", host="localhost", port=5555, dummy=False):
        super().__init__()
        title = "GR00T Robot Eval" + (" [DUMMY]" if dummy else "")
        self.setWindowTitle(title)
        self.setMinimumSize(900, 700)

        self._default_checkpoint = checkpoint
        self._default_host = host
        self._default_port = port
        self._dummy = dummy

        self._collector = None
        self._sender = None
        self._spin_thread = None
        self._worker = None
        self._init_pose_worker = None
        self._current_ckpt_config = None
        self._video_keys = []
        self._ref_frames = {}  # {cam_key: rgb_ndarray} from init pose
        self._action_mask_checks = {}  # key -> QCheckBox; populated on checkpoint load

        # Dummy-mode episode context (picked via Init Pose button)
        self._dummy_dataset_path = ""
        self._dummy_episode_idx = None
        self._dummy_ckpt_config = None

        # Last init pose selection (reused by Reset Pose button)
        self._last_init_dataset_path = None
        self._last_init_episode_idx = None
        self._last_init_lang = None

        self._build_ui()
        if not self._dummy and self._default_checkpoint:
            self._auto_init_ros2(self._default_checkpoint)

    # ---- UI setup ----

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Row 1: Checkpoint + Server
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Checkpoint:"))
        self.ckpt_input = QLineEdit(self._default_checkpoint)
        self.ckpt_input.setPlaceholderText("output/.../checkpoint-XXXXX")
        self.ckpt_input.setReadOnly(True)
        row1.addWidget(self.ckpt_input, stretch=3)
        row1.addWidget(QLabel("Host:"))
        self.host_input = QLineEdit(self._default_host)
        self.host_input.setFixedWidth(120)
        row1.addWidget(self.host_input)
        row1.addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(self._default_port)
        row1.addWidget(self.port_input)
        layout.addLayout(row1)

        # Row 2: Init Pose + Language
        row2 = QHBoxLayout()
        self.init_pose_btn = QPushButton("Init Pose")
        self.init_pose_btn.clicked.connect(self._on_init_pose)
        row2.addWidget(self.init_pose_btn)
        self.reset_pose_btn = QPushButton("Reset Pose")
        self.reset_pose_btn.clicked.connect(self._on_reset_pose)
        self.reset_pose_btn.setEnabled(False)
        row2.addWidget(self.reset_pose_btn)
        row2.addWidget(QLabel("Language:"))
        self.lang_input = QLineEdit()
        self.lang_input.setPlaceholderText("loaded from dataset (editable)")
        row2.addWidget(self.lang_input, stretch=1)
        layout.addLayout(row2)

        # Row 3: Horizon slider
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Action Horizon:"))
        self.horizon_slider = QSlider(Qt.Horizontal)
        self.horizon_slider.setRange(1, 64)
        self.horizon_slider.setValue(32)
        self.horizon_slider.setTickInterval(8)
        self.horizon_slider.setTickPosition(QSlider.TicksBelow)
        row3.addWidget(self.horizon_slider)
        self.horizon_label = QLabel("32")
        self.horizon_label.setFixedWidth(30)
        row3.addWidget(self.horizon_label)
        self.horizon_slider.valueChanged.connect(
            lambda v: self.horizon_label.setText(str(v))
        )
        layout.addLayout(row3)

        # Row 4: Action mask checkboxes (populated dynamically from checkpoint config)
        self._mask_row_widget = QWidget()
        self._mask_row_layout = QHBoxLayout(self._mask_row_widget)
        self._mask_row_layout.setContentsMargins(0, 0, 0, 0)
        self._mask_row_layout.addWidget(QLabel("Send:"))
        self._mask_row_layout.addStretch(1)
        layout.addWidget(self._mask_row_widget)

        # Row 5: Buttons + Status
        row4 = QHBoxLayout()
        self.start_btn = QPushButton("Start Episode")
        self.start_btn.clicked.connect(self._on_start_episode)
        row4.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop Episode")
        self.stop_btn.clicked.connect(self._on_stop_episode)
        self.stop_btn.setEnabled(False)
        row4.addWidget(self.stop_btn)
        self.status_label = QLabel("Status: Idle")
        row4.addWidget(self.status_label, stretch=1)
        layout.addLayout(row4)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.video_label.setMinimumHeight(300)
        self.video_label.setStyleSheet("background-color: #1a1a1a;")
        layout.addWidget(self.video_label, stretch=1)

        # Action history visualization panel (in a scroll area so all subplots are reachable)
        self.viz_label = QLabel()
        self.viz_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.viz_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.viz_label.setStyleSheet("background-color: #1a1a1a;")
        self._viz_scroll = QScrollArea()
        self._viz_scroll.setWidget(self.viz_label)
        self._viz_scroll.setWidgetResizable(True)
        self._viz_scroll.setMinimumHeight(300)
        self._viz_scroll.setStyleSheet("QScrollArea { background-color: #1a1a1a; border: none; }")
        layout.addWidget(self._viz_scroll, stretch=2)

    # ---- Actions ----

    def _auto_init_ros2(self, checkpoint_path):
        """Create ROS2 nodes and start live preview at launch without waiting for Init Pose."""
        try:
            ckpt_config = parse_checkpoint_config(checkpoint_path)
        except Exception as e:
            self.status_label.setText(f"Status: Config error at startup: {e}")
            return
        state_keys = ckpt_config["state_keys"]
        video_keys = ckpt_config["video_keys"]
        action_keys = ckpt_config["action_keys"]
        self._collector = AiWorkerObsCollector(
            enabled_state_keys=state_keys,
            video_keys=video_keys,
            use_compressed_rgb=True,
        )
        self._sender = AiWorkerCommandSender(enabled_action_keys=action_keys)
        self._video_keys = video_keys
        self._current_ckpt_config = ckpt_config
        self._spin_thread = Ros2SpinThread(
            self._collector, self._sender, self._video_keys,
            ref_frames_getter=lambda: self._ref_frames,
        )
        self._spin_thread.frame_ready.connect(self._update_video_display)
        self._spin_thread.start()
        self.status_label.setText("Status: Live preview active — press Init Pose to begin")

    def _on_init_pose(self):
        ckpt_path = self.ckpt_input.text().strip()
        if not ckpt_path:
            self.status_label.setText("Status: No checkpoint path set")
            return

        try:
            ckpt_config = parse_checkpoint_config(ckpt_path)
        except Exception as e:
            self.status_label.setText(f"Status: Config error: {e}")
            return

        dataset_path = ckpt_config.get("dataset_path", "")
        if dataset_path and not os.path.isabs(dataset_path):
            dataset_path = os.path.normpath(os.path.join(os.getcwd(), dataset_path))

        if not dataset_path or not os.path.isdir(dataset_path):
            self.status_label.setText(f"Status: Dataset not found: {dataset_path}")
            return

        if self._dummy:
            self._dummy_init_pose(dataset_path, ckpt_config)
            return

        # Ensure ROS2 nodes exist
        state_keys = ckpt_config["state_keys"]
        video_keys = ckpt_config["video_keys"]
        action_keys = ckpt_config["action_keys"]

        # Build action mask checkboxes now so the user can configure before Start Episode
        self._rebuild_mask_checkboxes(action_keys)

        if self._collector is None:
            self.status_label.setText("Status: Initializing ROS2 nodes...")
            QApplication.processEvents()
            self._collector = AiWorkerObsCollector(
                enabled_state_keys=state_keys,
                video_keys=video_keys,
                use_compressed_rgb=True,
            )
            self._sender = AiWorkerCommandSender(
                enabled_action_keys=action_keys,
            )
            self._video_keys = video_keys
            # Start live preview immediately — paused while InitPoseWorker runs
            self._spin_thread = Ros2SpinThread(
                self._collector, self._sender, self._video_keys,
                ref_frames_getter=lambda: self._ref_frames,
            )
            self._spin_thread.frame_ready.connect(self._update_video_display)
            self._spin_thread.start()

        self._current_ckpt_config = ckpt_config

        # Pick instruction + episode in GUI thread (same dialog logic as dummy mode)
        init_mod = _load_init_pose_module()
        instructions = init_mod.get_unique_instructions(dataset_path)
        if not instructions:
            self.status_label.setText("Status: No task instructions in dataset")
            return

        if len(instructions) == 1:
            picked = instructions[0]
        else:
            picked, ok = QInputDialog.getItem(
                self, "Select task instruction",
                "Choose an instruction for this episode:",
                instructions, 0, editable=False,
            )
            if not ok:
                return

        matching_eps = init_mod.get_episodes_for_instruction(dataset_path, picked)
        if not matching_eps:
            self.status_label.setText(f"Status: No episodes match: {picked}")
            return

        import random
        episode_idx = random.choice(matching_eps)
        self._last_init_dataset_path = dataset_path
        self._last_init_episode_idx = episode_idx
        self._last_init_lang = picked
        self.lang_input.setText(picked)

        # Pause spin thread — init pose worker will spin the nodes
        if self._spin_thread:
            self._spin_thread.pause()

        # Run init pose in background thread
        self.init_pose_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self._init_pose_worker = InitPoseWorker(
            self._sender, self._collector, dataset_path,
            video_keys=self._video_keys,
            episode_idx=episode_idx,
        )
        self._init_pose_worker.status_update.connect(self._on_status_update)
        self._init_pose_worker.lang_loaded.connect(self._on_lang_loaded)
        self._init_pose_worker.ref_frames_ready.connect(self._on_ref_frames_ready)
        self._init_pose_worker.finished.connect(self._on_manual_init_pose_finished)
        self._init_pose_worker.error_occurred.connect(self._on_error)
        self._init_pose_worker.start()
        self.status_label.setText("Status: Resetting to init pose...")

    # ---- Dummy-mode helpers ----

    def _dummy_init_pose(self, dataset_path, ckpt_config):
        """Show instruction picker, choose random matching episode, populate lang."""
        init_mod = _load_init_pose_module()
        instructions = init_mod.get_unique_instructions(dataset_path)
        if not instructions:
            self.status_label.setText("Status: No task instructions in dataset")
            return

        if len(instructions) == 1:
            picked = instructions[0]
        else:
            picked, ok = QInputDialog.getItem(
                self, "Select task instruction",
                "Choose an instruction for this episode:",
                instructions, 0, editable=False,
            )
            if not ok:
                return

        matching_eps = init_mod.get_episodes_for_instruction(dataset_path, picked)
        if not matching_eps:
            self.status_label.setText(f"Status: No episodes match: {picked}")
            return

        import random
        episode_idx = random.choice(matching_eps)

        self._dummy_dataset_path = dataset_path
        self._dummy_episode_idx = episode_idx
        self._dummy_ckpt_config = ckpt_config
        self._video_keys = ckpt_config["video_keys"]
        self._current_ckpt_config = ckpt_config
        self._last_init_dataset_path = dataset_path
        self._last_init_episode_idx = episode_idx
        self._last_init_lang = picked
        self.lang_input.setText(picked)
        # Build action mask checkboxes now so the user can configure before Start Episode
        self._rebuild_mask_checkboxes(ckpt_config["action_keys"])
        self.reset_pose_btn.setEnabled(True)
        self.status_label.setText(
            f"Status: [DUMMY] ep {episode_idx} / {picked!r} — press Start Episode"
        )

    def _on_lang_loaded(self, lang):
        self.lang_input.setText(lang)

    def _on_ref_frames_ready(self, ref_frames):
        self._ref_frames = ref_frames

    def _on_manual_init_pose_finished(self):
        self.init_pose_btn.setEnabled(True)
        self.reset_pose_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.status_label.setText("Status: Idle (init pose reached)")

        # Resume or start live preview
        if self._spin_thread is not None:
            self._spin_thread.resume()
        elif self._collector is not None:
            self._spin_thread = Ros2SpinThread(
                self._collector, self._sender, self._video_keys,
                ref_frames_getter=lambda: self._ref_frames,
            )
            self._spin_thread.frame_ready.connect(self._update_video_display)
            self._spin_thread.start()

    def _on_reset_pose(self):
        if self._last_init_dataset_path is None:
            return

        if self._dummy:
            self.status_label.setText(
                f"Status: [DUMMY] ep {self._last_init_episode_idx} — press Start Episode"
            )
            return

        # Real robot: re-run InitPoseWorker with stored params, no dialog
        if self._spin_thread:
            self._spin_thread.pause()

        self.init_pose_btn.setEnabled(False)
        self.reset_pose_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self._init_pose_worker = InitPoseWorker(
            self._sender, self._collector, self._last_init_dataset_path,
            video_keys=self._video_keys,
            episode_idx=self._last_init_episode_idx,
        )
        self._init_pose_worker.status_update.connect(self._on_status_update)
        self._init_pose_worker.lang_loaded.connect(self._on_lang_loaded)
        self._init_pose_worker.ref_frames_ready.connect(self._on_ref_frames_ready)
        self._init_pose_worker.finished.connect(self._on_manual_init_pose_finished)
        self._init_pose_worker.error_occurred.connect(self._on_error)
        self._init_pose_worker.start()
        self.status_label.setText("Status: Resetting to init pose...")

    def _set_running_state(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.init_pose_btn.setEnabled(not running)
        has_last = self._last_init_dataset_path is not None
        self.reset_pose_btn.setEnabled(not running and has_last)
        self.lang_input.setReadOnly(running)
        self.host_input.setReadOnly(running)
        self.port_input.setReadOnly(running)

    def _on_start_episode(self):
        if self._dummy:
            self._start_dummy_episode()
            return

        ckpt_path = self.ckpt_input.text().strip()
        if not ckpt_path:
            self.status_label.setText("Status: No checkpoint path set")
            return

        # Parse checkpoint config
        try:
            ckpt_config = parse_checkpoint_config(ckpt_path)
        except Exception as e:
            self.status_label.setText(f"Status: Config error: {e}")
            return

        # Update slider max from checkpoint config
        max_horizon = min(ckpt_config.get("action_horizon", 32), 32)
        self.horizon_slider.setRange(1, max_horizon)
        if self.horizon_slider.value() > max_horizon:
            self.horizon_slider.setValue(max_horizon)

        video_keys = ckpt_config["video_keys"]
        state_keys = ckpt_config["state_keys"]
        action_keys = ckpt_config["action_keys"]
        language_key = ckpt_config["language_key"]

        # Recreate ROS2 nodes if config changed
        config_changed = (
            self._current_ckpt_config is None
            or self._current_ckpt_config["video_keys"] != video_keys
            or self._current_ckpt_config["state_keys"] != state_keys
            or self._current_ckpt_config["action_keys"] != action_keys
        )

        if config_changed:
            self.status_label.setText("Status: Initializing ROS2 nodes...")
            QApplication.processEvents()
            self._teardown_ros2()

            self._collector = AiWorkerObsCollector(
                enabled_state_keys=state_keys,
                video_keys=video_keys,
                use_compressed_rgb=True,
            )
            self._sender = AiWorkerCommandSender(
                enabled_action_keys=action_keys,
            )
            self._video_keys = video_keys

            # Start spin thread
            self._spin_thread = Ros2SpinThread(
                self._collector, self._sender, video_keys,
                ref_frames_getter=lambda: self._ref_frames,
            )
            self._spin_thread.frame_ready.connect(self._update_video_display)
            self._spin_thread.start()

        self._current_ckpt_config = ckpt_config

        # Pause spin thread -- worker will spin during its loop
        if self._spin_thread:
            self._spin_thread.pause()

        # Create policy client + adapter
        host = self.host_input.text()
        port = self.port_input.value()

        try:
            policy_client = PolicyClient(host=host, port=port)
        except Exception as e:
            self.status_label.setText(f"Status: Policy error: {e}")
            if self._spin_thread:
                self._spin_thread.resume()
            return

        adapter = AiWorkerAdapter(
            policy_client=policy_client,
            state_keys=state_keys,
            action_keys=action_keys,
            video_keys=video_keys,
            language_key=language_key,
            dims_by_key=DIMS_BY_KEY,
        )

        # Start episode worker
        self._worker = EpisodeWorker(
            collector=self._collector,
            sender=self._sender,
            adapter=adapter,
            horizon=self.horizon_slider.value(),
            lang=self.lang_input.text(),
            model_video_keys=video_keys,
            action_keys=action_keys,
            get_action_mask=self._get_action_mask,
        )
        self._worker.frame_ready.connect(self._update_video_display)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.episode_finished.connect(self._on_episode_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.action_viz_ready.connect(self._update_viz_display)

        # Live horizon updates
        self.horizon_slider.valueChanged.connect(self._on_horizon_changed)

        self._worker.start()
        self._set_running_state(True)
        self.status_label.setText("Status: Episode running")

    def _start_dummy_episode(self):
        """Dummy-mode Start Episode: requires Init Pose to have picked an episode first."""
        if self._dummy_episode_idx is None or not self._dummy_dataset_path:
            self.status_label.setText("Status: Press Init Pose first to pick an instruction")
            return

        ckpt_config = self._dummy_ckpt_config
        video_keys = ckpt_config["video_keys"]
        state_keys = ckpt_config["state_keys"]
        action_keys = ckpt_config["action_keys"]
        language_key = ckpt_config["language_key"]

        max_horizon = min(ckpt_config.get("action_horizon", 32), 32)
        self.horizon_slider.setRange(1, max_horizon)
        if self.horizon_slider.value() > max_horizon:
            self.horizon_slider.setValue(max_horizon)

        # Build fresh dummy nodes each episode (so the collector restarts from step 0).
        self._collector = DummyCollector(
            dataset_path=self._dummy_dataset_path,
            episode_idx=self._dummy_episode_idx,
            enabled_state_keys=state_keys,
            video_keys=video_keys,
        )
        self._sender = DummySender(
            enabled_action_keys=action_keys,
            total_frames=self._collector.num_frames,
        )
        self._video_keys = video_keys

        host = self.host_input.text()
        port = self.port_input.value()
        try:
            policy_client = PolicyClient(host=host, port=port)
        except Exception as e:
            self.status_label.setText(f"Status: Policy error: {e}")
            return

        adapter = AiWorkerAdapter(
            policy_client=policy_client,
            state_keys=state_keys,
            action_keys=action_keys,
            video_keys=video_keys,
            language_key=language_key,
            dims_by_key=DIMS_BY_KEY,
        )

        self._worker = EpisodeWorker(
            collector=self._collector,
            sender=self._sender,
            adapter=adapter,
            horizon=self.horizon_slider.value(),
            lang=self.lang_input.text(),
            exec_sec=0.0,
            use_ros=False,
            model_video_keys=video_keys,
            action_keys=action_keys,
            get_action_mask=self._get_action_mask,
        )
        self._worker.frame_ready.connect(self._update_video_display)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.episode_finished.connect(self._on_episode_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.action_viz_ready.connect(self._update_viz_display)

        self.horizon_slider.valueChanged.connect(self._on_horizon_changed)

        self._worker.start()
        self._set_running_state(True)
        self.status_label.setText(
            f"Status: [DUMMY] ep {self._dummy_episode_idx} running ({self._collector.num_frames} frames)"
        )

    def _on_stop_episode(self):
        if self._worker:
            self.status_label.setText("Status: Stopping...")
            self._worker.stop()

    def _on_episode_finished(self):
        self._set_running_state(False)
        self.viz_label.clear()

        # Don't overwrite an error message that _on_error already set
        if self.status_label.text().startswith("Status: ERROR"):
            if self._spin_thread:
                self._spin_thread.resume()
            return

        if self._dummy:
            self._finalize_dummy_episode()
            self.status_label.setText(
                "Status: [DUMMY] episode done — press Init Pose to start another"
            )
            return

        # Resume spin thread for live preview
        if self._spin_thread:
            self._spin_thread.resume()

        self.status_label.setText("Status: Idle (press Init Pose to reset)")

    def _finalize_dummy_episode(self):
        """Save plot for current dummy episode (possibly partial). Idempotent."""
        if not isinstance(self._sender, DummySender):
            return
        ckpt_path = self.ckpt_input.text().strip()
        if not ckpt_path or self._dummy_episode_idx is None:
            return
        try:
            ckpt_config = self._dummy_ckpt_config or {}
            save_dir = os.path.join(ckpt_path, "dummy_eval")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"ep_{self._dummy_episode_idx}.jpeg")
            self._sender.finalize(
                save_path=save_path,
                state_keys=ckpt_config.get("state_keys", []),
                action_keys=ckpt_config.get("action_keys", []),
                action_horizon=self.horizon_slider.value(),
            )
        except Exception as e:
            print(f"[dummy] finalize failed: {e}")

    def _on_horizon_changed(self, value):
        if self._worker:
            self._worker.horizon = value

    def _on_status_update(self, msg):
        self.status_label.setText(f"Status: {msg}")

    def _on_error(self, msg):
        self.status_label.setText(f"Status: ERROR - {msg}")
        self._set_running_state(False)
        if self._dummy:
            self._finalize_dummy_episode()
            return
        if self._spin_thread:
            self._spin_thread.resume()

    # ---- Mask checkboxes ----

    def _rebuild_mask_checkboxes(self, action_keys):
        """Recreate per-key checkboxes. Checked = send, unchecked = zero."""
        for cb in self._action_mask_checks.values():
            self._mask_row_layout.removeWidget(cb)
            cb.deleteLater()
        self._action_mask_checks.clear()
        # Re-add the stretch after clearing
        for i in reversed(range(self._mask_row_layout.count())):
            item = self._mask_row_layout.itemAt(i)
            if item and item.spacerItem():
                self._mask_row_layout.removeItem(item)
        for key in action_keys:
            cb = QCheckBox(key)
            cb.setChecked(True)
            self._mask_row_layout.addWidget(cb)
            self._action_mask_checks[key] = cb
        self._mask_row_layout.addStretch(1)

    def _get_action_mask(self):
        """Returns set of action keys currently unchecked (should be zeroed before send)."""
        return {k for k, cb in self._action_mask_checks.items() if not cb.isChecked()}

    # ---- Video display ----

    def _update_video_display(self, rgb):
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        ch = rgb.shape[2] if rgb.ndim == 3 else 1
        bytes_per_line = ch * w
        qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _update_viz_display(self, rgb):
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        # tobytes() makes a safe copy; rgb.data (memoryview) can be GC'd before Qt renders
        qimage = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        label_w = self.viz_label.width()
        if label_w <= 0:
            label_w = 800
        scaled = pixmap.scaledToWidth(label_w, Qt.SmoothTransformation)
        # Expand the label so the full chart is visible without clipping
        self.viz_label.setMinimumHeight(scaled.height())
        self.viz_label.setPixmap(scaled)

    # ---- Cleanup ----

    def _teardown_ros2(self):
        if self._spin_thread:
            self._spin_thread.stop()
            self._spin_thread.wait(5000)
            self._spin_thread = None
        if self._collector is not None and hasattr(self._collector, "destroy_node"):
            self._collector.destroy_node()
        self._collector = None
        if self._sender is not None and hasattr(self._sender, "destroy_node"):
            self._sender.destroy_node()
        self._sender = None

    def shutdown(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)
        if self._init_pose_worker:
            self._init_pose_worker.wait(5000)
        # Save partial plot if a dummy episode was in progress.
        if self._dummy:
            self._finalize_dummy_episode()
        self._teardown_ros2()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GR00T Robot Eval GUI")
    parser.add_argument("--checkpoint", default="", help="Default checkpoint path")
    parser.add_argument("--host", default="localhost", help="Default policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Default policy server port")
    parser.add_argument("--dummy", action="store_true",
                        help="Feed dataset frames instead of live ROS2 topics (no robot required)")
    args = parser.parse_args()

    if not args.dummy:
        if not _ROS2_AVAILABLE:
            raise RuntimeError("ROS2 not installed. Pass --dummy to run without a robot.")
        rclpy.init()
    app = QApplication(sys.argv)
    window = EvalGuiWindow(
        checkpoint=args.checkpoint, host=args.host, port=args.port, dummy=args.dummy,
    )
    window.show()
    exit_code = app.exec_()
    window.shutdown()
    if not args.dummy and _ROS2_AVAILABLE:
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
