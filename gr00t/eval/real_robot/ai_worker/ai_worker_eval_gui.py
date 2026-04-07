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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QSlider, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import rclpy

from gr00t.eval.real_robot.ai_worker.ai_worker_eval import (
    AiWorkerObsCollector,
    AiWorkerCommandSender,
    AiWorkerAdapter,
    DIMS_BY_KEY,
    concat_obs_rgb,
    run_eval_loop,
    parse_checkpoint_config,
)
from gr00t.policy.server_client import PolicyClient


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

    def __init__(self, collector, sender, video_keys):
        super().__init__()
        self._collector = collector
        self._sender = sender
        self._video_keys = video_keys
        self._stop = False
        self._paused = False

    def run(self):
        last_frame_time = 0.0
        while not self._stop and rclpy.ok():
            if self._paused:
                time.sleep(0.01)
                continue

            rclpy.spin_once(self._collector, timeout_sec=0.01)
            rclpy.spin_once(self._sender, timeout_sec=0.0)

            # Emit frames at ~10 fps for live preview
            now = time.time()
            if now - last_frame_time > 0.1 and self._collector.latest_rgb_by_key:
                rgb = concat_obs_rgb(self._collector.latest_rgb_by_key)
                if rgb is not None:
                    self.frame_ready.emit(rgb)
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

    def __init__(
        self, collector, sender, adapter, horizon, lang, exec_sec=1.0
    ):
        super().__init__()
        self._collector = collector
        self._sender = sender
        self._adapter = adapter
        self._horizon = horizon
        self._lang = lang
        self._exec_sec = exec_sec
        self._stop_event = __import__("threading").Event()

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, val):
        self._horizon = val

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            run_eval_loop(
                collector=self._collector,
                sender=self._sender,
                adapter=self._adapter,
                lang_instruction=self._lang,
                action_horizon=self._horizon,
                exec_sec=self._exec_sec,
                should_stop=self._stop_event.is_set,
                on_obs=lambda obs: self.frame_ready.emit(
                    concat_obs_rgb(obs.get("rgb", {}))
                ),
                on_status=lambda msg: self.status_update.emit(msg),
                get_action_horizon=lambda: self._horizon,
            )
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.episode_finished.emit()


# ---------------------------------------------------------------------------
# InitPoseWorker -- resets robot to init pose after episode
# ---------------------------------------------------------------------------

class InitPoseWorker(QThread):
    status_update = pyqtSignal(str)
    lang_loaded = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, sender, collector, dataset_path):
        super().__init__()
        self._sender = sender
        self._collector = collector
        self._dataset_path = dataset_path

    def run(self):
        try:
            mod = _load_init_pose_module()

            self.status_update.emit("Loading init pose from dataset...")
            positions, ref_ep, lang = mod.compute_init_pose(self._dataset_path)
            if lang:
                self.lang_loaded.emit(lang)

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
    def __init__(self, checkpoint="", host="localhost", port=5555):
        super().__init__()
        self.setWindowTitle("GR00T Robot Eval")
        self.setMinimumSize(900, 700)

        self._default_checkpoint = checkpoint
        self._default_host = host
        self._default_port = port

        self._collector = None
        self._sender = None
        self._spin_thread = None
        self._worker = None
        self._init_pose_worker = None
        self._current_ckpt_config = None
        self._video_keys = []

        self._build_ui()

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

        # Row 4: Buttons + Status
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

    # ---- Actions ----

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

        # Ensure ROS2 nodes exist
        state_keys = ckpt_config["state_keys"]
        video_keys = ckpt_config["video_keys"]
        action_keys = ckpt_config["action_keys"]

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

        self._current_ckpt_config = ckpt_config

        # Pause spin thread — init pose worker will spin the nodes
        if self._spin_thread:
            self._spin_thread.pause()

        # Run init pose in background thread
        self.init_pose_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self._init_pose_worker = InitPoseWorker(
            self._sender, self._collector, dataset_path,
        )
        self._init_pose_worker.status_update.connect(self._on_status_update)
        self._init_pose_worker.lang_loaded.connect(self._on_lang_loaded)
        self._init_pose_worker.finished.connect(self._on_manual_init_pose_finished)
        self._init_pose_worker.error_occurred.connect(self._on_error)
        self._init_pose_worker.start()
        self.status_label.setText("Status: Resetting to init pose...")

    def _on_lang_loaded(self, lang):
        self.lang_input.setText(lang)

    def _on_manual_init_pose_finished(self):
        self.init_pose_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.status_label.setText("Status: Idle (init pose reached)")

        # Resume or start live preview
        if self._spin_thread is not None:
            self._spin_thread.resume()
        elif self._collector is not None:
            self._spin_thread = Ros2SpinThread(
                self._collector, self._sender, self._video_keys
            )
            self._spin_thread.frame_ready.connect(self._update_video_display)
            self._spin_thread.start()

    def _set_running_state(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.init_pose_btn.setEnabled(not running)
        self.lang_input.setReadOnly(running)
        self.ckpt_input.setReadOnly(running)
        self.host_input.setReadOnly(running)
        self.port_input.setReadOnly(running)

    def _on_start_episode(self):
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
        max_horizon = ckpt_config.get("action_horizon", 32)
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
                self._collector, self._sender, video_keys
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
        )
        self._worker.frame_ready.connect(self._update_video_display)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.episode_finished.connect(self._on_episode_finished)
        self._worker.error_occurred.connect(self._on_error)

        # Live horizon updates
        self.horizon_slider.valueChanged.connect(self._on_horizon_changed)

        self._worker.start()
        self._set_running_state(True)
        self.status_label.setText("Status: Episode running")

    def _on_stop_episode(self):
        if self._worker:
            self.status_label.setText("Status: Stopping...")
            self._worker.stop()

    def _on_episode_finished(self):
        self._set_running_state(False)

        # Resume spin thread for live preview
        if self._spin_thread:
            self._spin_thread.resume()

        self.status_label.setText("Status: Idle (press Init Pose to reset)")

    def _on_horizon_changed(self, value):
        if self._worker:
            self._worker.horizon = value

    def _on_status_update(self, msg):
        self.status_label.setText(f"Status: {msg}")

    def _on_error(self, msg):
        self.status_label.setText(f"Status: ERROR - {msg}")
        self._set_running_state(False)
        if self._spin_thread:
            self._spin_thread.resume()

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

    # ---- Cleanup ----

    def _teardown_ros2(self):
        if self._spin_thread:
            self._spin_thread.stop()
            self._spin_thread.wait(5000)
            self._spin_thread = None
        if self._collector:
            self._collector.destroy_node()
            self._collector = None
        if self._sender:
            self._sender.destroy_node()
            self._sender = None

    def shutdown(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)
        if self._init_pose_worker:
            self._init_pose_worker.wait(5000)
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
    args = parser.parse_args()

    rclpy.init()
    app = QApplication(sys.argv)
    window = EvalGuiWindow(
        checkpoint=args.checkpoint, host=args.host, port=args.port,
    )
    window.show()
    exit_code = app.exec_()
    window.shutdown()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
