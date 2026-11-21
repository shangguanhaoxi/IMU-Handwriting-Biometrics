"""Subject-level gravity harmonization for six-axis IMU recordings.

The config is provisional until each subject's acquisition/export format is
verified. For raw accelerometer signals, subtract an estimated slow component.
This is an approximation, not a perfect gravity-compensated inertial signal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


DEFAULT_GRAVITY_CONFIG = "imu_gravity_config.json"


def load_gravity_config(path=DEFAULT_GRAVITY_CONFIG):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    linear = config.get("linear_acceleration_subjects")
    raw = config.get("raw_acceleration_subjects")
    if not isinstance(linear, list) or not isinstance(raw, list):
        raise ValueError("Gravity config must list linear_acceleration_subjects and raw_acceleration_subjects")
    if not all(isinstance(s, str) and s.strip() for s in linear + raw):
        raise ValueError("All gravity-config subject IDs must be nonempty strings")
    if len(set(linear + raw)) != len(linear + raw):
        raise ValueError("Duplicate or overlapping subjects in gravity config")
    correction = config.get("raw_gravity_removal", {})
    method = correction.get("method")
    if method not in ("lowpass", "mean"):
        raise ValueError("raw_gravity_removal.method must be 'lowpass' or 'mean'")
    if method == "lowpass":
        cutoff = float(correction.get("cutoff_hz", 0))
        order = int(correction.get("order", 0))
        if not np.isfinite(cutoff) or cutoff <= 0 or order < 1 or order > 8:
            raise ValueError("Invalid gravity low-pass cutoff_hz or order")
    config["_digest"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def validate_gravity_subjects(config, subjects):
    listed = set(config["linear_acceleration_subjects"] + config["raw_acceleration_subjects"])
    discovered = set(subjects)
    if listed != discovered:
        raise ValueError(
            f"Gravity config subjects do not match discovered subjects. "
            f"Missing={sorted(discovered - listed)}; extra={sorted(listed - discovered)}. "
            "Update the JSON explicitly rather than silently assuming a sensor type."
        )


def correct_gravity(x, t, subject, config):
    """Return corrected 6-axis signal and estimated gravity-vector series.

    Linear-acceleration recordings are returned unchanged with a zero estimate.
    Raw acceleration is corrected along the three measured sensor axes.
    """
    x = np.asarray(x)
    t = np.asarray(t, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 6 or len(x) != len(t):
        raise ValueError("Expected six-axis samples and matching timestamps")
    if subject in config["linear_acceleration_subjects"]:
        return x.copy(), np.zeros_like(x[:, :3], dtype=np.float64)
    if subject not in config["raw_acceleration_subjects"]:
        raise ValueError(f"Subject not configured for gravity removal: {subject}")
    if len(t) < 4 or not np.all(np.diff(t) > 0):
        raise ValueError("Gravity correction requires at least four increasing timestamps")
    acc = x[:, :3].astype(np.float64)
    method = config["raw_gravity_removal"]["method"]
    if method == "mean":
        gravity = np.broadcast_to(acc.mean(axis=0, keepdims=True), acc.shape).copy()
    else:
        dt = float(np.median(np.diff(t)))
        sample_rate = 1.0 / dt
        cutoff = float(config["raw_gravity_removal"]["cutoff_hz"])
        order = int(config["raw_gravity_removal"]["order"])
        if not np.isfinite(sample_rate) or cutoff >= sample_rate / 2.0:
            raise ValueError(
                f"Gravity cutoff {cutoff} Hz must be below Nyquist frequency {sample_rate / 2.0} Hz"
            )
        sos = butter(order, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        # filtfilt has a nonzero minimum padding requirement; short recordings
        # fail explicitly rather than using a different undocumented method.
        gravity = sosfiltfilt(sos, acc, axis=0)
    out = x.copy()
    out[:, :3] = (acc - gravity).astype(x.dtype)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite signal after gravity correction")
    return out, gravity
