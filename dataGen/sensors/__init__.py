"""Onboard surveillance sensor and tracker models (doc 3.5).

Targets are non-cooperative: no ADS-B / Remote ID, only noisy, intermittent,
delayed onboard tracks. Missing detections are masked, never interpolated.

Modules (later phase):
    model   -- polar noise, detection probability, latency
    tracker -- track id assignment, ageing and drop-after-misses logic
"""
