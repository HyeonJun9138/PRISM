"""Agent models: platform envelopes, maneuver logic and intent policies.

Modules (later phase):
    base      -- Agent container binding state, envelope, intent and mode
    platforms -- PerformanceEnvelope tables per PlatformType
    maneuver  -- stochastic maneuver-mode switching with dwell time (doc 3.3)
    intent    -- intent policies that emit mode preferences
    behavior  -- per-agent personality parameters (turn frequency, sharpness)
"""
