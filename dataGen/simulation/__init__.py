"""Simulation execution: stepping, episode assembly and branch rollouts.

Modules (later phase):
    engine  -- fixed-step integration at SIM_DT with LOG_DECIMATION logging
    episode -- one full episode: build, run, log, summarise
    rollout -- N-times branch rollouts from a WorldSnapshot (doc 3.3)
"""
