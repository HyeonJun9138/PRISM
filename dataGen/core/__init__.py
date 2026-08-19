"""Core primitives shared by every PRISM dataGen package.

Modules:
    types      -- IntEnum label sets and display tables (no logic)
    constants  -- SI constants, timing, separation criteria, paths
    state      -- AgentState / PerformanceEnvelope / snapshots, wrap_angle
    rng        -- RNGHub, deterministic named random streams
    kinematics -- (later phase) mode -> command mapping
    integrator -- (later phase) fixed-step state propagation

Import submodules explicitly (``from core.state import wrap_angle``); this
package deliberately re-exports nothing so that import order stays trivial.
"""
