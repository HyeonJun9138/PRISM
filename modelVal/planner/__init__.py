"""Frenet-frame motion planner for the modelVal ownship.

Ported from the standalone JS demo at D:\\FrenetFramePath (rev 2026-08).
The planner is deliberately kept semantically identical to the original --
the paper's experiment swaps only the RISK INPUT (none / CV / PRISM mixture),
never the planner itself.
"""
