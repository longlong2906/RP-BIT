"""Kaggle-only A5 held-out at a 10 s cap: the primary A5 arm against the self-stopping C++ SEM-RRT* only, every
method stopping on its own convergence, one timed lane.

Same controller as fairtime.py with this experiment's registration: logs/a4_rpbit_a5_10s_20261122/registration.json.
"""
from kaggle_a4_rpbit.fairtime import main

if __name__ == '__main__':
    raise SystemExit(main(experiment='a5_10s'))
