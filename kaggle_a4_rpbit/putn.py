"""Kaggle-only PUTN comparison: the locked primary A4, the self-stopping C++ SEM-RRT* and two self-stopping PUTN arms
(source tau and the declared 35-degree tau slope cap), 10 s cap, one timed lane.

Same controller as fairtime.py with this experiment's registration: logs/a4_rpbit_putn_20261119/registration.json.
"""
from kaggle_a4_rpbit.fairtime import main

if __name__ == '__main__':
    raise SystemExit(main(experiment='putn'))
