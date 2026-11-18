"""Kaggle-only self-stop comparison: the locked primary A4 against the C++ SEM-RRT* port, both stopping on
convergence (SEM with the A4 rule: < 0.1% relative improvement over 4,000 iterations), 20 s cap.

Same controller as fairtime.py with this experiment's registration: logs/a4_rpbit_selfstop_20261117/registration.json.
"""
from kaggle_a4_rpbit.fairtime import main

if __name__ == '__main__':
    raise SystemExit(main(experiment='selfstop'))
