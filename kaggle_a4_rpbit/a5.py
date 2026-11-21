"""Kaggle-only A5 held-out: the primary A5 arm and its C1 + C2 arm beside the locked A4 and the self-stopping C++
SEM-RRT*, every method stopping on its own convergence under a 20 s cap, one timed lane.

Same controller as fairtime.py with this experiment's registration: logs/a4_rpbit_a5_heldout_20261122/registration.json.
"""
from kaggle_a4_rpbit.fairtime import main

if __name__ == '__main__':
    raise SystemExit(main(experiment='a5'))
