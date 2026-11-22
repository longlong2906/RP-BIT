"""Kaggle-only A5 ablation: the 2^4 factorial of the four A5 mechanisms (16 cells, cell 0000 = A4, 1111 = the A5
primary) beside the self-stopping C++ SEM-RRT*, every method stopping on its own convergence under a 20 s cap, one
timed lane.

Same controller as fairtime.py with this experiment's registration: logs/a4_rpbit_a5_ablation_20261122/registration.json.
"""
from kaggle_a4_rpbit.fairtime import main

if __name__ == '__main__':
    raise SystemExit(main(experiment='a5_ablation'))
