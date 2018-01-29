# -*- coding: utf-8 -*-
"""
This scrpit uses nqueens.py module for solving 8-queens problem by using
a genetic algorithm
"""
import sys
print('Python version:', sys.version)

import nqueens as nq


solver = nq.Solver_8_queens()
best_fit, epoch_num, visualization = solver.solve()
print('Best solution:')
print('Fitness:', best_fit)
print('Iterations:', epoch_num)
print(visualization)
