import numpy as np
from sudokuGA import SudokuGA

init_puzzle = np.array([
  [8, 7, 0, 0, 0, 6, 5, 9, 0], 
  [5, 0, 3, 0, 8, 2, 7, 6, 0], 
  [2, 6, 0, 1, 7, 0, 0, 0, 8], 
  [0, 9, 6, 0, 0, 8, 0, 0, 0], 
  [0, 3, 0, 0, 0, 9, 2, 8, 4], 
  [0, 0, 2, 7, 0, 4, 6, 0, 0], 
  [0, 0, 4, 0, 5, 0, 0, 1, 6], 
  [6, 1, 7, 0, 0, 0, 0, 0, 0], 
  [0, 0, 0, 2, 6, 0, 0, 0, 3]
])

ga = SudokuGA(init_puzzle)
ga.solve()