import numpy as np
import random
random.seed()

class SudokuGA:
    
    def __init__(self, puzzle, population_size=1000, mutation_rate=0.05, max_generations=10000, mutation_rate_factor=1.5, stagnation_threshold=50, reset_threshold=100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.mutation_rate_factor = mutation_rate_factor
        self.stagnation_threshold = stagnation_threshold
        self.reset_threshold = reset_threshold

        self.puzzle = self.fill_obvious_cells(puzzle)
        self.is_solveable = self.check_solveable()
        self.fixed_cells = (self.puzzle != 0)
        self.population = self.generate_population()

        self.best_fitness = 243
        
    def check_solveable(self):
        for i in range(9):
            row = self.puzzle[i, :]
            row = row[row != 0]
            if len(row) != len(np.unique(row)):
                return False

            col = self.puzzle[:, i]
            col = col[col != 0]
            if len(col) != len(np.unique(col)):
                return False

            square = self.puzzle[(i//3)*3:(i//3)*3+3, (i%3)*3:(i%3)*3+3].flatten()
            square = square[square != 0]
            if len(square) != len(np.unique(square)):
                return False
        return True
    
    def fill_obvious_cells(self, puzzle):
        new_chromosome = puzzle.copy()
        cells_filled = 0

        for i in range(9):
            row = puzzle[i, :]
            missing_numbers_in_row = [n for n in range(1, 10) if n not in row]

            for j in range(9):
                if puzzle[i, j] != 0:
                    continue

                col = puzzle[:, j]
                missing_numbers_in_col = [n for n in range(1, 10) if n not in col]

                square_row_start = (i // 3) * 3
                square_col_start = (j // 3) * 3
                square = puzzle[square_row_start:square_row_start + 3, square_col_start:square_col_start + 3].flatten()
                missing_numbers_in_square = [n for n in range(1, 10) if n not in square]

                possible_numbers = [
                    n for n in missing_numbers_in_row
                    if n in missing_numbers_in_col and n in missing_numbers_in_square
                ]

                if len(possible_numbers) == 1:
                    new_chromosome[i, j] = possible_numbers[0]
                    cells_filled += 1
        return new_chromosome
    
    def generate_population(self):
      population = []
      for _ in range(self.population_size):
          individual = self.puzzle.copy()
          for i in range(9):
              # Fill only the non-fixed cells in each row with random values
              missing_numbers = [n for n in range(1, 10) if n not in individual[i]]
              np.random.shuffle(missing_numbers)
              individual[i] = [val if self.fixed_cells[i, j] else missing_numbers.pop(0) for j, val in enumerate(individual[i])]
          population.append(individual)
      return population
    
    def calculate_fitness(self, chromosome):
        fitness_score = 0
        for i in range(9):
          col = chromosome[:, i]
          square = chromosome[(i//3)*3:(i//3)*3+3, (i%3)*3:(i%3)*3+3].flatten()
          fitness_score += (9 - len(np.unique(col))) + (9 - len(np.unique(square)))
        return fitness_score
    
    def selection(self, fitness_scores):
      # Sort the population by fitness scores (ascending order)
      sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda x: x[0])]
      
      # Return the top half of the sorted population
      return sorted_population[:len(sorted_population) // 2]
    
    def cross_over(self, parent1, parent2):
      offspring1, offspring2 = parent1.copy(), parent2.copy()
      # Choose a random row to perform crossover
      row = random.randint(0, 8)
      # swap rows
      offspring1[row], offspring2[row] = offspring2[row], offspring1[row]
      return offspring1, offspring2

    def mutate(self, chromosome):
      mutated_chromosome = chromosome.copy()
      
      # Select a random row that has at least two non-fixed cells
      row = random.randint(0, 8)
      non_fixed_indices = [col for col in range(9) if not self.fixed_cells[row, col]]
      
      # Keep selecting rows until we find one with at least two non-fixed cells
      while len(non_fixed_indices) < 2:
          row = random.randint(0, 8)
          non_fixed_indices = [col for col in range(9) if not self.fixed_cells[row, col]]
      
      # Randomly select two different non-fixed cells to swap
      col1, col2 = random.sample(non_fixed_indices, 2)
      
      # Swap the values of the selected cells
      mutated_chromosome[row, col1], mutated_chromosome[row, col2] = (
          mutated_chromosome[row, col2],
          mutated_chromosome[row, col1]
      )
      
      return mutated_chromosome      
    
    def check_can_be_solved_by_swap(self, chromosome):
      wrong_columns = []
      
      # Identify columns that contain duplicates
      for i in range(9):
          col = chromosome[:, i]
          if len(col) != len(np.unique(col)):
              wrong_columns.append(i)

      # If there are not exactly 2 wrong columns, return None (no solution by simple swap)
      if len(wrong_columns) != 2:
          return None

      # Get the indices of the two wrong columns
      col1, col2 = wrong_columns

      # Find two rows with non-fixed, incorrect cells in these columns
      for row in range(9):
          # Check if the two cells in the identified columns are not fixed
          if not self.fixed_cells[row, col1] and not self.fixed_cells[row, col2]:
              # Check if swapping these two cells could potentially solve the problem
              if chromosome[row, col1] != chromosome[row, col2]:  # Only swap if values are different
                  # Create a copy of the chromosome to test the swap
                  test_chromosome = chromosome.copy()
                  test_chromosome[row, col1], test_chromosome[row, col2] = test_chromosome[row, col2], test_chromosome[row, col1]
                  
                  # Check if the swap results in a fitness score of 0
                  if self.calculate_fitness(test_chromosome) == 0:
                      print("Swapped cells at", (row, col1), "and", (row, col2), "to find a solution.")
                      return test_chromosome

      return None

    def solve(self):
      if not self.is_solveable:
        print("The puzzle is not solveable.")
        return

      stagnation_counter = 0
      reset_counter = 0
      mutation_rate = self.mutation_rate

      

      for generation in range(self.max_generations):
        fitness_scores = [self.calculate_fitness(chromosome) for chromosome in self.population]

        if 0 in fitness_scores:
          print("Solution found in generation", generation)
          print(self.population[fitness_scores.index(0)])
          return True

        selected_population = self.selection(fitness_scores)
        self.population = selected_population.copy()
         
        for i in range(0, len(selected_population), 2):
          parent1 = random.choice(selected_population)
          parent2 = random.choice(selected_population)
          offspring1, offspring2 = self.cross_over(parent1, parent2)
          self.population.extend([offspring1, offspring2])
        
        for i in range(len(self.population)):
          if random.random() < mutation_rate:
            self.population[i] = self.mutate(self.population[i])
        
        min_fitness = min(fitness_scores)
      
        if(min_fitness == 2):
          solution = self.check_can_be_solved_by_swap(self.population[fitness_scores.index(min_fitness)])
          if solution is not None:
            print("Solution found in generation", generation)
            print(solution)
            return True

        if self.best_fitness <= min_fitness:
          stagnation_counter += 1
          reset_counter += 1
        else:
          stagnation_counter = 0
          reset_counter = 0
          self.best_fitness = min_fitness

        if stagnation_counter >= self.stagnation_threshold:
          mutation_rate *= self.mutation_rate_factor
          stagnation_counter = 0

        if reset_counter >= self.reset_threshold:
          print("Resetting population.")
          self.population = self.generate_population()
          reset_counter = 0
          stagnation_counter = 0
          mutation_rate = self.mutation_rate
        
        print("Generation", generation, "Fitness:", min_fitness)

      print("Solution not found.")
      return False