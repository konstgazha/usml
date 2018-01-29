# -*- coding: utf-8 -*-
import random
import copy


class Solver_8_queens:
    
    def __init__(self, pop_size=100, cross_prob=0.70, mut_prob=0.015):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.population = []
        self.generation = 0
        self.board_size = 8
        self.goal = 28
        self.goal_reached = False
        self.best_fit = 0
        self.best_individual = None

    def solve(self, min_fitness=1.0, max_epochs=1000):
        self.generate_first_population()
        self.compute_fitness()
        if not min_fitness and not max_epochs:
            min_fitness = 1.0
        while True:
            self.find_best_params()
            if min_fitness:
                self.goal_reached = self.is_goal_reached(min_fitness)
                if self.goal_reached:
                    break
            if max_epochs:
                if self.generation == max_epochs:
                    break
            for i in range(self.pop_size):
                self.population[i].probability = self.get_chromosome_probability(i)
                if i == 0:
                    self.population[i].bottom_edge = 0
                else:
                    self.population[i].bottom_edge = self.population[i - 1].top_edge
                self.population[i].top_edge = self.population[i].bottom_edge + self.population[i].probability
            self.reproduction()
            self.crossover()
            self.mutate()
            self.compute_fitness()
        return self.best_fit / self.goal, self.generation, self.get_visualization()
        

    def crossover(self):
        for i in range(self.pop_size):
            first_parent = random.randint(0, self.pop_size - 1)
            second_parent = random.randint(0, self.pop_size - 1)
            point = random.randint(0, len(self.population[first_parent].cells))
            if self.population[first_parent].cells == self.population[second_parent].cells:
                continue
            if random.uniform(0, 1) > self.cross_prob:
                tmp = self.population[first_parent].cells[-point:]
                self.population[first_parent].cells[-point:] = self.population[second_parent].cells[-point:]
                self.population[second_parent].cells[-point:] = tmp
                self.population[first_parent].child = True
                self.population[second_parent].child = True

    def mutate(self):
        for i in range(self.pop_size):
            if random.uniform(0, 1) > self.cross_prob and self.population[i].child:
                index = random.randint(0, len(self.population[i].cells) - 1)
                self.population[i].cells[index] = 1 - self.population[i].cells[index]
    
    def reproduction(self):
        self.generation += 1
        new_population = []
        while len(new_population) < self.pop_size:
            roulette_choise = random.uniform(0, 1)
            for i in range(self.pop_size):
                if self.population[i].bottom_edge <= roulette_choise <= self.population[i].top_edge:
                    self.population[i].child = False
                    new_population.append(copy.deepcopy(self.population[i]))
        self.population = new_population

    def is_goal_reached(self, min_fitness):
        _goal_reached = False
        for i in range(self.pop_size):
            if self.population[i].fitness >= self.goal * min_fitness:
                _goal_reached = True
        return _goal_reached

    def find_best_params(self):
        for i in range(self.pop_size):
            if self.population[i].fitness > self.best_fit:
                self.best_fit = self.population[i].fitness
                self.best_individual = copy.deepcopy(self.population[i])

    def get_chromosome_probability(self, index):
        return self.population[index].fitness / sum([i.fitness for i in self.population])

    def compute_fitness(self):
        for i in range(self.pop_size):
            self.population[i].cells_to_board()
            self.population[i].compute_fitness()

    def generate_first_population(self):
        for i in range(self.pop_size):
            chessboard = Chessboard(self.board_size, self.goal)
            chessboard.cells_to_board()
            chessboard.randomly_fill_cells()
            chessboard.board_to_cells()
            self.population.append(chessboard)

    def get_visualization(self):
        visualization = ''
        for i in range(len(self.best_individual.board)):
            row = ''
            for j in range(len(self.best_individual.board)):
                if self.best_individual.board[i][j] == 0:
                    row += '+'
                if self.best_individual.board[i][j] == 1:
                    row += 'Q'
            visualization += row + '\n'
        return visualization


class Chessboard:

    def __init__(self, board_size, goal):
        self.board_size = board_size
        self.goal = goal
        self.board = []
        self.digit_positions = []
        self.child = False
        i = 0
        while True:
            if board_size <= 2**i:
                self.cells = [0] * sum([i for _ in range(board_size)])
                self.power = i
                break
            i += 1
        self.randomly_fill_cells()
        
    def cells_to_board(self):
        self.board = []
        for i in range(self.board_size):
            binary_position = self.cells[i*self.power:i*self.power + self.power]
            digit_position = int(''.join(str(digit) for digit in binary_position), 2)
            self.digit_positions.append(digit_position)
            row = [0] * self.board_size
            row[digit_position] = 1
            self.board.append(row)
    
    def board_to_cells(self):
        self.cells = []
        str_number = ''
        for i in self.digit_positions:
            binary_mask = "{0:0" + str(self.power) + "b}"
            str_number += binary_mask.format(i)
        for i in range(len(str_number)):
            self.cells.append(int(str_number[i]))
    
    def randomly_fill_cells(self):
        for i in range(len(self.cells)):
            self.cells[i] = random.randint(0, 1)

    def compute_fitness(self):
        self.fitness = self.goal
        self.collisions = 0
        positive_diag, negative_diag = self.get_diagonals()
        self.collisions += self.diagonal_collisions(positive_diag)
        self.collisions += self.diagonal_collisions(negative_diag)
        self.collisions += self.rectilinear_collisions()
        self.fitness -= self.collisions

    def get_diagonals(self):
        positive_diags = []
        negative_diags = []
        for p in range(self.board_size * 2 - 1):
            pos_diag = []
            neg_diag = []
            for q in range(min(p, self.board_size - 1), max(0, p - self.board_size + 1) - 1, -1):
                pos_diag.append(self.board[q][p - q])
                neg_diag.append(self.board[self.board_size - 1 - q][p - q])
            positive_diags.append(pos_diag)
            negative_diags.append(neg_diag)
        return positive_diags, negative_diags
    
    def rectilinear_collisions(self):
        vertical_queens = []
        horizontal_queens = []
        for i in range(self.board_size):
            col = 0
            row = 0
            for j in range(self.board_size):
                col += self.board[j][i]
                row += self.board[i][j]
            if col > 1:
                vertical_queens.append(col)
            if row > 1:
                horizontal_queens.append(row)
        collisions = self.compute_collision(vertical_queens)
        collisions += self.compute_collision(horizontal_queens)
        return collisions
        
    def diagonal_collisions(self, diag):
        queens_quantities = [k for k in [sum(i) for i in diag] if k > 1]
        return self.compute_collision(queens_quantities)

    def compute_collision(self, queens_quantities):
        diag_collisions = 0
        for i in queens_quantities:
            for j in range(1, i):
                diag_collisions += j
        return diag_collisions        
