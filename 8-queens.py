import numpy as np
import random

class Queens(object):

    # Initialize the instance variables
    def __init__(self, board_size=4, population_size=10, mutation_rate=0.025, verbose=True):
        self.board_size = board_size
        self.population_size = population_size
        self.population = None
        self.solution = None
        self.scores = None
        self.target_score = self.get_target_score()
        self.mutation_rate = mutation_rate

    # Generate initial population
    def initial_population(self):
        queens = np.array(range(0, self.board_size))
        population = []
        for _ in range(0, self.population_size):
            np.random.shuffle(queens)
            population.append(np.copy(queens))
        return population

    # Calculate the score for each placement
    def fitness(self, placement):
        score = 0
        y=0
        while y < self.board_size:
            x = placement[y]
            y1 = y+1
            
            while y1 < self.board_size:
                x1 = placement[y1]
                # same row, same column, diagonal
                if (x == x1) or (y == y1) or ((x + y1) == (y + x1)) or ((x + y) == (x1 + y1)):
                    pass
                else:
                    score = score + 1
                y1 = y1+1
            y = y+1
        return score

    # Calculate target score for validation
    def get_target_score(self):
        n = self.board_size - 1
        target_score = n * (n+1) / 2
        return target_score

    # Evaluate the population
    def evaluate_population(self, population=None):
        scores = []
        if not population:
            population = self.population

        for queens in population:
            scores.append(self.fitness(queens))
        return scores

    def select_individual_by_tournament(self):
        
        # Select 2 random populations and return best of them
        pop1 = random.randint(0, self.population_size - 1)
        pop2 = random.randint(0, self.population_size - 1)
        while pop2 == pop1:
            pop2 = random.randint(0, self.board_size - 1)

        if self.scores[pop1] >= self.scores[pop2]:
            selected = pop1
        else:
            selected = pop2
        return np.copy(self.population[selected])

    def crossover_breed(self, parent1, parent2):

        # Set crossover edge, avoid ends
        crossover_edge = random.randint(1, self.board_size - 2)

        # Create sub populations
        pop1 = np.hstack((parent1[0: crossover_edge], parent2[crossover_edge:]))
        pop2 = np.hstack((parent2[0: crossover_edge], parent1[crossover_edge:]))

        return pop1, pop2

    def mutate_population(self, population, mutation_probability=0.025):

        mutation_count = int(self.population_size * mutation_probability)

        for _ in range(0, mutation_count):
            swap_idx_start = random.randint(0, self.population_size - 1)
            swap_idx1 = random.randint(0, self.board_size - 1)
            swap_idx2 = random.randint(0, self.board_size - 1)
            
            # Swap the positions
            population[swap_idx_start][swap_idx1], population[swap_idx_start][swap_idx2] = \
                population[swap_idx_start][swap_idx2], population[swap_idx_start][swap_idx1]

    def __call__(self):
        print("Target Score is: {}".format(self.target_score))

        generations = 100
        selection_size = 20
        best_placement = []

        for _ in range(1, selection_size):
            # Start from randomness
            self.population = self.initial_population()

            for generation_index in range(0, generations):
                self.scores = self.evaluate_population()
                print("Generation {} score: {}, Scores: {}".format(generation_index, sum(self.scores), self.scores))

                # sort population based on scores
                sorted_population = sorted(zip(self.scores, self.population), key=lambda x: x[0], reverse=True)

                # check for target reached
                if sorted_population[0][0] >= self.target_score:
                    target_solution = sorted_population[0][1]
                    print("\nFinal Solution: {}".format(target_solution))
                    return target_solution

                # update best solution so far
                if not best_placement or best_placement[0] < sorted_population[0][0]:
                    best_placement = sorted_population[0]

                # Crossover breeding
                new_population = []
                while len(new_population) < self.population_size:
                    parent1 = self.select_individual_by_tournament()
                    parent2 = self.select_individual_by_tournament()
                    child1, child2 = self.crossover_breed(parent1, parent2)
                    new_population.append(child1)
                    new_population.append(child2)

                # Mutate the population
                self.mutate_population(population=new_population, mutation_probability=self.mutation_rate)

                # Get scores of new population
                new_scores = self.evaluate_population(population=new_population)
                min_new_pop_score = min(new_scores)

                # if there are any better scores in old population, then keep those individuals in new population
                for score, individual in sorted_population:
                    if score > min_new_pop_score:
                        if (new_population == individual).all(1).any():
                            # This individual is already present
                            pass
                        else:
                            new_population.append(individual)
                            new_scores.append(score)

                # select top scores from new population for gen next keeping same population count
                sorted_new_population = sorted(zip(new_scores, new_population), key=lambda x: x[0], reverse=True)
                next_gen_population = []
                for i in range(0, self.population_size):
                    next_gen_population.append(sorted_new_population[i][1])

                # now we have got our next generation which is better than previous generation
                self.population = next_gen_population

        # Could not reach final solution, but lets see what is the best solution we hav got
        print("\nBest Solution: {}".format(best_placement[1]))
        return None


if __name__ == "__main__":
    eight_queens = Queens(8, 50, 0.05)
    eight_queens()
    from datetime import date
    print(date.today())