import random

from deap import base, algorithms
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

ONE_MAX_LENGHT = 100  # Длина битовой строки, которую бу отимизировать

POPULATION_SIZE = 200  # Размер популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации
MAX_GENERATION = 50  # Максимальное количество поколений

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Начальная приспособленность особи
creator.create("Individual", list, fitness=creator.FitnessMax)  # Значение приспособленности индивида


def one_max_fitness(individual):
    return sum(individual), #кортеж, требование пакета DEAP

toolbox = base.Toolbox()
toolbox.register("zero_or_one", random.randint, 0, 1)
toolbox.register("individual_creator", tools.initRepeat, creator.Individual, toolbox.zero_or_one, ONE_MAX_LENGHT)
toolbox.register("population_creator", tools.initRepeat, list, toolbox.individual_creator)

population = toolbox.population_creator(n=POPULATION_SIZE)

generation_counter = 0  # счетчик числа поколений

fitness_values = list(map(one_max_fitness, population))

for individual, fitness_value in zip(population, fitness_values):
    individual.fitness.values = fitness_value

max_fitness_values = []
mean_fitness_values = []


toolbox.register("evaluate", one_max_fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGHT)

population, logbook = algorithms.eaSimple(population, toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATION,
                                          verbose=True)


# plt.plot(max_fitness_values, color='red')
# plt.plot(mean_fitness_values, color='green')
# plt.xlabel("Поколение")
# plt.ylabel("Макс/средняя приспособленность")
# plt.title("Зависимость максимальной и средней приспособленности от поколения")
# plt.show()
