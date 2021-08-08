import random

import matplotlib.pyplot as plt

ONE_MAX_LENGHT = 100  # Длина битовой строки, которую бу отимизировать

POPULATION_SIZE = 200  # Размер популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации
MAX_GENERATION = 50  # Максимальное количество поколений

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class FitnessMax:
    def __init__(self):
        self.values = [0]  # Начальная приспособленность особи


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()  # Значение приспособленности индивида


def one_max_fitness(individual):
    return sum(individual),


def individual_creator():
    return Individual([random.randint(0, 1) for _ in range(ONE_MAX_LENGHT)])


def population_creator(n=0):
    return list([individual_creator() for _ in range(n)])


population = population_creator(n=POPULATION_SIZE)  # Создаем популяцию
generation_counter = 0  # счетчик числа поколений

fitness_values = list(map(one_max_fitness, population))

for individual, fitness_value in zip(population, fitness_values):
    individual.fitness.values = fitness_value

max_fitness_values = []
mean_fitness_values = []


def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind


# турнирная функция, отбираем наиболее приспособленных особей
def sel_tournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring


# Функция скрещивания
def cx_one_point(child1, child2):
    s = random.randint(2, len(child1) - 3)
    child1[s:], child2[s:] = child2[s:], child1[s:]


# Функция мутации
def mut_flip_bit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1


fitness_values = [individual.fitness.values[0] for individual in population]

while max(fitness_values) < ONE_MAX_LENGHT and generation_counter < MAX_GENERATION:
    generation_counter += 1  # Увеличиваем счетчик поколений
    offspring = sel_tournament(population, len(population))  # отбираем особи турнирным отбором
    offspring = list(map(clone, offspring))  # слонируем особи

    # производим скрещивание четной особи и не четной
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cx_one_point(child1, child2)

    # Производим мутацию
    for mutant in offspring:
        if random.random() < P_MUTATION:
            mut_flip_bit(mutant, indpb=1.0 / ONE_MAX_LENGHT)

    fresh_fitness_values = list(map(one_max_fitness, offspring))  # Обновляем значение приспособленности новой популяции

    # Присваиваем новые значения приспособленности новым особям
    for individual, fitness_value in zip(offspring, fresh_fitness_values):
        individual.fitness.values = fitness_value

    population[:] = offspring

    fitness_values = [ind.fitness.values[0] for ind in population]

    max_fitness = max(fitness_values)
    mean_fitness = sum(fitness_values) / len(fitness_values)
    max_fitness_values.append(max_fitness)
    mean_fitness_values.append(mean_fitness)

    print(
        f"Поколение {generation_counter}: "
        f"Макс. приспособелнность - {max_fitness}, "
        f"ср. приспособленность - {mean_fitness}")

    best_index = fitness_values.index(max(fitness_values))
    print("Лучший индивидуум: \n", *population[best_index], "\n")

plt.plot(max_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel("Поколение")
plt.ylabel("Макс/средняя приспособленность")
plt.title("Зависимость максимальной и средней приспособленности от поколения")
plt.show()
