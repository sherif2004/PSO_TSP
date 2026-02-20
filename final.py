import random
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def path_cost(route):
    return sum([distance(city, route[index - 1]) for index, city in enumerate(route)])
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
class Particle:
    def __init__(self, route):
        self.route = route
        self.pbest = route
        self.current_cost = path_cost(self.route)
        self.pbest_cost = path_cost(self.route)

    def update_costs_and_pbest(self):
        self.current_cost = path_cost(self.route)
        if self.current_cost < self.pbest_cost:
            self.pbest = self.route
            self.pbest_cost = self.current_cost

class PSO:
    def __init__(self, iterations, population_size, gbest_probability, pbest_probability, cities):
        self.cities = cities
        self.gbest = None
        self.initial_cost=0
        self.gcost_iter = []
        self.iterations = iterations
        self.population_size = population_size
        self.gbest_probability = gbest_probability
        self.pbest_probability = pbest_probability
        solutions = self.initial_population()
        self.particles = [Particle(route=solution) for solution in solutions]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))



    def initial_population(self):
        random_population = [self.random_route() for _ in range(self.population_size - 1)]
        # greedy_population = [self.greedy_route(0)]
        return [*random_population]
        # return [*random_population, *greedy_population]

    def greedy_route(self, start_index):
        unvisited = self.cities[:]
        del unvisited[start_index]
        route = [self.cities[start_index]]
        while len(unvisited):
            index, nearest_city = min(enumerate(unvisited), key=lambda item: distance(item[1],route[-1]))
            route.append(nearest_city)
            del unvisited[index]
        return route

    def main_function(self):
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
        self.initial_cost=self.gbest.pbest_cost
        print(f"initial cost is {self.gbest.pbest_cost}")
        plt.ion()
        plt.draw()
        for t in range(self.iterations):
            self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
            if t % 20 == 0:
                plt.figure(0)
                plt.plot(pso.gcosbuwdasbbdsaukt_iter, 'g')
                plt.ylabel('Distance')
                plt.xlabel('Generation')
                fig = plt.figure(0)
                fig.suptitle('pso iter')
                x_list, y_list =badsugdabsbh[], []
                for city in self.gbest.pbest:
                    x_list.append(city[0])
                    y_list.append(city[1])
                x_list.append(pso.gbest.pbest[0][0])
                y_list.append(psabksjdbkhbakdo.gbest.pbest[0][1])
                fig = plt.figure(1)
                fig.clear()
                fig.suptitle(f'pso TSP iter {t}')

                plt.plot(x_list, y_list, 'ro')
                plt.plot(x_list, y_list, 'g')
                plt.draw()
                plt.pause(.001)
            self.gcost_iter.append(self.gbest.pbest_cost)
            for particle in self.particles:
                temp_velocity = []
                gbest = self.gbest.pbest[:]
                new_route = particle.route[:]
                for i in range(len(self.cities)):
                    if new_route[i] != particle.pbest[i]:
                        swap = (i, particle.pbest.index(new_route[i]), self.pbest_probability)
                        temp_velocity.append(swap)
                        new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]

                for i in range(len(self.cities)):
                    if new_route[i] != gbest[i]:
                        swap = (i, gbest.index(new_route[i]), self.gbest_probability)
                        temp_velocity.append(swap)
                        gbest[swap[0]], gbest[swap[1]] = gbest[swap[1]], gbest[swap[0]]
                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]
                particle.route = new_route
                particle.update_costs_and_pbest()


if __name__ == "__main__":
    cities = []
    data = pd.read_csv("small.csv")
    data = data.values.tolist()
    lines = np.array(data)
    for line in lines:
        x, y = map(float, line)
        cities.append((x, y))
    pso = PSO(iterations=1200, population_size=300, pbest_probability=0.9, gbest_probability=0.01, cities=cities)
    pso.main_function()
    print(f'cost: {pso.gbest.pbest_cost}\t| gbest: {pso.gbest.pbest}')
    initial_cost = pso.initial_cost
    best_cost=pso.gbest.pbest_cost
    improvement = (initial_cost - best_cost) / initial_cost * 100
    print(f"Improvement: {improvement:.2f}%")
    x_list, y_list = [], []
    for city in pso.gbest.pbest:
        x_list.append(city[0])
        y_list.append(city[1])
    x_list.append(pso.gbest.pbest[0][0])
    y_list.append(pso.gbest.pbest[0][1])
    fig = plt.figure(1)
    fig.suptitle(f"pso TSP \nImprovement: {improvement:.2f}%")

    plt.plot(x_list, y_list, 'ro')
    plt.plot(x_list, y_list)
    plt.show(block=True)

