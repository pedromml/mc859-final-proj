import time
import math
from itertools import combinations
import random
from collections import deque
from typing import List, Tuple
from task_scheduling import *
from copy import deepcopy
from functools import partial

def is_every_dependecy_in_solution(task: Task, solution: Solution, g: TASK_SCHEDULING):
        for dep in g.dependency_list[task.identifier]:
            if dep not in solution.tasks:
                return False
        return True

def top_level_scheduling(g: TASK_SCHEDULING):
    tl = [-1 for _ in range(g.n_tasks)]

    def rec_top_level(identifier: int, tl: List[int]):
        if tl[identifier] != -1:
            return

        pred = g.dependency_list[identifier]
        if len(pred) == 0:
            tl[identifier] = 0
            return

        for dep in pred:
            if tl[dep] == -1:
                rec_top_level(dep, tl)

        tl_possibilities = [g.tasks[dep].cost + tl[dep] + g.edges[(dep, identifier)] for dep in pred]

        tl[identifier] = max(tl_possibilities)

        return
    
    for i in range(1, g.n_tasks):
        rec_top_level(i, tl)

    def order_by_tl(t: Task):
        return tl[t.identifier]
    
    solution_list = g.tasks[:]
    solution_list.sort(key = order_by_tl)

    return solution_list

def local_search(old_sol: Solution, g: TASK_SCHEDULING) -> Solution:
    best_sol = deepcopy(old_sol)
    # Troca de duas tasks adjacentes
    for i in range(1, len(old_sol.tasks)):
        t1 = old_sol.tasks[i - 1]
        t2 = old_sol.tasks[i]
        task_list = old_sol.tasks[:]
        # Só trocar se t1 não for dependência de t2
        if t1 in g.dependency_list[t2]:
            continue
        task_list[i] = t1
        task_list[i - 1] = t2
        new_sol = Solution(task_list, g)
        if new_sol.cost() < best_sol.cost():
            best_sol = new_sol

    return best_sol

def constructive_heuristic(alpha: float, g: TASK_SCHEDULING, pop_in_construction: bool = False) -> Solution:

        # Candidate list
        scheduling = top_level_scheduling(g)

        CL = scheduling
        solution = Solution([] ,g)

        while(len(CL) > 0):
            # Only allow in CL tasks that already have all their dependencies fulfilled
            
            filtered_CL = list(filter(partial(is_every_dependecy_in_solution, solution=solution, g=g), CL))
            if filtered_CL is None or len(filtered_CL) == 0:
                print("ERROR: filtered_CL is empty")

            aux_sol_list = [Solution(solution.tasks + [e.identifier], g) for e in filtered_CL]
            aux_sol_list.sort(key = lambda x : x.cost())

            cost_threshold = aux_sol_list[0].cost() + alpha * (aux_sol_list[-1].cost() - aux_sol_list[0].cost()) 

            def filter_aux_RCL(sol: Solution):
                return sol.cost() <= cost_threshold

            RCL = list(filter(filter_aux_RCL, aux_sol_list))

            random.shuffle(RCL)
            element_to_be_added = RCL[0].tasks[-1]

            solution = RCL[0]
     
            CL.remove(g.tasks[element_to_be_added])

            # Intensificação: Se pop_in_construction for True, rodar a busca local durante a construção da solução, quando ela tiver 50% do tamanho total
            if pop_in_construction and len(solution.tasks) == g.n_tasks // 2:
                solution = local_search(solution, g)
        
        return solution


def GRASP(g: TASK_SCHEDULING, execution_time: int, alpha: float, pop_in_construction: bool = False, diver: bool = False) -> Solution:
    start_time = time.time()
    total_seconds = 0
    incumbent_solution = None



    incumbent_solution = constructive_heuristic(alpha, g, pop_in_construction)

    # Main GRASP loop
    iterations_on_same_incumb = 0
    new_alpha = alpha

    while total_seconds < execution_time:
        # Estratégia de diversificação: Se ficar preso na mesma incumb por muito tempo, aumenta alpha
        if iterations_on_same_incumb == 0:
            new_alpha = alpha
        if iterations_on_same_incumb > 50:
            new_alpha += 1
            if new_alpha > 1: new_alpha = 1

        new_sol = constructive_heuristic(new_alpha, g, pop_in_construction)
        if new_sol is None:
            print("ERROR: Constructive heuristic returned None")

        new_sol = local_search(new_sol, g)
        if new_sol is None:
            print("ERROR: Local search returned None")

        current_time = time.time()
        total_seconds = round(current_time - start_time, 3)

        if(incumbent_solution is None or new_sol.cost() < incumbent_solution.cost()):
            print(f"{total_seconds} -> {new_sol.cost()}")
            incumbent_solution = new_sol
            iterations_on_same_incumb = 0
        elif diver:
            iterations_on_same_incumb += 1

    return incumbent_solution

def GA(g: TASK_SCHEDULING, execution_time: int, pop_size: float, pop_from_GRASP: List[Solution] = None) -> Solution:
    start_time = time.time()
    total_seconds = 0
    incumbent_solution = None

    population = []

    def initialize_pop(pop: List[Solution]):
        nonlocal population
        if pop is not None:
            population = pop
            return
        else:
            for i in range(pop_size):
                a = constructive_heuristic(0.3, g)
                population.append(a)

    def tournament_selection(pop: List[Solution]):
        parents = []
        while(len(parents) < pop_size):
            # Select two random pairs of chromossomes in pop, take the best from each and save them in parents
            aux_pop = pop[:]
            random.shuffle(aux_pop)
            parent1 = aux_pop[0]
            if aux_pop[1].cost() < parent1.cost():
                parent1 = aux_pop[1]
            parent2 = aux_pop[2]
            if aux_pop[3].cost() < parent2.cost():
                parent2 = aux_pop[3]
            parents.append(parent1)
            parents.append(parent2)
        
        # Trims parents if it has more items than pop_size
        parents = parents[:pop_size]
        return parents

    def get_best_chromossome(pop: List[Solution]):
        best_sol = pop[0]
        for e in pop:
            if e.cost() < best_sol.cost():
                best_sol = e
        return best_sol

    def get_worst_chromossome(pop: List[Solution]):
        worst_sol = pop[0]
        for e in pop:
            if e.cost() > best_sol.cost():
                worst_sol = e
        return worst_sol

    def crossover(pop: List[Solution]):
        return pop

    def mutation(pop: List[Solution]):
        new_pop = pop[:]
        for j in range(len(new_pop)):
            chrom = new_pop[j]
            for i in range(g.n_tasks):
                indexes = random.sample(range(0, g.n_tasks), 2)
                t1 = chrom.tasks[indexes[0]]
                t2 = chrom.tasks[indexes[1]]
                tasklist = chrom.tasks[:]
                tasklist[indexes[0]] = t2
                tasklist[indexes[1]] = t1
                new_sol = Solution(tasklist, g)
                if new_sol.is_feasible():
                    new_pop[j] = new_sol
                    break

        return new_pop
        
    initialize_pop(pop_from_GRASP)
    incumbent_solution = get_best_chromossome(population)
    print(f"{total_seconds} -> {incumbent_solution.cost()}")

    # Main GA loop
    while total_seconds < execution_time:
        parents = tournament_selection(population)
        if parents is None or len(parents) == 0:
            print("ERROR: tournament_selection returned None")
        offspring = crossover(parents)
        if offspring is None or len(offspring) == 0:
            print("ERROR: crossover returned None")
        mutations = mutation(offspring)
        if mutations is None or len(mutations) == 0:
            print("ERROR: mutation returned None")
        new_pop = mutations

        if len(new_pop) != pop_size:
            print("ERROR: new_pop has ", len(new_pop), " elements, but pop_size is ", pop_size)

        current_time = time.time()
        total_seconds = round(current_time - start_time, 3)

        best_sol = get_best_chromossome(new_pop)
        worst_sol = get_worst_chromossome(new_pop)

        if best_sol is not None:
            if incumbent_solution is None or best_sol.cost() < incumbent_solution.cost():
                
                print(f"{total_seconds} -> {best_sol.cost()}")
                incumbent_solution = best_sol
            else: # Se new_pop só tiver solutions piores que incumbent_solution, add incumb em new_pop
                if worst_sol is not None:
                    new_pop.remove(worst_sol)
                    new_pop.append(incumbent_solution)

    return incumbent_solution