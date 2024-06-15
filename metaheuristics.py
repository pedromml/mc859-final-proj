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

def constructive_heuristic(alpha: float, g: TASK_SCHEDULING, inten: bool = False) -> Solution:

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

            # Intensificação: Se inten for True, rodar a busca local durante a construção da solução, quando ela tiver 50% do tamanho total
            if inten and len(solution.tasks) == g.n_tasks // 2:
                solution = local_search(solution, g)
        
        return solution


def GRASP(g: TASK_SCHEDULING, execution_time: int, alpha: float, inten: bool = False, diver: bool = False) -> Solution:
    start_time = time.time()
    total_seconds = 0
    incumbent_solution = None

    incumbent_solution = constructive_heuristic(alpha, g, inten)
    pop_for_GA = deque([], 10)
    pop_for_GA.appendleft(incumbent_solution)

    # Main GRASP loop
    iterations_on_same_incumb = 0
    new_alpha = alpha

    itera = 0
    while total_seconds < execution_time:
        itera+=1
        
        # Estratégia de diversificação: Se ficar preso na mesma incumb por muito tempo, aumenta alpha
        if iterations_on_same_incumb == 0:
            new_alpha = alpha
        if iterations_on_same_incumb > 50:
            new_alpha += 1
            if new_alpha > 1: new_alpha = 1

        new_sol = constructive_heuristic(new_alpha, g, inten)
        if new_sol is None:
            print("ERROR: Constructive heuristic returned None")

        new_sol = local_search(new_sol, g)
        if new_sol is None:
            print("ERROR: Local search returned None")

        current_time = time.time()
        total_seconds = round(current_time - start_time, 3)

        if(incumbent_solution is None or new_sol.cost() < incumbent_solution.cost()):
            # print(f"{total_seconds} -> makespan {new_sol.cost()} flowtime {new_sol.flowtime}")
            incumbent_solution = new_sol
            iterations_on_same_incumb = 0
            
            pop_for_GA.appendleft(new_sol)
        elif diver:
            iterations_on_same_incumb += 1

    return list(pop_for_GA)

def GA(g: TASK_SCHEDULING, execution_time: int, pop_size: float, pop_from_GRASP: List[Solution] = None, mutation_rate = 0.01, inten: bool = False, diver: bool = False) -> Solution:
    start_time = time.time()
    total_seconds = 0
    incumbent_solution = None

    population = []

    def order_by_fitness(sol: Solution):
        return sol.cost()

    def print_pop(pop):
        pop2 = pop[:]
        pop2.sort(key=order_by_fitness)
        for i in range(len(pop2)):
            print(f"elem {i} fit {pop2[i].cost()}", pop2[i].tasks)

    def initialize_pop(pop: List[Solution]):
        nonlocal population
        if pop == None:
            pop = []
        population = pop
        while(len(population) < pop_size):
            a = constructive_heuristic(1, g)
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
        # Ordered crossover
        new_pop = pop[:]
        # select parents from parent list
        for par_i in range(0, len(new_pop), 2):
            if(par_i+1 >= len(new_pop)):
                continue
                print(f"ERROR: Problem in indexing parent in crossover pop size = {len(new_pop)} index = {par_i +1}")
            par1 = new_pop[par_i]
            par2 = new_pop[par_i+1]

            # initialize offspring
            of1 = Solution([-1 for _ in range(g.n_tasks)], g)
            of2 = Solution([-1 for _ in range(g.n_tasks)], g)

            # choose crossover points
            indexes = random.sample(range(0, g.n_tasks), 2)
            indexes.sort()

            # copy tasks between indexes from each parent to the repesctive offspring
            of1.tasks[indexes[0]:indexes[1]] = par1.tasks[indexes[0]:indexes[1]]
            of2.tasks[indexes[0]:indexes[1]] = par2.tasks[indexes[0]:indexes[1]]

            # get the tasks that are not in each offspring in the order in which they appear in par2
            of1_remaining_tasks = []
            for i in range(len(par2.tasks)):
                if(par2.tasks[i] not in of1.tasks):
                    of1_remaining_tasks.append(par2.tasks[i])

            of2_remaining_tasks = []
            for i in range(len(par1.tasks)):
                if(par1.tasks[i] not in of2.tasks):
                    of2_remaining_tasks.append(par1.tasks[i])

            for of, rem_t_list in [(of1, of1_remaining_tasks), (of2, of2_remaining_tasks)]:
                while(len(rem_t_list)) > 0:
                    t = rem_t_list.pop(0)
                    t_dep = g.dependency_list[t]
                    for i in range(len(of.tasks)):
                        if of.tasks[i] != -1:
                            # already occupied
                            continue

                        current_tasks = of.tasks[0:i]

                        for dep in t_dep:
                            if dep not in current_tasks:
                                # dependencies not met for t
                                # got to next i
                                i += 1
                                if i >= len(of.tasks):
                                    print("ERROR: Cant find a place for task in crossover")
                        
                        # check if it is not occupied again
                        if of.tasks[i] != -1:
                            # already occupied
                            continue

                        of.tasks[i] = t
                        break
                    
            # here, offsprings should be complete and valid
            if not of1.is_complete():
                print("ERROR: Offspring 1 is not complete")
                exit(1)
            if not of2.is_complete():
                print("ERROR: Offspring 2 is not complete")
                exit(1)

            of1.make_feasible()
            of2.make_feasible()

            if of1.is_feasible() and not is_sol_already_in_pop(of1, pop):
                new_pop[par_i] = of1
            if of2.is_feasible() and not is_sol_already_in_pop(of1, pop):
                new_pop[par_i + 1] = of2

        return new_pop

    def mutation(pop: List[Solution]):
        new_pop = pop[:]
        for j in range(len(new_pop)):
            chrom = new_pop[j]
            number_of_mutations = 0
            for _ in range(g.n_tasks):
                # apply mutation if random is < than mutation rate for each 
                if random.random() < mutation_rate:
                    number_of_mutations += 1
            for _ in range(number_of_mutations):
                # try n times
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

    def is_sol_already_in_pop(sol, pop):
        for i in range(len(pop)):
            if sol.tasks == pop[i].tasks:
                return True
        return False

    def check_how_many_similar_sol_in_two_pop(pop1, pop2):
        pop1_tl = [sol.tasks for sol in pop1]
        pop2_tl = [sol.tasks for sol in pop2]
        total_siml = 0
        for i in range(pop_size):
            for j in range(pop_size):
                if pop1_tl[i] == pop2_tl[j]:
                    total_siml += 1
                    break
        return total_siml
    
    def is_same_pop(pop1, pop2):
        pop1_tl = [sol.tasks for sol in pop1]
        pop2_tl = [sol.tasks for sol in pop2]
        for i in range(pop_size):
            if pop1_tl[i] != pop2_tl[i]:
                return False

        return True

    initialize_pop(pop_from_GRASP)
    incumbent_solution = get_best_chromossome(population)
    # print(f"{total_seconds} -> makespan {incumbent_solution.cost()} flowtime {incumbent_solution.flowtime}")

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

        # intensification
        if(inten):
            population.sort(key= order_by_fitness, reverse=True)
            new_pop.sort(key=order_by_fitness)
            new_pop[int(0.2*pop_size) : len(new_pop)] = population[int(0.2*pop_size) : len(population)]

        best_sol = get_best_chromossome(new_pop)
        worst_sol = get_worst_chromossome(new_pop)

        if best_sol is not None:
            if incumbent_solution is None or best_sol.cost() < incumbent_solution.cost():
                # print(f"{total_seconds} -> makespan {best_sol.cost()} flowtime {best_sol.flowtime}")
                incumbent_solution = best_sol
            else: # Se new_pop só tiver solutions piores que incumbent_solution, add incumb em new_pop
                if worst_sol is not None:
                    new_pop.remove(worst_sol)
                    new_pop.append(incumbent_solution)

    return incumbent_solution