import sys
import time
import random
from task_scheduling import *
from metaheuristics import *


def main(instance: str, seconds: int, solver: str):
    graph = read_instance(instance)
    if solver == 'GRASP':
        sol_list = GRASP(graph, seconds, 0.3)
        sol = sol_list[0]
    if solver == 'GRASP_INTEN':
        sol_list = GRASP(graph, seconds, 0.3, inten=True)
        sol = sol_list[0]
    if solver == 'GRASP_DIVER':
        sol_list = GRASP(graph, seconds, 0.3, diver=True)
        sol = sol_list[0]
    if solver == 'GRASP_INTEN_DIVER':
        sol_list = GRASP(graph, seconds, 0.3, inten=True, diver=True)
        sol = sol_list[0]

    if solver == 'GA':
        pop_size = 2 * len(graph.tasks)
        if pop_size > 30:
            pop_size = 30
        mutation_rate = 1 / graph.n_tasks
        if mutation_rate > 0.2:
            mutation_rate = 0.2
        sol = GA(graph, seconds, pop_size, mutation_rate=mutation_rate)
    if solver == 'GA_INTEN':
        pop_size = 2 * len(graph.tasks)
        if pop_size > 30:
            pop_size = 30
        mutation_rate = 1 / graph.n_tasks
        if mutation_rate > 0.2:
            mutation_rate = 0.2
        sol = GA(graph, seconds, pop_size, mutation_rate=mutation_rate, inten = True)
    if solver == 'GA_DIVER':
        pop_size = 2 * len(graph.tasks)
        if pop_size > 30:
            pop_size = 30
        mutation_rate = 10 / graph.n_tasks
        if mutation_rate > 0.2:
            mutation_rate = 0.2
        sol = GA(graph, seconds, pop_size, mutation_rate=mutation_rate)
    if solver == 'GA_INTEN_DIVER':
        pop_size = 2 * len(graph.tasks)
        if pop_size > 30:
            pop_size = 30
        mutation_rate = 10 / graph.n_tasks
        if mutation_rate > 0.2:
            mutation_rate = 0.2
        sol = GA(graph, seconds, pop_size, mutation_rate=mutation_rate, inten = True)
    
    if solver == 'GRASP_GA':
        mutation_rate = 10 / graph.n_tasks
        if mutation_rate > 0.2:
            mutation_rate = 0.2
        pop_size = 2 * len(graph.tasks)
        if pop_size > 30:
            pop_size = 30

        sol_list = GRASP(graph, seconds / 2, 0.3, inten=True, diver=True)

        sol_list = list(reversed(sol_list)) # best at the beginning
        sol_list = sol_list[0:pop_size]

        sol = GA(graph, seconds / 2, pop_size, mutation_rate=mutation_rate, pop_from_GRASP=sol_list)

    if not sol.is_complete():
        print(f"Solution for {solver} is not complete")
    if not sol.is_feasible():
        print(f"Solution for {solver} is not feasible")

    return sol


def run(solver, total_seconds, instance):
    # if len(sys.argv) != 4:
        # print("Call the program with a solver name, max running time and instance name")
        # exit(0)
    # _, solver, total_seconds, instance = sys.argv
    random.seed(a=0, version=2)
    start = time.time()
    sol = main(instance, int(total_seconds), solver)
    end = time.time()
    total_time = round(end - start, 3)
    print(f"Instance: {instance}\t Solver: {solver}\t time: {total_time}\t Makespan: {sol.cost()}\t Flowtime: {sol.flowtime}")
    # print(f"Solution task ordering {sol.tasks}")

if __name__ == "__main__":
    sec = 4 * 60
    ver = int(sys.argv[1])
    if(ver == 1):
        run("GRASP", str(sec), "instance1.txt")
        run("GRASP", str(sec), "instance2.txt")
        run("GRASP", str(sec), "instance3.txt")
        run("GRASP", str(sec), "instance4.txt")
        run("GRASP", str(sec), "instance5.txt")
        print()
        run("GRASP_INTEN", str(sec), "instance1.txt")
        run("GRASP_INTEN", str(sec), "instance2.txt")
        run("GRASP_INTEN", str(sec), "instance3.txt")
        run("GRASP_INTEN", str(sec), "instance4.txt")
        run("GRASP_INTEN", str(sec), "instance5.txt")
        print()
        run("GRASP_DIVER", str(sec), "instance1.txt")
        run("GRASP_DIVER", str(sec), "instance2.txt")
        run("GRASP_DIVER", str(sec), "instance3.txt")
        run("GRASP_DIVER", str(sec), "instance4.txt")
        run("GRASP_DIVER", str(sec), "instance5.txt")
    if(ver == 2):
        run("GRASP_INTEN_DIVER", str(sec), "instance1.txt")
        run("GRASP_INTEN_DIVER", str(sec), "instance2.txt")
        run("GRASP_INTEN_DIVER", str(sec), "instance3.txt")
        run("GRASP_INTEN_DIVER", str(sec), "instance4.txt")
        run("GRASP_INTEN_DIVER", str(sec), "instance5.txt")
        print()
        run("GA", str(sec), "instance1.txt")
        run("GA", str(sec), "instance2.txt")
        run("GA", str(sec), "instance3.txt")
        run("GA", str(sec), "instance4.txt")
        run("GA", str(sec), "instance5.txt")
        print()
        run("GA_INTEN", str(sec), "instance1.txt")
        run("GA_INTEN", str(sec), "instance2.txt")
        run("GA_INTEN", str(sec), "instance3.txt")
        run("GA_INTEN", str(sec), "instance4.txt")
        run("GA_INTEN", str(sec), "instance5.txt")
    if(ver == 3):
        run("GA_DIVER", str(sec), "instance1.txt")
        run("GA_DIVER", str(sec), "instance2.txt")
        run("GA_DIVER", str(sec), "instance3.txt")
        run("GA_DIVER", str(sec), "instance4.txt")
        run("GA_DIVER", str(sec), "instance5.txt")
        print()
        run("GA_INTEN_DIVER", str(sec), "instance1.txt")
        run("GA_INTEN_DIVER", str(sec), "instance2.txt")
        run("GA_INTEN_DIVER", str(sec), "instance3.txt")
        run("GA_INTEN_DIVER", str(sec), "instance4.txt")
        run("GA_INTEN_DIVER", str(sec), "instance5.txt")
        print()
        run("GRASP_GA", str(sec), "instance1.txt")
        run("GRASP_GA", str(sec), "instance2.txt")
        run("GRASP_GA", str(sec), "instance3.txt")
        run("GRASP_GA", str(sec), "instance4.txt")
        run("GRASP_GA", str(sec), "instance5.txt")