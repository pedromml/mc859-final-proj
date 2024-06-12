import sys
import time
import random
from task_scheduling import *
from metaheuristics import *


def main(instance: str, seconds: int, solver: str):
    graph = read_instance(instance)
    if solver == 'GRASP':
        sol = GRASP(graph, seconds, 0.3)
    if solver == 'GRASP_INTEN':
        sol = GRASP(graph, seconds, 0.3, pop_in_construction=True)
    if solver == 'GRASP_DIVER':
        sol = GRASP(graph, seconds, 0.3, diver=True)
    if solver == 'GRASP_INTEN_DIVER':
        sol = GRASP(graph, seconds, 0.3, pop_in_construction=True, diver=True)
    if solver == 'GA':
        sol = GA(graph, seconds, len(graph.tasks))

    if not sol.is_complete():
        print(f"Solution for {solver} is not complete")

    return sol


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Call the program with a solver name, max running time and instance name")
        exit(0)
    _, solver, total_seconds, instance = sys.argv
    random.seed(a=0, version=2)
    start = time.time()
    sol = main(instance, int(total_seconds), solver)
    end = time.time()
    total_time = round(end - start, 3)
    print(f"Instance: {instance}\t time: {total_time}\t Cost: {sol.cost()}")
    print(f"Solution task ordering {sol.tasks}")
