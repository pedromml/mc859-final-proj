import time
import math
from itertools import combinations
import random
from collections import deque
from typing import List, Tuple
from task_scheduling import *
from copy import deepcopy
from functools import partial

def GRASP(g: TASK_SCHEDULING, execution_time: int, alpha: float) -> Solution:
    start_time = time.time()
    total_seconds = 0
    incumbent_solution = None

    def is_every_dependecy_in_solution(task: Task, solution: Solution):
        for dep in g.dependency_list[task.identifier]:
            if dep not in solution.tasks:
                return False
        return True

    def local_search(old_sol: Solution) -> Solution:
        best_sol = deepcopy(old_sol)
        # Troca de duas tasks adjacentes
        for i in range(1, g.n_tasks):
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

        # TODO implementar outro operador

        return best_sol

    def top_level_scheduling():
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

    def constructive_heuristic() -> Solution:

        # Candidate list
        scheduling = top_level_scheduling()

        CL = scheduling
        solution = Solution([] ,g)

        while(len(CL) > 0):
            # Only allow in CL tasks that already have all their dependencies fulfilled
            filtered_CL = []
            for t in CL:
                if is_every_dependecy_in_solution(t, solution):
                    filtered_CL.append(t)
            if filtered_CL is None or len(filtered_CL) == 0:
                print("ERROR: filtered_CL is empty")

            RCL_size = math.ceil(len(CL) * alpha) # TODO ajeitar o cáculo que usa alpha pra incluir o custo das soluções
            RCL = filtered_CL[:RCL_size]
            random.shuffle(RCL)
            element_to_be_added = RCL[0]

            solution.tasks.append(element_to_be_added.identifier)

            CL.remove(element_to_be_added)
        
        return solution

    # Main GRASP loop
    while total_seconds < execution_time:
        new_sol = constructive_heuristic()
        if new_sol is None:
            print("ERROR: Constructive heuristic returned None")

        new_sol = local_search(new_sol)
        if new_sol is None:
            print("ERROR: Local search returned None")

        current_time = time.time()
        total_seconds = round(current_time - start_time, 3)

        if(incumbent_solution is None or new_sol.cost() < incumbent_solution.cost()):
            print(f"{total_seconds} -> {new_sol.cost()}")
            incumbent_solution = new_sol

    return incumbent_solution