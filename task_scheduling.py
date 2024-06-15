from typing import Dict, List, Self, Tuple
from dataclasses import dataclass
from itertools import takewhile, permutations, pairwise
import math


@dataclass
class Task:
    identifier: int
    cost: int

@dataclass
class TASK_SCHEDULING:
    n_processors: int
    n_tasks: int
    tasks: List[Task]
    edges: Dict[Tuple[int, int], int]
    dependency_list: List[List[int]]

    def print_edges(self):
        for e in self.edges:
            print(e)

    def print_dep_list(self):
        for e in self.dependency_list:
            print(e)

@dataclass
class Solution:
    tasks: List[int]
    g: TASK_SCHEDULING
    flowtime: int = -1
    makespan: int = -1
    time: float = 0.0

    def cost(self):
        scheduling = [[] for i in range(self.g.n_processors)]
        for i in range(len(self.tasks)):
            p = i % self.g.n_processors
            scheduling[p].append(self.tasks[i])
        
        task_finishing_time = {}
        def order_by_tft(t: int):
            return task_finishing_time[t]
        
        current_total_cost = 0
        time_taken_per_processor = [0 for i in range(self.g.n_processors)]

        for i in range(len(self.tasks)):
            if i > len(self.tasks) -1:
                print("Opaaa")
                print(f"Custo: checando task {i}")
                break
            task = self.tasks[i]
            p = i % self.g.n_processors

            deps = self.g.dependency_list[task][:]
            deps.sort(key = order_by_tft)

            # supõe que dep já terminou por causa da ordenação
            itera = 0
            for dep in deps:
                if itera >= self.g.n_tasks + 2:
                    print("Loop infinito")
                    print(f"Custo: checando dep {dep}")
                    break
                if not dep in scheduling[p]:
                    # waiting time
                    if task_finishing_time[dep] > time_taken_per_processor[p]:
                        time_taken_per_processor[p] += task_finishing_time[dep] - time_taken_per_processor[p]
                    # communication time
                    time_taken_per_processor[p] += self.g.edges[(dep, task)]

            # execution time
            time_taken_per_processor[p] += self.g.tasks[task].cost
            task_finishing_time[task] = time_taken_per_processor[p]

        makespan = max(time_taken_per_processor)

        flowtime = 0
        for val in task_finishing_time.values():
            flowtime += val
        self.flowtime = flowtime

        self.makespan = makespan

        return makespan

    def is_feasible(self):
        used_tasks = []
        for t in self.tasks:
            if t not in used_tasks:
                for dep in self.g.dependency_list[t]:
                    if dep not in used_tasks:
                        return False
                used_tasks.append(t)
        return True

    def is_complete(self):
        # Has every task in the solution
        used_tasks = []
        for t in self.tasks:
            if t not in used_tasks:
                used_tasks.append(t)

        return len(used_tasks) == self.g.n_tasks

    def make_feasible(self):
        used_tasks = []
        for i in range(len(self.tasks)):
            t = self.tasks[i]
            if t not in used_tasks:
                for dep in self.g.dependency_list[t]:
                    if dep not in used_tasks:
                        index_t = self.tasks.index(t)
                        index_dep = self.tasks.index(dep)
                        self.tasks[index_t] = dep
                        self.tasks[index_dep] = t
                        used_tasks = []
                        i = 0
                        continue
                    
                used_tasks.append(t)


def read_instance(name: str):
    n_processsors = 0
    n_tasks = 0
    tasks = []
    edges = {}
    dependency_list = []

    def in_tasks_section(line):
        return 'EDGES' not in line
    def in_edges_section(line):
        return 'END' not in line
    
    def skip_lines(n, iterable):
        for _ in range(n):
            next(iterable)

    with open(f"instances/{name}", "r") as file:
        val = next(file)
        n_processors = int(val)
        val = next(file)
        n_tasks = int(val)

        dependency_list = [[] for _ in range(n_tasks)]

        skip_lines(1, file)
        for line in takewhile(in_tasks_section, file):
            identifier, cost = [int(v) for v in line.split()]
            tasks.append(Task(identifier, cost))
        for line in takewhile(in_edges_section, file):       
            left, right, cost = [int(v) for v in line.split()]
            edges[(left, right)] = cost
            dependency_list[right].append(left)


    return TASK_SCHEDULING(n_processors, n_tasks, tasks, edges, dependency_list)
