import sys
import random
if len(sys.argv) != 4:
  print("ERRO")

_, n_processors, n_tasks, instance = sys.argv
n_processors = int(n_processors)
n_tasks = int(n_tasks)


f = open(instance + ".txt", "w")
f.write(f"{n_processors}\n")
f.write(f"{n_tasks}\n")

f.write("TASKS (ID COST)\n")

for i in range(n_tasks):
  cost = random.randint(1, 5)
  f.write(f"{i} {cost}\n")

f.write("EDGES (TASK TASK COST)\n")

for i in range(6, n_tasks):
  n_of_deps = random.randint(0,5)
  deps = list(range(0, i))
  random.shuffle(deps)
  deps = deps[:n_of_deps]

  cost = random.randint(1, 5)

  for dep in deps:
    f.write(f"{dep} {i} {cost}\n")