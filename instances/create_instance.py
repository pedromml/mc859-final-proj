import sys
if len(sys.argv) != 4:
  print("ERRO")

_, n_processors, n_tasks, instance = sys.argv
n_processors = int(n_processors)
n_tasks = int(n_tasks)

n_total_tasks = 0

for i in range(n_tasks):
  n_total_tasks += n_tasks - i

f = open(instance + ".txt", "w")
f.write(f"{n_processors}\n")
f.write(f"{n_total_tasks}\n")

f.write("TASKS (ID COST)\n")

for i in range(n_total_tasks):
  f.write(f"{i} 3\n")

f.write("EDGES (TASK TASK COST)\n")

line = 0
task_id = 0
while n_tasks > 0:
  ini_id = task_id
  for i in range(n_tasks - 1):
    task_id += 1
    f.write(f"{ini_id} {task_id + n_tasks - 1} 2\n")
    f.write(f"{task_id} {task_id + n_tasks - 1} 2\n")
  task_id += 1
  n_tasks -= 1