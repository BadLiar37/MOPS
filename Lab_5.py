import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
import networkx as nx

# Task1
print("Task1")

# create model
model = LpProblem(name="Task1Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 18)}
# x[1]-F, x12-x[2]
# add constraints
model += (x[2]+x[3]-x[1]                               == 0, "1")
model += (x[2]+x[5]-x[4]-x[7]-x[6]                     == 0, "2")
model += (x[3]+x[4]+x[9]-x[5]-x[8]-x[10]               == 0, "3")
model += (x[6]+x[13]-x[15] -x[14]                      == 0, "4")
model += (x[10]+x[12]-x[11]-x[17]                      == 0, "5")
model += (x[7]+x[14]+x[8]+x[11]-x[12]-x[9]-x[13]-x[16] == 0, "6")
model += (x[15]+x[16]+x[17]-x[1]                       == 0, "7")

model += (x[2]  <= 20, "8")
model += (x[3]  <= 22, "9")
model += (x[4]  <= 4,  "10")
model += (x[5]  <= 2,  "11")
model += (x[7]  <= 10,  "12")
model += (x[8]  <= 10,  "13")
model += (x[9]  <= 10,  "14")
model += (x[6]  <= 12,  "15")
model += (x[13] <= 6,  "16")
model += (x[14] <= 6,  "17")
model += (x[10] <= 14, "18")
model += (x[12] <= 4,  "19")
model += (x[11] <= 4,  "20")
model += (x[17] <= 16,  "21")
model += (x[15] <= 14,  "22")
model += (x[16] <= 4,  "23")

# describe goal
model += x[2]+x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}"+" units bandwidth")
print(f"Max network bandwidth: {model.objective.value()}")

# Task2
print("Task2")

# create model
model = LpProblem(name="Task2Lol1", sense=LpMinimize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 10)}
# x[1]-F, x12-x[2]
# add constraints
model += (x[1]+x[2]               <= 20, "1")
model += (x[1]-x[4]-x[5]-x[3]     == 0, "2")
model += (x[2]+x[3]+x[9]-x[6]-x[7]== 0, "3")
model += (x[4]+x[6]-x[8]          == 5, "4")
model += (x[7]+x[5]+x[8]-x[9]     == 15, "5")

model += (x[1]  <= 15, "8")
model += (x[2]  <= 8, "9")
model += (x[7]  <= 4,  "10")
model += (x[4]  <= 4,  "11")
model += (x[6]  <= 15,  "12")
model += (x[5]  <= 10,  "13")
model += (x[9]  <= 4,  "14")

# describe goal
model += 12*x[1]+12*x[2]+5*x[3]+4*x[4]+9*x[5]+1*x[6]+5*x[7]+12*x[8]+5*x[9]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}"+" $")
print(f"Min delivery cost: {model.objective.value()}$")

# Task3
print("Task3")

# the adjacency matrix of network graph
networkGraph = np.array([[0,24,20, 0, 0, 0, 0],
                         [0, 0, 0,12, 0, 0,30],
                         [0,14, 0,10, 1, 0, 0],
                         [0, 0, 0, 0,15, 0, 7],
                         [0, 0, 0, 0, 0, 8, 0],
                         [0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0]])
# create network graph with using the adjacency matrix
DG = nx.from_numpy_matrix(np.matrix(networkGraph), create_using=nx.DiGraph)
# find shortest route using dijkstra algorithm
print("Critical route of network graph is :",nx.dijkstra_path(DG,0,6))
print("Length of critical route of network graph is:",nx.dijkstra_path_length(DG,0,6))
# make layout
layout = nx.shell_layout(DG)
# get weight of edges
labels = nx.get_edge_attributes(DG, "weight")
# draw weight of edges
nx.draw_networkx_edge_labels(DG,pos=layout, edge_labels=labels)
# draw network graph
nx.draw(DG, node_color='red',node_size=1000,with_labels=True)
# output picture
plt.show()