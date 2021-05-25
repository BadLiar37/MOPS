import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
import networkx as nx
#Task1
print("Task1")
# the adjacency matrix of network graph
networkGraph = np.array([[0, 9, 11, 15, 0, 0, 0, 0],
                         [0, 0, 0, 6, 11, 0, 0, 0],
                         [0, 0, 0, 4, 0, 13, 0, 0],
                         [0 ,0, 0, 0, 7, 5, 10, 0],
                         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,17],
                         [0 ,0 ,0 ,0 ,0 ,0 ,3 ,12],
                         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,9 ],
                         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]])
# create network graph with using the adjacency matrix
DG = nx.from_numpy_matrix(np.matrix(networkGraph), create_using=nx.DiGraph)
print("Critical route of network graph is :",nx.dag_longest_path(DG))
print("Length of critical route of network graph is:",nx.dag_longest_path_length(DG))
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

# Task2
print("Task2")
attachments=np.array([5,10,25,50])
timeSpent=np.zeros(4)
T12N,T13N=0,0
# Part1
print("Part1")
# create model
model = LpProblem(name="Task2Lol1", sense=LpMinimize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 21)}
# add constraints
model += (x[15] + x[16] + x[17] + x[18] +x [19] + x[20] <= attachments[0], "First")
model += (x[1]  - T12N  >= 6,  "1")
model += (x[2]  - T13N  >= 12, "2")
model += (x[4]  - x[3]  >= 5,  "3")
model += (x[6]  - x[5]  >= 6,  "4")
model += (x[8]  - x[7]  >= 0,  "5")
model += (x[10] - x[9]  >= 10, "6")
model += (x[12] - x[11] >= 4,  "7")
model += (x[14] - x[13] >= 0,  "8")

model += (x[1]  - T12N  == 10*(1-0.02*x[15]),  "9")
model += (x[2]  - T13N  == 20*(1-0.04*x[16]), "10")
model += (x[4]  - x[3]  == 12*(1-0.03*x[17]),  "11")
model += (x[6]  - x[5]  == 14*(1-0.06*x[18]),  "12")
model += (x[10] - x[9]  == 16*(1-0.05*x[19]), "13")
model += (x[12] - x[11] == 6*(1-0.01*x[20]),  "14")

model += (x[3]  >= x[1] ,  "15")
model += (x[5]  >= x[1] ,  "16")
model += (x[9]  >= x[2] ,  "17")
model += (x[9]  >= x[4] ,  "18")
model += (x[7]  >= x[2] ,  "19")
model += (x[7]  >= x[4] ,  "20")
model += (x[11] >= x[6] ,  "21")
model += (x[11] >= x[8] ,  "22")
model += (x[13] >= x[10],  "23")
model += (x[13] >= x[12],  "24")

# describe goal
model += x[14]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
counter=0
for var in x.values():
    counter=counter+1
    if(counter==14):
        timeSpent[0]=var.value()
    if (counter < 15):
        print(f"{var.name}: {var.value()}"+" days")
        continue
    print(f"{var.name}: {var.value()}"+" $")

# Part2
print("Part2")
# create model
model = LpProblem(name="Task2Lol2", sense=LpMinimize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 21)}
# add constraints
model += (x[15] + x[16] + x[17] + x[18] +x [19] + x[20] <= attachments[1], "First")
model += (x[1]  - T12N  >= 6,  "1")
model += (x[2]  - T13N  >= 12, "2")
model += (x[4]  - x[3]  >= 5,  "3")
model += (x[6]  - x[5]  >= 6,  "4")
model += (x[8]  - x[7]  >= 0,  "5")
model += (x[10] - x[9]  >= 10, "6")
model += (x[12] - x[11] >= 4,  "7")
model += (x[14] - x[13] >= 0,  "8")

model += (x[1]  - T12N  == 10*(1-0.02*x[15]),  "9")
model += (x[2]  - T13N  == 20*(1-0.04*x[16]), "10")
model += (x[4]  - x[3]  == 12*(1-0.03*x[17]),  "11")
model += (x[6]  - x[5]  == 14*(1-0.06*x[18]),  "12")
model += (x[10] - x[9]  == 16*(1-0.05*x[19]), "13")
model += (x[12] - x[11] == 6*(1-0.01*x[20]),  "14")

model += (x[3]  >= x[1] ,  "15")
model += (x[5]  >= x[1] ,  "16")
model += (x[9]  >= x[2] ,  "17")
model += (x[9]  >= x[4] ,  "18")
model += (x[7]  >= x[2] ,  "19")
model += (x[7]  >= x[4] ,  "20")
model += (x[11] >= x[6] ,  "21")
model += (x[11] >= x[8] ,  "22")
model += (x[13] >= x[10],  "23")
model += (x[13] >= x[12],  "24")

# describe goal
model += x[14]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
counter=0
for var in x.values():
    counter=counter+1
    if(counter==14):
        timeSpent[1]=var.value()
    if (counter < 15):
        print(f"{var.name}: {var.value()}"+" days")
        continue
    print(f"{var.name}: {var.value()}"+" $")

# Part3
print("Part3")
# create model
model = LpProblem(name="Task2Lol3", sense=LpMinimize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 21)}
# add constraints
model += (x[15] + x[16] + x[17] + x[18] +x [19] + x[20] <= attachments[2], "First")
model += (x[1]  - T12N  >= 6,  "1")
model += (x[2]  - T13N  >= 12, "2")
model += (x[4]  - x[3]  >= 5,  "3")
model += (x[6]  - x[5]  >= 6,  "4")
model += (x[8]  - x[7]  >= 0,  "5")
model += (x[10] - x[9]  >= 10, "6")
model += (x[12] - x[11] >= 4,  "7")
model += (x[14] - x[13] >= 0,  "8")

model += (x[1]  - T12N  == 10*(1-0.02*x[15]),  "9")
model += (x[2]  - T13N  == 20*(1-0.04*x[16]), "10")
model += (x[4]  - x[3]  == 12*(1-0.03*x[17]),  "11")
model += (x[6]  - x[5]  == 14*(1-0.06*x[18]),  "12")
model += (x[10] - x[9]  == 16*(1-0.05*x[19]), "13")
model += (x[12] - x[11] == 6*(1-0.01*x[20]),  "14")

model += (x[3]  >= x[1] ,  "15")
model += (x[5]  >= x[1] ,  "16")
model += (x[9]  >= x[2] ,  "17")
model += (x[9]  >= x[4] ,  "18")
model += (x[7]  >= x[2] ,  "19")
model += (x[7]  >= x[4] ,  "20")
model += (x[11] >= x[6] ,  "21")
model += (x[11] >= x[8] ,  "22")
model += (x[13] >= x[10],  "23")
model += (x[13] >= x[12],  "24")

# describe goal
model += x[14]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
counter=0
for var in x.values():
    counter=counter+1
    if(counter==14):
        timeSpent[2]=var.value()
    if (counter < 15):
        print(f"{var.name}: {var.value()}"+" days")
        continue
    print(f"{var.name}: {var.value()}"+" $")

# Part4
print("Part4")
# create model
model = LpProblem(name="Task2Lol4", sense=LpMinimize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 21)}
# add constraints
model += (x[15] + x[16] + x[17] + x[18] +x [19] + x[20] <= attachments[3], "First")
model += (x[1]  - T12N  >= 6,  "1")
model += (x[2]  - T13N  >= 12, "2")
model += (x[4]  - x[3]  >= 5,  "3")
model += (x[6]  - x[5]  >= 6,  "4")
model += (x[8]  - x[7]  >= 0,  "5")
model += (x[10] - x[9]  >= 10, "6")
model += (x[12] - x[11] >= 4,  "7")
model += (x[14] - x[13] >= 0,  "8")

model += (x[1]  - T12N  == 10*(1-0.02*x[15]),  "9")
model += (x[2]  - T13N  == 20*(1-0.04*x[16]), "10")
model += (x[4]  - x[3]  == 12*(1-0.03*x[17]),  "11")
model += (x[6]  - x[5]  == 14*(1-0.06*x[18]),  "12")
model += (x[10] - x[9]  == 16*(1-0.05*x[19]), "13")
model += (x[12] - x[11] == 6*(1-0.01*x[20]),  "14")

model += (x[3]  >= x[1] ,  "15")
model += (x[5]  >= x[1] ,  "16")
model += (x[9]  >= x[2] ,  "17")
model += (x[9]  >= x[4] ,  "18")
model += (x[7]  >= x[2] ,  "19")
model += (x[7]  >= x[4] ,  "20")
model += (x[11] >= x[6] ,  "21")
model += (x[11] >= x[8] ,  "22")
model += (x[13] >= x[10],  "23")
model += (x[13] >= x[12],  "24")

# describe goal
model += x[14]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
counter=0
for var in x.values():
    counter=counter+1
    if(counter==14):
        timeSpent[3]=var.value()
    if (counter < 15):
        print(f"{var.name}: {var.value()}"+" days")
        continue
    print(f"{var.name}: {var.value()}"+" $")

print("Spend 5$,time is",timeSpent[0],"Spend 10$,time is",timeSpent[1],"Spend 25$,time is",timeSpent[2],"Spend 50$,time is",timeSpent[3])

plt.plot(attachments,timeSpent,color = "green",marker='x')
plt.axis([0, 50, 0, 35])
plt.xlabel("attachments")
plt.ylabel("timeSpent")
plt.grid()
plt.title("LOL4")
plt.show()

#Task3
print("Task3")
#Part1
print("Part1")

DG = nx.DiGraph()
e=[('a','b',2),('a','c',4),('b','c',6),('b','d',4),('c','e',6),('d','e',3)]
DG.add_weighted_edges_from(e)
print("Critical route of network graph is :",nx.dag_longest_path(DG))
print("Length of critical route of network graph is:",nx.dag_longest_path_length(DG))
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
# array of costs for each work
CijMax=np.array([35,22,45,32,24,65])
print("the total cost of the project in the original mode",CijMax.sum(),"$")

#Part2
print("Part2")
model = LpProblem(name="Task3Lol2", sense=LpMinimize)
t = {i: LpVariable(name=f"t{i}", lowBound=0) for i in range(1, 6)}
model += (t[1]==0, "1")
model += (t[2]-t[1]>=2,  "2")
model += (t[3]-t[1]>=4,  "3")
model += (t[3]-t[2]>=6,  "4")
model += (t[4]-t[2]>=4,  "5")
model += (t[5]-t[3]>=6,  "6")
model += (t[5]-t[4]>=3,  "7")
model += (t[2]     >=0,  "8")
model += (t[3]     >=0,  "9")
model += (t[4]     >=0,  "10")
model += (t[5]     ==17, "11")
model +=35-2*(t[2]-t[1]-2)+22-1.5*(t[3]-t[1]-4)+45-8*(t[3]-t[2]-6)+32-6*(t[4]-t[2]-4)+24-3*(t[5]-t[3]-6)+65-2.5*(t[5]-t[4]-3)
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
qSecondPlayer=[0]*5
counter=0
print("duration of works")
for var in t.values():
    qSecondPlayer[counter]=var.value()
    counter=counter+1
    print(f"{var.name}: {var.value()}")


DG = nx.DiGraph()
e=[('a','b',2),('a','c',11),('b','c',9),('b','d',12),('c','e',6),('d','e',3)]
DG.add_weighted_edges_from(e)
print("Critical route of network graph is :",nx.dag_longest_path(DG))
print("Length of critical route of network graph is:",nx.dag_longest_path_length(DG))
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