import numpy as np
import matplotlib.pyplot as plt
from pulp import *

# Task1
print("Task1")

index=np.zeros(10)
task1ResultArray=np.zeros(10)
for i in range(0,10,1):
    index[i]=i
    a1=1-i/10
    a2=i/10
    # create model
    model = LpProblem(name="Task1Lol1", sense=LpMaximize)
    # create variables for constraints
    x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
    # add constraints
    model += (x[1]+2*x[2]+4*x[3]     <= 120, "1")
    model += (3*x[1]+x[2]+2*x[3]     <= 125, "2")
    model += (x[1]+2*x[2]+x[3]       <= 140, "3")
    model += (x[1]+2*x[2]+2*x[3]     <= 100, "4")
    model += (3*x[1]+2*x[2]+x[3]     <= 150, "5")
    model += (2*x[1]+3*x[2]+2*x[3]   <= 180, "6")
    # describe goal
    model += a1*(4*x[1]+3*x[2]+11*x[3])+a2*(2*x[1]+4*x[2]+7*x[3])
    status = model.solve()
    task1ResultArray[i]=model.objective.value()
    # print(f"status: {model.status}, {LpStatus[model.status]}")
    # print(f"objective: {model.objective.value()}")
    # for var in x.values():
    #     print(f"{var.name}: {var.value()}"+" $")
    # print(f"Min delivery cost: {model.objective.value()}$")

print(task1ResultArray)

plt.plot(index,task1ResultArray)
plt.xlabel('index')
plt.ylabel('task1ResultArray')
plt.title("LOL3")
# plt.axis([0, 1, 12,27])
plt.show()

# Task2
print("Task2")
assignmentByGain=580
assignmentByTradePrice=1080



#Lol1
print("Lol1")

#Part1
print("Part1")
# create model
model = LpProblem(name="Task2Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
# add constraints
model += (2*x[1] + 4* x[2] + 4 * x[3]  <= 400, "1")
model += (3 * x[1] +2* x[2] + 2 * x[3] <= 300, "2")
model += (4*x[1] + 5 * x[2] +3* x[3]   <= 500, "3")
model += (x[3]                         >= 30, "4")

# describe goal
model +=22*x[1]+12*x[2]+14*x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")

gain = model.objective.value()

#Part2
print("Part2")
# create model
model = LpProblem(name="Task2Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
# add constraints
model += (2 * x[1] + 4 * x[2] + 4 * x[3] <= 400, "1")
model += (3 * x[1] + 2 * x[2] + 2 * x[3] <= 300, "2")
model += (4 * x[1] + 5 * x[2] + 3 * x[3] <= 500, "3")
model += (x[3]                           >= 30, "4")
model +=(22*x[1]+12*x[2]+14*x[3]         >=gain-assignmentByGain,"5")

# describe goal
model += 24 * x[1] + 16 * x[2] + 30 * x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")

tradePrice = model.objective.value()

#Part3
print("Part3")

# create model
model = LpProblem(name="Task2Lol1", sense=LpMinimize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
# add constraints
model += (2 * x[1] + 4 * x[2] + 4 * x[3] <= 400, "1")
model += (3 * x[1] + 2 * x[2] + 2 * x[3] <= 300, "2")
model += (4 * x[1] + 5 * x[2] + 3 * x[3] <= 500, "3")
model += (x[3]                           >= 30, "4")
model +=(22*x[1]+12*x[2]+14*x[3]         >=gain-assignmentByGain,"5")
model +=(24*x[1]+16*x[2]+30*x[3]         >=tradePrice-assignmentByTradePrice,"6")
# describe goal
model += 20 * x[1] + 10 * x[2] + 15 * x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")

print("Уступка по прибыли:",assignmentByGain)
print("Результат целевой функции:",gain)
print("Уступка по оптовой цене:",assignmentByTradePrice)
print("Результат целевой функции:",tradePrice)
print("Финальное значение целевой функции:",model.objective.value())

#Lol2
print("Lol2")


model = LpProblem(name="Task2Lol2", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0,cat='Integer') for i in range(1, 4)}
# add constraints
model += (2 * x[1] + 4 * x[2] + 4 * x[3]   <= 400, "1")
model += (3 * x[1] + 2 * x[2] + 2 * x[3]   <= 300, "2")
model += (4 * x[1] + 5 * x[2] + 3 * x[3]   <= 500, "3")
model += (x[3]                             >= 30, "4")
model +=(20 * x[1] + 10 * x[2] + 15 * x[3] <=1600,"5")
model +=(24*x[1]+16*x[2]+30*x[3]>=2400,"6")
# describe goal
model += 22*x[1]+12*x[2]+14*x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")
print("Прибыль:",model.objective.value())

# Task3
print("Task3")

w1=0.3
w2=0.5
w3=0.2
p=2

#Objective fuctions

#function 1

model = LpProblem(name="Task3Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
#
model += ( x[1] + 2 * x[2] + 4 * x[3] <= 150, "1")
model += (3 * x[1] +  x[2] + 2 * x[3] <= 180, "2")
model += ( x[1] + 2 * x[2] +  x[3]    <= 240, "3")
# describe goal
model += 4*x[1]+3*x[2]+11*x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")
f1_ObjectiveValue=model.objective.value()

#function 2

model = LpProblem(name="Task3Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}

model += ( x[1] + 2 * x[2] + 4 * x[3] <= 150, "1")
model += (3 * x[1] +  x[2] + 2 * x[3] <= 180, "2")
model += ( x[1] + 2 * x[2] +  x[3]    <= 240, "3")
# describe goal
model += 2*x[1]+4*x[2]+7*x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")
f2_ObjectiveValue=model.objective.value()

#function 3

model = LpProblem(name="Task3Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}

model += ( x[1] + 2 * x[2] + 4 * x[3] <= 150, "1")
model += (3 * x[1] +  x[2] + 2 * x[3] <= 180, "2")
model += ( x[1] + 2 * x[2] +  x[3]    <= 240, "3")
# describe goal
model += 3*x[1]+5*x[2]+4*x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")

f3_ObjectiveValue=model.objective.value()


# next part

#function 1

model = LpProblem(name="Task3Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
#
model += ( x[1] + 2 * x[2] + 4 * x[3] <= 150, "1")
model += (3 * x[1] +  x[2] + 2 * x[3] <= 180, "2")
model += ( x[1] + 2 * x[2] +  x[3]    <= 240, "3")
# describe goal
model += (w1*(4*x[1]+3*x[2]+11*x[3]-4*x[1]+3*x[2]+11*x[3])+w1*(4*x[1]+3*x[2]+11*x[3]-4*x[1]+3*x[2]+11*x[3]))/1.5
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")
f1_Next_ObjectiveValue=model.objective.value()
#function 2
model = LpProblem(name="Task3Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}

model += ( x[1] + 2 * x[2] + 4 * x[3] <= 150, "1")
model += (3 * x[1] +  x[2] + 2 * x[3] <= 180, "2")
model += ( x[1] + 2 * x[2] +  x[3]    <= 240, "3")
# describe goal
model +=(w2*(2*x[1]+4*x[2]+7*x[3]-2*x[1]+4*x[2]+7*x[3])+w2*(2*x[1]+4*x[2]+7*x[3]-2*x[1]+4*x[2]+7*x[3]))/3
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")
f2_Next_ObjectiveValue=model.objective.value()

#functuion 3
model = LpProblem(name="Task3Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
model += ( x[1] + 2 * x[2] + 4 * x[3]<= 150, "1")
model += (3 * x[1] +  x[2] + 2 * x[3]<= 180, "2")
model += ( x[1] + 2 * x[2] +  x[3]<= 240, "3")
# describe goal
model += (w3*(3*x[1]+5*x[2]+4*x[3]-3*x[1]+5*x[2]+4*x[3])+w3*(3*x[1]+5*x[2]+4*x[3]-3*x[1]+5*x[2]+4*x[3]))
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}")
f3_Next_ObjectiveValue=model.objective.value()



d_F_F=(w1*(abs(f1_ObjectiveValue-f1_Next_ObjectiveValue)**p))**(1/p)+(w2*(abs(f2_ObjectiveValue-f2_Next_ObjectiveValue)**p))**(1/p)+(w3*(abs(f3_ObjectiveValue-f3_Next_ObjectiveValue)**p))**(1/p)
print("Final function value:",d_F_F)






















































































# dual task
#
# # function 1
# model = LpProblem(name="Task3Lol1", sense=LpMinimize)
# # create variables for constraints
# y = {i: LpVariable(name=f"y{i}", lowBound=0) for i in range(1, 4)}
#
# model += ( y[1] + 3* y[2] + y[3]     >= 4, "1")
# model += (2* y[1] +  y[2] + 2 * y[3] >= 3, "2")
# model += (4*y[1] + 2 * y[2] +  y[3]  >=11, "3")
# # describe goal
# model += 150*y[1]+180*y[2]+240*y[3]
# status = model.solve()
# print(f"status: {model.status}, {LpStatus[model.status]}")
# print(f"objective: {model.objective.value()}")
# for var in y.values():
#     print(f"{var.name}: {var.value()}")
#
# f1_DualObjectiveValue=model.objective.value()
#
# #function 2
#
# model = LpProblem(name="Task3Lol1", sense=LpMinimize)
# # create variables for constraints
# y = {i: LpVariable(name=f"y{i}", lowBound=0) for i in range(1, 4)}
#
# model += ( y[1] + 3* y[2] + y[3]     >= 2, "1")
# model += (2* y[1] +  y[2] + 2 * y[3] >= 4, "2")
# model += (4*y[1] + 2 * y[2] +  y[3]  >=7, "3")
# # describe goal
# model += 150*y[1]+180*y[2]+240*y[3]
# status = model.solve()
# print(f"status: {model.status}, {LpStatus[model.status]}")
# print(f"objective: {model.objective.value()}")
# for var in y.values():
#     print(f"{var.name}: {var.value()}")
#
# f2_DualObjectiveValue=model.objective.value()
#
# #fucntion 3
#
# model = LpProblem(name="Task3Lol1", sense=LpMinimize)
# # create variables for constraints
# y = {i: LpVariable(name=f"y{i}", lowBound=0) for i in range(1, 4)}
#
# model += ( y[1] + 3* y[2] + y[3]     >= 3, "1")
# model += (2* y[1] +  y[2] + 2 * y[3] >= 5, "2")
# model += (4*y[1] + 2 * y[2] +  y[3]  >=4, "3")
# # describe goal
# model += 150*y[1]+180*y[2]+240*y[3]
# status = model.solve()
# print(f"status: {model.status}, {LpStatus[model.status]}")
# print(f"objective: {model.objective.value()}")
# for var in y.values():
#     print(f"{var.name}: {var.value()}")
#
# f3_DualObjectiveValue=model.objective.value()