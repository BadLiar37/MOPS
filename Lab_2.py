import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable,LpMinimize

def findPredominateStategies(payMatrix):
    tempColumn=[0]*4
    tempRow =[0] * 4
    numberOfTempRow=0
    numberOfTempColumn=0
    for i in range(len(payMatrix)):
        for j in range(len(payMatrix[i])):
            if(payMatrix[i][j]>tempColumn[i]):
                tempColumn[i]=payMatrix[i][j]
                numberOfTempColumn=j
            if(payMatrix[j][i]<tempRow[i]):
                tempRow[i]=payMatrix[j][i]
                numberOfTempRow=j
    payMatrix=np.delete(payMatrix,numberOfTempRow,0)
    payMatrix=np.delete(payMatrix,numberOfTempColumn,1)

    return payMatrix

def findMinimalElementInArray(payMatrix):
    temp=0
    for i in range(len(payMatrix)):
        for j in range(len(payMatrix[i])):
            if(payMatrix[i][j]<temp):
                temp=payMatrix[i][j]
    return temp

def makeMatrixElementsPositive(payMatrix,minElement):
    for i in range(len(payMatrix)):
        for j in range(len(payMatrix[i])):
            payMatrix[i][j]=payMatrix[i][j]+abs(minElement)

    return payMatrix

"Task1"
print("Task1")
# Point1
print("Point1")
#создаём модель
model = LpProblem(name="Task1Lol1", sense=LpMinimize)
# Описываем переменные
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(0, 3)}
# Добавляем ограничения
model += (14*x[0] + 15*x[1] + 33*x[2] >= 1, "First")
model += (20*x[0] + 11*x[1] + 9 *x[2] >= 1, "Second")
model += (32*x[0] + 19*x[1] + 16*x[2] >= 1, "Third")
model += (8*x[0]  + 37*x[1] + 34*x[2] >= 1, "Fourth")
# Описываем цель
model +=  x[0] +  x[1] +x[2]
# Решаем задачу оптимизации
status = model.solve()
# Выводим результаты решения
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
pFirstPlayer=[0]*3
counter=0

for var in x.values():
     pFirstPlayer[counter]=var.value()
     counter=counter+1
     print(f"{var.name}: {var.value()}")

priceOfGameOne=1/model.objective.value()
print("Price of Game one:",priceOfGameOne)
for i in range(0,len(pFirstPlayer)):
    pFirstPlayer[i]=pFirstPlayer[i]*priceOfGameOne
print("optimal mixed strategies of first Player:",pFirstPlayer)


# Point2
print("Point2")
model = LpProblem(name="Task1Lol2", sense=LpMaximize)
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(0, 4)}
model += (14*x[0] + 20*x[1] + 32*x[2] + 8*x[3]  <= 1, "First")
model += (15*x[0] + 11*x[1] + 19*x[2] + 37*x[3] <= 1, "Second")
model += (33*x[0] + 9*x[1]  + 16*x[2] + 34*x[3] <= 1, "Third")
model +=  x[0] +  x[1] + x[2] +  x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
qSecondPlayer=[0]*4
counter=0
for var in x.values():
    qSecondPlayer[counter]=var.value()
    counter=counter+1
    print(f"{var.name}: {var.value()}")

priceOfGameOne=1/model.objective.value()
print("Price of Game one:",priceOfGameOne)
for i in range(0,len(qSecondPlayer)):
    qSecondPlayer[i]=qSecondPlayer[i]*priceOfGameOne
print("optimal mixed strategies of second Player:",qSecondPlayer)


#Task2
print("Task2")
matrixTask2=np.array([[-4,-5,-1,6],
                      [-1,0,-3,5],
                      [-3,1,-5,5],
                      [-8,-7,-6,0]])

print(matrixTask2)
matrixTask2=findPredominateStategies(matrixTask2)
print(matrixTask2)
minElement=findMinimalElementInArray(matrixTask2)
print("Minimal Element",minElement)
matrixTask2=makeMatrixElementsPositive(matrixTask2,minElement)
print(matrixTask2)
# Point1
print("Point1")
model = LpProblem(name="Task2Lol1", sense=LpMaximize)
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(0, 3)}
model += (x[0]  + 4*x[2] <= 1, "First")
model += (4*x[0] + 5*x[1] + 2*x[2] <= 1, "Second")
model += (2*x[0] + 6*x[1] <= 1, "Third")
model +=  x[0] +  x[1] +x[2]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
pFirstPlayer=[0]*3
counter=0

for var in x.values():
     pFirstPlayer[counter]=var.value()
     counter=counter+1
     print(f"{var.name}: {var.value()}")
for name, constraint in model.constraints.items():
     print(f"{name}: {constraint.value()}")

priceOfGameOne=1/model.objective.value()
print("Price of Game one:",priceOfGameOne+minElement)
for i in range(0,len(pFirstPlayer)):
    pFirstPlayer[i]=pFirstPlayer[i]*priceOfGameOne
print("optimal mixed strategies of first Player:",pFirstPlayer)


# Point2
print("Point2")
model = LpProblem(name="Task2Lol2", sense=LpMinimize)
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(0, 3)}
model += (1*x[0] + 4*x[1] + 2*x[2] >= 1, "First")
model += (         5*x[1] + 6*x[2] >= 1, "Second")
model += (4*x[0] + 2*x[1]          >= 1, "Third")
model +=  x[0] +  x[1] + x[2]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
qSecondPlayer=[0]*3
counter=0

for var in x.values():
    qSecondPlayer[counter]=var.value()
    counter=counter+1
    print(f"{var.name}: {var.value()}")
for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")

priceOfGameOne=1/model.objective.value()
print("Price of Game one:",priceOfGameOne+minElement)
for i in range(0,len(qSecondPlayer)):
    qSecondPlayer[i]=qSecondPlayer[i]*priceOfGameOne
print("optimal mixed strategies of second Player:",qSecondPlayer)
matrixA=np.array([qSecondPlayer])

print("market share of 1 enterprise:",53+priceOfGameOne+minElement,"%")
print("market share of 2 enterprise:",47-priceOfGameOne-minElement,"%")
print("The Average gain of The First Player is:",np.dot(np.dot(pFirstPlayer,matrixTask2),qSecondPlayer))
#Task3
print("Task3")

matrixTask3=np.array([[0,0.5,5/6],
                      [1,0.75,0.5]])
pFirstPlayer=np.array([3/8,5/8])

qSecondPlayer=np.array([0.25,0,0.75])
#Point1
print("Point1")
print("The Average gain of The First Player in mixed strategies is:",np.dot(np.dot(pFirstPlayer,matrixTask3),qSecondPlayer))
#Point2
print("Point2")
qSecondPlayer=np.array([0,1,0])
print("The gain of The First Player in clean strategy is:",np.dot(np.dot(pFirstPlayer,matrixTask3),qSecondPlayer))
