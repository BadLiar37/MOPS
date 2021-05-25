import numpy as np
import math
import matplotlib.pyplot as plt
from pulp import *
import networkx as nx
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



# Task1
# print("Task1")
# # arrays

c1=np.array([10,10,8,9,11,12,8,9,11,11,10,12])
c2=np.array([9,9,10,10,10,11,12,11,9,10,11,11])
c3=np.array([10,12,9,12,8,8,10,8,10,8,11,12])

a11=np.array([8,11,11,8,9,9,11,10,11,11,12,12])
a12=np.array([11,10,11,8,12,12,10,9,8,8,8,12])

a21=np.array([10,11,10,12,9,8,12,9,8,9,11,12])
a22=np.array([9,8,11,8,11,10,12,12,10,9,9,9])

a31=np.array([12,10,12,11,10,11,9,11,8,11,10,8])
a32=np.array([9,11,10,11,11,10,10,11,8,12,9,10])

b1=np.array([23,21,20,23,21,24,24,20,22,24,21,21])
b2=np.array([23,22,23,22,22,22,22,21,24,20,23,20])
b3=np.array([24,22,20,24,20,21,21,20,22,23,20,20])










# create model
model = LpProblem(name="Task1Lol1", sense=LpMaximize)
# create variables for constraints
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
# add constraints
model += (a11.mean()*x[1]+a12.mean()*x[2]               <= b1.mean(), "1")
model += (a21.mean()*x[1]+a22.mean()*x[3]               <= b2.mean(), "2")
model += (a31.mean()*x[1]+a32.mean()*x[3]               <= b3.mean(), "3")

# describe goal
model += c1.mean()*x[1]+c2.mean()*x[2]+c3.mean()*x[3]
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.name}: {var.value()}"+" $")
print(f"Min delivery cost: {model.objective.value()}$")




# # Task2
print("Task2")
beta=np.array([0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.9,0.95,0.97,0.99,0.999])
optimalValues=np.zeros(len(beta))
index=np.zeros(len(beta))
c1=np.array([12,10,10,10,11,10,11,11,12,10,11,12])
c2=np.array([8,10,11,11,12,8,11,10,12,12,10,9])
c3=np.array([11,11,11,12,8,10,11,10,8,12,9,12])






a11=np.array([10,10,11,9,10,10,9,8,11,12,12,9])
a12=np.array([12,9,11,9,10,10,12,12,12,9,12,9])
a13=np.array([9,12,8,8,8,10,11,11,8,12,8,12])



a21=np.array([11,12,11,11,11,8,9,11,9,9,9,8])
a22=np.array([8,11,9,10,9,8,11,10,12,12,12,11])
a23=np.array([12,12,9,12,11,8,11,11,9,11,11,12])

a31=np.array([9,11,10,9,11,11,10,11,12,9,8,9])
a32=np.array([12,10,9,9,11,9,11,9,10,11,12,9])
a33=np.array([12,8,11,8,12,11,8,12,9,11,9,8])

b1=np.array([24,22,24,21,20,22,21,24,23,22,23,24])
b2=np.array([23,23,23,20,21,20,23,22,22,23,22,21])
b3=np.array([24,22,21,24,24,24,20,23,24,23,24,22])

for i in range(0,len(beta)):
    # create model
    model = LpProblem(name="Task2Lol1", sense=LpMaximize)
    # create variables for constraints
    x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
    # add constraints
    model += (a11.mean()*x[1]+a12.mean()*x[2]+a13.mean()*x[3]+norm.ppf(beta[i])*(np.var(a11,ddof = 1)*x[1]+np.var(a11,ddof = 1)*x[1]+np.var(a12,ddof = 1)*x[2]+np.var(a12,ddof = 1)*x[2]+np.var(a13,ddof = 1)*x[3]+np.var(a13,ddof = 1)*x[3]+np.var(b1))/4<= b1.mean(), "1")
    model += (a21.mean()*x[1]+a22.mean()*x[2]+a23.mean()*x[3]+norm.ppf(beta[i])*(np.var(a21,ddof = 1)*x[1]+np.var(a21,ddof = 1)*x[1]+np.var(a22,ddof = 1)*x[2]+np.var(a22,ddof = 1)*x[2]+np.var(a23,ddof = 1)*x[3]+np.var(a23,ddof = 1)*x[3]+np.var(b2))/2.25<= b2.mean(), "2")
    model += (a31.mean()*x[1]+a32.mean()*x[2]+a33.mean()*x[3]+norm.ppf(beta[i])*(np.var(a31,ddof = 1)*x[1]+np.var(a31,ddof = 1)*x[1]+np.var(a32,ddof = 1)*x[2]+np.var(a32,ddof = 1)*x[2]+np.var(a33,ddof = 1)*x[3]+np.var(a33,ddof = 1)*x[3]+np.var(b1))/8<= b3.mean(), "3")

    # describe goal
    model += c1.mean()*x[1]+c2.mean()*x[2]+c3.mean()*x[3]
    status = model.solve()
    index[i]=i
    optimalValues[i]=model.objective.value()


print(optimalValues)

plt.plot(beta,optimalValues)
plt.xlabel('index')
plt.ylabel('optimalValues')
plt.title("LOL3")
plt.axis([0, 1, 12,27])
plt.show()
#
#
#Task3
print("Task3")
beta1=np.array([0.2,0.4,0.6,0.8,0.9,0.95,0.99])
beta2=np.array([0.2,0.4,0.6,0.8,0.9,0.95,0.99])
optimalValuesB1B2=np.zeros([7,7])
print(optimalValuesB1B2)

for i in range(0,len(beta1)):
    for j in range(0,len(beta2)):
        # create model
        model = LpProblem(name="Task2Lol1", sense=LpMaximize)
        # create variables for constraints
        x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}
        # add constraints
        model += (a11.mean()*x[1]+a12.mean()*x[2]+a13.mean()*x[3]+norm.ppf(beta1[i])*(np.var(a11,ddof = 1)*x[1]+np.var(a11,ddof = 1)*x[1]+np.var(a12,ddof = 1)*x[2]+np.var(a12,ddof = 1)*x[2]+np.var(a13,ddof = 1)*x[3]+np.var(a13,ddof = 1)*x[3]+np.var(b1))/4<= b1.mean(), "1")
        model += (a21.mean()*x[1]+a22.mean()*x[2]+a23.mean()*x[3]+norm.ppf(beta2[j])*(np.var(a21,ddof = 1)*x[1]+np.var(a21,ddof = 1)*x[1]+np.var(a22,ddof = 1)*x[2]+np.var(a22,ddof = 1)*x[2]+np.var(a23,ddof = 1)*x[3]+np.var(a23,ddof = 1)*x[3]+np.var(b2))/2.25<= b2.mean(), "2")
        model += (a31.mean()*x[1]+a32.mean()*x[2]+a33.mean()*x[3]+norm.ppf(0.9)*(np.var(a31,ddof = 1)*x[1]+np.var(a31,ddof = 1)*x[1]+np.var(a32,ddof = 1)*x[2]+np.var(a32,ddof = 1)*x[2]+np.var(a33,ddof = 1)*x[3]+np.var(a33,ddof = 1)*x[3]+np.var(b1))/8<= b3.mean(), "3")

        # describe goal
        model += c1.mean()*x[1]+c2.mean()*x[2]+c3.mean()*x[3]
        status = model.solve()
        optimalValuesB1B2[i][j]=model.objective.value()


print(optimalValuesB1B2)



fig = plt.figure()
ax = plt.axes(projection="3d")
def z_function(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.array(beta1)
y = np.array(beta2)

X, Y = np.meshgrid(x, y)
Z = optimalValuesB1B2

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()