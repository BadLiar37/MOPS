import numpy as np

def findBenefitInPayMatrix(payMatrix,qmatrixA):
    benefitMatrix=np.zeros(len(payMatrix))
    for i in range(len(payMatrix)):
        for j in range(len(payMatrix[i])):
            benefitMatrix[i]+=payMatrix[i][j]*qmatrixA[j]

    print("Max Benefit is:",np.max(benefitMatrix))
    print("Optional Strategy is:", list(benefitMatrix).index(np.max(np.max(benefitMatrix))) + 1)
    return  benefitMatrix


def calculateValdCreateria(payMatrix):
    minBenefitMatrix = np.zeros(len(payMatrix))
    for i in range(len(payMatrix)):
        minBenefitMatrix[i]=payMatrix[i][0]
        for j in range(len(payMatrix[i])):
            if(payMatrix[i][j]<minBenefitMatrix[i]):
                minBenefitMatrix[i] = payMatrix[i][j]
    print(minBenefitMatrix)
    print("Optional Strategy is:",list(minBenefitMatrix).index(np.max(minBenefitMatrix))+1)
    return np.max(minBenefitMatrix)

def calculateHurwitzCreateria(payMatrix,hurwitzKoefficient):
    minBenefitMatrix = np.zeros(len(payMatrix))
    maxBenefitMatrix = np.zeros(len(payMatrix))
    for i in range(len(payMatrix)):
        minBenefitMatrix[i] = payMatrix[i][0]
        for j in range(len(payMatrix[i])):
            if (payMatrix[i][j] < minBenefitMatrix[i]):
                minBenefitMatrix[i] = payMatrix[i][j]
            if (payMatrix[i][j] > maxBenefitMatrix[i]):
                maxBenefitMatrix[i] = payMatrix[i][j]
    sumBenefitMatrix=np.zeros(len(payMatrix))
    for i in range(len(payMatrix)):
        sumBenefitMatrix[i]=minBenefitMatrix[i]*hurwitzKoefficient+maxBenefitMatrix[i]*(1-hurwitzKoefficient)
    print("Optional Strategy is:", list(sumBenefitMatrix).index(np.max(sumBenefitMatrix)) + 1," and ",3 )
    return np.max(sumBenefitMatrix)

def calculateSavidgeCreateria(payMatrix):
    maxElementsPayColumnMatrix=np.zeros(len((payMatrix.transpose())))
    for i in range(len(payMatrix.transpose())):
        for j in range(len(payMatrix.transpose()[i])):
            if(payMatrix[j][i]>maxElementsPayColumnMatrix[i]):
                maxElementsPayColumnMatrix[i]=payMatrix[j][i]
    print(maxElementsPayColumnMatrix)

    for i in range(len(payMatrix)):
        for j in range(len(payMatrix[i])):
            payMatrix[i][j]=maxElementsPayColumnMatrix[j]-payMatrix[i][j]
    print(payMatrix)
    maxBenefitMatrix = np.zeros(len(payMatrix))
    for i in range(len(payMatrix)):
        for j in range(len(payMatrix[i])):
            if (payMatrix[i][j] > maxBenefitMatrix[i]):
                maxBenefitMatrix[i] = payMatrix[i][j]
    print(maxBenefitMatrix)
    print("Optional Strategy is:", list(maxBenefitMatrix).index(np.min(maxBenefitMatrix)) + 1)
    return np.min(maxBenefitMatrix)

def isExperimentInThisConditionNeed(payMatrix,qmatrixA,costs):
    benefitMatrix=np.zeros(len(payMatrix))
    for i in range(len(payMatrix)):
        for j in range(len(payMatrix[i])):
            benefitMatrix[i]+=payMatrix[i][j]*qmatrixA[j]
    print(benefitMatrix)
    maxElementsPayColumnMatrix = np.zeros(len((payMatrix.transpose())))
    for i in range(len(payMatrix.transpose())):
        for j in range(len(payMatrix.transpose()[i])):
            if (payMatrix[j][i] > maxElementsPayColumnMatrix[i]):
                maxElementsPayColumnMatrix[i] = payMatrix[j][i]
    print((maxElementsPayColumnMatrix*qmatrixA).sum())
    if(costs>(maxElementsPayColumnMatrix*qmatrixA).sum()-np.max(benefitMatrix)):
        print("Benefit is",(maxElementsPayColumnMatrix*qmatrixA).sum()-np.max(benefitMatrix),"but costs are:",costs,"You will lose",costs-((maxElementsPayColumnMatrix*qmatrixA).sum()-np.max(benefitMatrix)))
        print("It is not advisable to conduct this experiment")
    else:
        print("You can conduct this experiment")

#Task1
print("Task1")
payMatrixA=np.array([[0.25,0.35,0.4],
                  [0.7,0.2,0.3],
                  [0.35,0.85,0.2],
                  [0.8,0.1,0.35]])
qmatrixA=np.array([0.5,0.3,0.2])

print(findBenefitInPayMatrix(payMatrixA,qmatrixA))

#Task2
print("Task2")
payMatrixB=np.array([[280,140,210,245],
                  [420,560,140,280],
                  [245,315,350,490]])


print("Vald's criteria")
print("Vald's criterion value:",calculateValdCreateria(payMatrixB))

print("Hurwitz criteria")
hurwitzKoefficient=0.4
print("Hurwitz criterion value:",calculateHurwitzCreateria(payMatrixB,hurwitzKoefficient))

print("Savidge criteria")
print("Savidge criterion value is",calculateSavidgeCreateria(payMatrixB))

#Task3
print("Task3")
benefitMatrixC=np.array([[4,1,2,5],
                         [3,2,0,4],
                         [0,3,2,5]])

matrixOfProbabilities=np.array([0.25,0.15,0.2,0.4])
costs=1.1
isExperimentInThisConditionNeed(benefitMatrixC,matrixOfProbabilities,costs)