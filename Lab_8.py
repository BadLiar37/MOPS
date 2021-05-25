import random
import numpy as np

#implementation of excel function ВПР
def VPR(randomNumbers,probability,count_of_clients):
    numbersOfNewComerClients=np.zeros(len(randomNumbers))
    for i in range(len(randomNumbers)):
        for j in range(0,4,1):
            if (probability[j] / 100 > randomNumbers[i]):
                break;
            if (probability[j] / 100 <= randomNumbers[i]):
                numbersOfNewComerClients[i] = count_of_clients[j]

    return numbersOfNewComerClients

def MIN(numberOfParkingLots,numbersOfNewComerClients,numberOfClientsInTheQueue):
    if(numberOfClientsInTheQueue>0 and numbersOfNewComerClients==0): return 1
    return numberOfParkingLots if numbersOfNewComerClients>=numberOfParkingLots else numbersOfNewComerClients


count_of_clients=np.array([0,1,2,3])
probability=np.array([70,12,16,2])
integral_probability=np.array([0,70,82,98])
five_minutes_intervals=np.zeros(100)
expected_clients=0
for i in range(len(count_of_clients)):
    expected_clients+=count_of_clients[i]*probability[i]/100
print("expected clients:",(expected_clients))
for i in range(0,len(five_minutes_intervals)):
    five_minutes_intervals[i]=i
print("Five minutes intervals,number:",five_minutes_intervals)
randomNumbers=np.zeros(100)
for i in range(0,len(randomNumbers)):
    randomNumbers[i]=random.uniform(0,1)
print("Random number between 0 and 1",randomNumbers)
randomNumbers[0]=0
numbersOfNewComerClients=VPR(randomNumbers,integral_probability,count_of_clients)
print("Number of Clients:",numbersOfNewComerClients)
#////////////////////////////////////
#change to second situation!!
#values=2
#////////////////////////////////////
numberOfParkingLots=1
print("number of parking lots:",numberOfParkingLots)
numberOfClientsInTheQueue=np.zeros(100)
numberOfClientsInService=np.zeros(100)
numberOfClientsInWait=np.zeros(100)
for i in range(1,len(numberOfClientsInWait)):
    numberOfClientsInTheQueue[i]=numbersOfNewComerClients[i]+numberOfClientsInWait[i-1]
    numberOfClientsInService[i]=MIN(numberOfParkingLots,numbersOfNewComerClients[i],numberOfClientsInTheQueue[i])
    numberOfClientsInWait[i]=numberOfClientsInTheQueue[i]-numberOfClientsInService[i]
print("number Of Clients In The Queue:",numberOfClientsInTheQueue)
print("number Of Clients In The Service:",numberOfClientsInService)
print("number Of Clients In The Wait:",numberOfClientsInWait)
print("///////////////")
print("Number of Clients(Max):",np.max(numbersOfNewComerClients))
print("Number of Clients(Average):",numbersOfNewComerClients.mean())
print("Number of Clients(Sum):",numbersOfNewComerClients.sum())
print("///////////////")
print("number Of Clients In The Queue(Max):",np.max(numberOfClientsInTheQueue))
print("number Of Clients In The Queue(Average):",numberOfClientsInTheQueue.mean())
print("number Of Clients In The Queue(Sum):",numberOfClientsInTheQueue.sum())
print("///////////////")
print("number Of Clients In The Service(Max)",np.max(numberOfClientsInService))
print("number Of Clients In The Service(Average):",numberOfClientsInService.mean())
print("number Of Clients In The Service(Sum):",numberOfClientsInService.sum())
print("///////////////")
print("number Of Clients In The Wait(Max)",np.max(numberOfClientsInWait))
print("number Of Clients In The Wait(Average):",numberOfClientsInWait.mean())
print("number Of Clients In The Wait(Sum):",numberOfClientsInWait.sum())