import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


#Task1
print("Task1")

DG = nx.DiGraph()
DG.add_weighted_edges_from([
                            (1, 2, 16),(1, 9, 14),
                            (2, 3, 13),(2,10, 11),
                            (3, 4, 17),(3,11, 18),
                            (4, 5, 14),(4,12, 14),
                            (5, 6, 17),(5,13, 20),
                            (6, 7, 12),(6,14, 13),
                            (7, 8, 19),(7,15, 14),
                                        (8, 16, 9),
                            (9, 10, 14),(9, 17, 18),
                            (10,11, 20),(10,18, 8),
                            (11, 12, 8),(11,19, 20),
                            (12, 13, 12),(12,20, 12),
                            (13, 14, 10),(13,21, 17),
                            (14, 15, 10),(14,22, 13),
                            (15, 16, 11),(15,23, 17),
                                        (16, 24, 10),
                            (17, 18, 18),(17,25, 11),
                            (18, 19, 15),(18,26, 11),
                            (19, 20, 16),(19,27, 8),
                            (20, 21, 8),(20,28,  20),
                            (21, 22,17),(21,29,  15),
                            (22, 23, 10),(22,30, 13),
                            (23, 24, 11),(23,31, 10),
                                        (24,32,9),
                            (25, 26, 17), (25, 33, 14),
                            (26, 27, 18), (26, 34, 8),
                            (27, 28, 10), (27, 35, 10),
                            (28, 29, 13), (28, 36, 9),
                            (29, 30, 9),  (29, 37, 9),
                            (30, 31, 14), (30, 38, 17),
                            (31, 32, 8),  (31, 39, 10),
                                      (32,40,14),
                            (33,34,12),(33,41,18),
                            (34,35,20),(34,42,11),
                            (35,36,11),(35,43, 8),
                            (36,37,12),(36,44, 8),
                            (37,38,19),(37,45,16),
                            (38,39,12),(38,46,16),
                            (39,40, 8),(39,47,11),
                                      (40,48,9),
                            (41,42,20),
                            (42,43,20),
                            (43,44,14),
                            (44,45,11),
                            (45,46,17),
                            (46,47,11),
                            (47,48,20)
                            ])
minimum_costs=nx.dijkstra_path_length(DG,1,48)
print("Critical route of network graph is :",nx.dijkstra_path(DG,1,48))
print("Length of critical route of network graph is:",minimum_costs)

#Task2
print("Task2")
DG1 = nx.DiGraph()
DG1.add_weighted_edges_from([
                            (1, 2, 19),(1, 9, 23),
                            (2, 3, 15),(2,10, 17),
                            (3, 4, 23),(3,11, 23),
                            (4, 5, 22),(4,12, 15),
                            (5, 6, 16),(5,13, 15),
                            (6, 7, 16),(6,14, 24),
                            (7, 8, 23),(7,15, 15),
                                        (8,16,22),
                            (9, 10, 15),(9, 17, 23),
                            (10,11, 16),(10,18, 19),
                            (11, 12, 15),(11,19, 19),
                            (12, 13, 21),(12,20, 24),
                            (13, 14, 20),(13,21, 16),
                            (14, 15, 20),(14,22, 20),
                            (15, 16, 18),(15,23, 24),
                                        (16, 24, 23),
                            (17, 18, 21),(17,25, 23),
                            (18, 19, 23),(18,26, 15),
                            (19, 20, 15),(19,27, 18),
                            (20, 21, 15),(20,28, 19),
                            (21, 22, 18),(21,29, 16),
                            (22, 23, 16),(22,30, 18),
                            (23, 24, 23),(23,31, 21),
                                         (24,32, 19),
                            (25, 26, 18), (25, 33, 23),
                            (26, 27, 20), (26, 34, 20),
                            (27, 28, 17), (27, 35, 18),
                            (28, 29, 20), (28, 36, 20),
                            (29, 30, 16), (29, 37, 23),
                            (30, 31, 18), (30, 38, 22),
                            (31, 32, 16), (31, 39, 19),
                                          (32,40,  16),
                            (33,34,22),(33,41,19),
                            (34,35,22),(34,42,15),
                            (35,36,21),(35,43,20),
                            (36,37,24),(36,44,22),
                            (37,38,21),(37,45,21),
                            (38,39,23),(38,46,21),
                            (39,40,17),(39,47,23),
                                      (40,48,22),
                            (41,42,22),
                            (42,43,15),
                            (43,44,24),
                            (44,45,24),
                            (45,46,16),
                            (46,47,19),
                            (47,48,19)
                            ])
max_gain=nx.dag_longest_path_length(DG1)
print("Critical route of network graph is :",nx.dag_longest_path(DG1))
print("Length of critical route of network graph is:",max_gain)
print("The effect-cost relationship:",max_gain/minimum_costs)

#Task3
print("Task3")
#Lol1
print("Lol1")

minimum_costs_array=np.zeros(10000)
index=np.zeros(10000)

for i in range(0,len(minimum_costs_array)):
    DG2 = nx.DiGraph()
    DG2.add_weighted_edges_from([
                                (1, 2, random.randrange(18, 25)),(1, 9, random.randrange(18, 25)),
                                (2, 3, random.randrange(18, 25)),(2,10, random.randrange(18, 25)),
                                (3, 4, random.randrange(18, 25)),(3,11, random.randrange(18, 25)),
                                (4, 5, random.randrange(18, 25)),(4,12, random.randrange(18, 25)),
                                (5, 6, random.randrange(18, 25)),(5,13, random.randrange(18, 25)),
                                (6, 7, random.randrange(18, 25)),(6,14, random.randrange(18, 25)),
                                (7, 8, random.randrange(18, 25)),(7,15, random.randrange(18, 25)),
                                            (8, 16, random.randrange(18, 25)),
                                (9, 10, random.randrange(18, 25)),(9, 17, random.randrange(18,  25)),
                                (10,11, random.randrange(18, 25)),(10,18, random.randrange(18,  25)),
                                (11, 12, random.randrange(18, 25)),(11,19, random.randrange(18, 25)),
                                (12, 13, random.randrange(18, 25)),(12,20, random.randrange(18, 25)),
                                (13, 14, random.randrange(18, 25)),(13,21, random.randrange(18, 25)),
                                (14, 15, random.randrange(18, 25)),(14,22, random.randrange(18, 25)),
                                (15, 16, random.randrange(18, 25)),(15,23, random.randrange(18, 25)),
                                            (16, 24, random.randrange(18, 25)),
                                (17, 18, random.randrange(18, 25)),(17,25, random.randrange(18, 25)),
                                (18, 19, random.randrange(18, 25)),(18,26, random.randrange(18, 25)),
                                (19, 20, random.randrange(18, 25)),(19,27, random.randrange(18, 25)),
                                (20, 21, random.randrange(18, 25)),(20,28,  random.randrange(18, 25)),
                                (21, 22,random.randrange(18, 25)),(21,29,  random.randrange(18, 25)),
                                (22, 23, random.randrange(18, 25)),(22,30, random.randrange(18, 25)),
                                (23, 24, random.randrange(18, 25)),(23,31, random.randrange(18, 25)),
                                            (24,32,random.randrange(18, 25)),
                                (25, 26, random.randrange(18, 25)), (25, 33, random.randrange(18, 25)),
                                (26, 27, random.randrange(18, 25)), (26, 34, random.randrange(18, 25)),
                                (27, 28, random.randrange(18, 25)), (27, 35, random.randrange(18, 25)),
                                (28, 29, random.randrange(18, 25)), (28, 36, random.randrange(18, 25)),
                                (29, 30, random.randrange(18, 25)),  (29, 37, random.randrange(18, 25)),
                                (30, 31, random.randrange(18, 25)), (30, 38, random.randrange(18, 25)),
                                (31, 32, random.randrange(18, 25)),  (31, 39, random.randrange(18, 25)),
                                          (32,40,random.randrange(18, 25)),
                                (33,34,random.randrange(18, 25)),(33,41,random.randrange(18, 25)),
                                (34,35,random.randrange(18, 25)),(34,42,random.randrange(18, 25)),
                                (35,36,random.randrange(18, 25)),(35,43, random.randrange(18, 25)),
                                (36,37,random.randrange(18, 25)),(36,44, random.randrange(18, 25)),
                                (37,38,random.randrange(18, 25)),(37,45,random.randrange(18, 25)),
                                (38,39,random.randrange(18, 25)),(38,46,random.randrange(18, 25)),
                                (39,40, random.randrange(18, 25)),(39,47,random.randrange(18, 25)),
                                          (40,48,random.randrange(18, 25)),
                                (41,42,random.randrange(18, 25)),
                                (42,43,random.randrange(18, 25)),
                                (43,44,random.randrange(18, 25)),
                                (44,45,random.randrange(18, 25)),
                                (45,46,random.randrange(18, 25)),
                                (46,47,random.randrange(18, 25)),
                                (47,48,random.randrange(18, 25))
                                ])
    index[i]=i
    minimum_costs_array[i]=nx.dijkstra_path_length(DG2,1,48)

print(minimum_costs_array)
plt.bar(index,minimum_costs_array)
plt.show()
#Lol3
print("Lol3")
def find_probability(minimum_costs_array,cost):
    probability=0;
    for i in range(0,len(minimum_costs_array)):
        if(minimum_costs_array[i]<cost):
            probability+=1/len(minimum_costs_array)

    return probability*100


print("the probability that the cost will be less than 225 is:",find_probability(minimum_costs_array,225)," %")
print("the probability that the cost will be less than 230 is:",find_probability(minimum_costs_array,230)," %")
print("the probability that the cost will be less than 235 is:",find_probability(minimum_costs_array,235)," %")
print("the probability that the cost will be less than 240 is:",find_probability(minimum_costs_array,240)," %")
print("the probability that the cost will be less than 245 is:",find_probability(minimum_costs_array,245)," %")


