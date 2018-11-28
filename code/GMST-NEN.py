import numpy as np, random, operator
import time


#Open input file
infile = open('D:\\BKHN\\Project 1\\tuan_6\\Type_1_Small\\10kroB100.clt', 'r')

# Read instance header
Name = infile.readline().strip().split()[2] # NAME
FileType = infile.readline().strip().split()[2] # TYPE
Dimension = infile.readline().strip().split()[2] # DIMENSION
NumberOfClusters = infile.readline().strip().split()[2] # NUMBER OF CLUSTERS
Type = infile.readline().strip().split()[2] # EDGE_WEIGHT_TYPE
infile.readline()

# Read node list
nodelist = []
N = int(Dimension)
for i in range(0, int(Dimension)):
    x,y = infile.readline().strip().split()[1:]
    nodelist.append([int(x), int(y)])

infile.readline()
SoureVertex = int(infile.readline().strip().split()[2])  #SOURE VERTEX

ClusterList = []
for i in range(0, int(NumberOfClusters)):
    clusterStr = infile.readline().strip().split()
    clusterStr.remove(clusterStr[0])
    clusterStr.remove(clusterStr[len(clusterStr)-1])
    cluster = []
    for j in range(0, len(clusterStr)):
        cluster.append(int(clusterStr[j]))
    ClusterList.append(cluster)             #Clusterlist save the index of city

# Close input file
infile.close()

# for i in range(0, len(ClusterList)):
#     for j in range(0, len(ClusterList[i])):
#         print(ClusterList[i][j], end = ' ')
#     print()



class City:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __showDetail__(self):
        #return str(self.id) + ": " + "(" + str(self.x) + "," + str(self.y) + ")"
        return str(self.id)
    
    def showId(self):
        return self.id
    

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance


cityList = []
for i in range(len(nodelist)):
    cityList.append(City(x = nodelist[i][0], y = nodelist[i][1], id = i))

#KRUSKAL ALGORITHM

parent = dict()
rank = dict()

def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0

def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]: rank[root2] += 1

def kruskal(graph):
    for vertice in graph['vertices']:
        make_set(vertice)

    sumOfDis = 0
    minimum_spanning_tree = {
        'vertices':  [],
        'edges': set([]) 
    }
    minimum_spanning_tree['vertices'] = graph['vertices']
    edges = list(graph['edges'])
    edges.sort()
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            sumOfDis = sumOfDis + weight
            union(vertice1, vertice2)
            minimum_spanning_tree['edges'].add(edge)
    return minimum_spanning_tree, sumOfDis



def calculateSumOfDis(p, r):            #type(p): City - r is cluster which p locates
    sum = 0
    for i in range (0 , int(NumberOfClusters)):
        if i == r:
            continue
        else:
            for j in range (0, len(ClusterList[i])):
                cityIndex = ClusterList[i][j]
                dis = p.distance(cityList[cityIndex])
                sum = sum + dis
    
    return sum    

def makeFullyGraph(listNode):
    graph = {
        'vertices': [],
        'edges': set([])
    }
    
    for i in range(0, len(listNode)):
        graph['vertices'].append(listNode[i].showId())
    
    for i in range(0, len(listNode)-1):
        for j in range(i+1, len(listNode)):
            weight = listNode[i].distance(listNode[j])
            vertice1 = listNode[i].showId()
            vertice2 = listNode[j].showId()
            graph['edges'].add((weight,vertice1,vertice2))
            graph['edges'].add((weight,vertice2,vertice1))

    return graph




def initialize():

    ListNodeInEachCluster = []     #list of City, type = list of City  
                                   #list includes one city with lowest sum in each clusters
    for i in range (0, int(NumberOfClusters)):
        #find the node with the lowest sum of edge costs to all nodes in the other clusters 
        iCity = cityList[ClusterList[i][0]]
        lowestSum = calculateSumOfDis(iCity, i)

        for j in range(0, len(ClusterList[i])):
            indCity = ClusterList[i][j]
            p = cityList[indCity]
            sumOfDis = calculateSumOfDis(p, i)

            if(sumOfDis < lowestSum):
                lowestSum = sumOfDis
                iCity = p
        
        ListNodeInEachCluster.append(iCity)

    #return ListNodeInEachCluster

    graph = makeFullyGraph(ListNodeInEachCluster)

    initMST, sumOfDis = kruskal(graph)

    return initMST, sumOfDis, ListNodeInEachCluster

#MST, sumOfDis = initialize()
# print(str(sumOfDis))

# edges = list(MST['edges'])
# sum = 0
# for edge in edges:
#     sum = sum + edge[0]

# print(str(sum))

def calculateDisInMST(MST):

    edges = list(MST['edges'])
    sum = 0
    for edge in edges:
        sum = sum + edge[0]

    return sum

def NEN (initMST, listNode):
    minDis = calculateDisInMST(initMST)
    MST = initMST
    NENList = []
    bestInNENList = initMST             #Tim trong NENList đâu là cây thông tốt nhất
    disOfBestMST = calculateDisInMST(bestInNENList)

    for i in range (0, int(NumberOfClusters)):      #duyet cac cluster
        initListNode = listNode
        for j in range(0, len(ClusterList[i])):     #duyet cac dinh trong cluster
            cityIndex = ClusterList[i][j]
            if cityIndex == MST['vertices'][i]:     #neu dinh da co trong MST   
                continue
            else:
                initListNode[i] = cityList[cityIndex]       

            graph = makeFullyGraph(initListNode)
            newMST, newMSTDis = kruskal(graph)

            disOfNewMST = calculateDisInMST(newMST)
            if disOfNewMST < disOfBestMST:
                disOfBestMST = disOfNewMST
                bestInNENList = newMST

            NENList.append([newMST, initListNode])

    return NENList, bestInNENList

initMST, initMSTDis, initMSTList = initialize()
NENList, bestMST = NEN(initMST, initMSTList)



print(str(len(NENList)))

# for i in range(0, len(NENList)):
#     MST = NENList[i][0]
#     edges = list(MST['edges'])
#     print("MST " + str(i))
#     for edge in edges:
#         weight, v1, v2 = edge
#         print(weight, end =" ")
#         print(v1, end = " ")
#         print(v2, end = " ")
#         print()
    

print("Best MST obtained: ")
edges = list(bestMST['edges'])
for edge in edges:
        weight, v1, v2 = edge
        print(weight, end =" ")
        print(v1, end = " ")
        print(v2, end = " ")
        print()

print("Sum of distance in best MST: ")
print(str(calculateDisInMST(bestMST)))
