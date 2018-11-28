import numpy as np, random, operator
import time
import collections


#Open input file
infile = open('D:\\BKHN\\Project 1\\tuan_6\\Type_1_Small\\5eil51.clt', 'r')

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

#------------------------------------------------
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

#------------------------------------------------
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

cityList = []
for i in range(len(nodelist)):
    cityList.append(City(x = nodelist[i][0], y = nodelist[i][1], id = i))



cityInClusterList = []          #luu lai danh sach cac thanh pho trong moi cluster 
for i in range(int(NumberOfClusters)): 
    citiesInEachCluster = []        #type = list of City
    for j in range(0, len(ClusterList[i])):
        cityIndex = ClusterList[i][j]
        citiesInEachCluster.append(City(x = nodelist[cityIndex][0], y = nodelist[cityIndex][1], id = cityIndex))
    cityInClusterList.append(citiesInEachCluster)




#------------------------MAKE FULL GRAPH IN EACH CLUSTER------------------------
ListGraphInEachCluster = []
for i in range(int(NumberOfClusters)):
    graph = makeFullyGraph(cityInClusterList[i])   
    ListGraphInEachCluster.append(graph) 


#------------------------KRUSKAL ALGORITHM------------------------
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

#------------------------DIJKSTRA ALGORITHM------------------------

def dijsktra(graph, initial, CluTree):
    visited = {initial: 0}
    path = {}

    nodes = set(graph['vertices'])
    edges = list(graph['edges'])

    while nodes: 
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.remove(min_node)
        current_cost = visited[min_node]
        

        for edge in edges:
            weight, vertice1, vertice2 = edge
            if vertice1 == min_node:
                cost = current_cost + weight
                if vertice2 not in visited or cost < visited[vertice2]:
                    visited[vertice2] = cost
                    path[vertice2] = min_node
                    
    for to_vertice, from_vertice in path.items():
        CluTree['edges'].add((visited[to_vertice], from_vertice, to_vertice))
        CluTree['edges'].add((visited[to_vertice], to_vertice, from_vertice))
    

#------------------------BFS to calculate total cost from source vertex------------------------
def bfs(graph, root): 
    visited, queue = set(), collections.deque([root])
    visited.add(root)
    nodes = set(graph['vertices'])
    edges = list(graph['edges'])
    parent = np.zeros(len(graph['vertices']))
    cost = np.zeros(len(graph['vertices']))

    while queue: 
        vertice = queue.popleft()

        for edge in edges:
            weight, from_vertice, to_vertice = edge
            if from_vertice == vertice:
                if to_vertice not in visited:
                    visited.add(to_vertice)
                    queue.append(to_vertice)
                    parent[to_vertice] = from_vertice
                    cost[to_vertice] = cost[from_vertice] + weight
        
    totalCost = 0
    for i in range(len(cost)):
        totalCost = totalCost + cost[i]
    
    return totalCost

#-------------------INITIALIZE THE LIST OF NODE BETWEEN CLUSTER-------------------

def initListNodeBetweenCluster():

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
   
    return ListNodeInEachCluster

#------------------------MAKE CLUSTER TREE------------------------
def makeCluTree(ListNodeInEachCluster):
    CluTree = {
    'vertices':  [],
    'edges' : set([])
    }

    CluTree['vertices'] = np.arange(len(nodelist))

    graph = makeFullyGraph(ListNodeInEachCluster)

    initMST, sumOfDis = kruskal(graph)
    
    #add initMST cluster to CluTree
    for edge in list(initMST['edges']):
        weight, vertice1, vertice2 = edge
        CluTree['edges'].add((weight,vertice1,vertice2))
        CluTree['edges'].add((weight,vertice2,vertice1))
        

    for i in range(0, len(ListNodeInEachCluster)):
        initialVertexInEachCluster = ListNodeInEachCluster[i].showId()       
        dijsktra(ListGraphInEachCluster[i], initialVertexInEachCluster, CluTree)

    return CluTree


#-------------------RVNS-------------------

def RVNS():
    ListNodeInEachCluster = initListNodeBetweenCluster()
    CluTree = makeCluTree(ListNodeInEachCluster)
    initCost = bfs(CluTree, SoureVertex)

    bestListNode = ListNodeInEachCluster

    bestCluTree = CluTree
    bestCost = initCost

    k = 1
    while k < 3:
        newListNode = []
        for item in bestListNode:
            newListNode.append(item)

        if k == 1:
            #choose pi' != pi
            while True:
                idClu = random.randint(0, (int(NumberOfClusters)-1))
                idCity = random.randint(0, (len(cityInClusterList[idClu])-1))
                if newListNode[idClu].showId() != cityInClusterList[idClu][idCity].showId():
                    newListNode[idClu] = cityInClusterList[idClu][idCity]
                    break

            newCluTree = makeCluTree(newListNode)
            newCost = bfs(newCluTree, SoureVertex)
            if newCost < bestCost:
                bestCluTree = newCluTree
                bestListNode = newListNode
                continue

        if k == 2:
            #Choose pi' != pi in V1 and pj' != pj in V2
            while True:
                idClu1 = random.randint(0, (int(NumberOfClusters)-1))
                idCity1 = random.randint(0, (len(cityInClusterList[idClu1])-1))

                idClu2 = random.randint(0, (int(NumberOfClusters)-1))
                idCity2 = random.randint(0, (len(cityInClusterList[idClu2])-1))
                if idClu1 != idClu2:
                    if newListNode[idClu1].showId() != cityInClusterList[idClu1][idCity1].showId():
                        newListNode[idClu1] = cityInClusterList[idClu1][idCity1]
                        if newListNode[idClu2].showId() != cityInClusterList[idClu2][idCity2].showId():
                            newListNode[idClu2] = cityInClusterList[idClu2][idCity2]
                            break

            newCluTree = makeCluTree(newListNode)
            newCost = bfs(newCluTree, SoureVertex)
            if newCost < bestCost:
                bestCluTree = newCluTree
                bestListNode = ListNodeInEachCluster
                k = 1
                continue

        k = k + 1



    return bestCost, bestCluTree

def move(fromListNode, toListNode):
    for i in range(0, len(fromListNode)):
        toListNode[i] = fromListNode[i]
        
#---------------------FIND BEST NEIGHBOUR IN NEIGHBOURHOOD STRUCTURE--------------------

def neigbourhood_structure(ListNodeInEachCluster, CluTree, k):
    bestListNode = []
    for item in ListNodeInEachCluster:
        bestListNode.append(item)
    
    bestCluTree = makeCluTree(bestListNode)
    bestCost = bfs(bestCluTree, SoureVertex)


    newListNode = []
    for item in ListNodeInEachCluster:
        newListNode.append(item)

    if k == 1: 
        for i in range(0, int(NumberOfClusters)):
            for j in range(0, len(cityInClusterList[i])):
                if cityInClusterList[i][j].showId() != newListNode[i]:
                    newListNode[i] = cityInClusterList[i][j]
                    newCluTree = makeCluTree(newListNode)
                    newCost = bfs(newCluTree, SoureVertex)
                    #print("k = 1 " + str(newCost))
                    if newCost < bestCost:
                        move(newListNode, bestListNode)
                        bestCost = newCost
                        move(ListNodeInEachCluster, newListNode)
    
    if k == 2:
        MST, sumofDis = kruskal(makeFullyGraph(newListNode))
        edges = MST['edges']
        for edge in edges:
            weight, from_vertice, to_vertice = edge
            #find clusters which include 2 vertices
            idClu1 = 0
            idClu2 = 0
            idFromCity = 0
            idToCity = 0 
            for i in range(len(cityInClusterList)):
                for j in range(len(cityInClusterList[i])):
                    if from_vertice == cityInClusterList[i][j].showId():
                        idClu1 = i
                        idFromCity = j
                    if to_vertice == cityInClusterList[i][j].showId():
                        idClu2 = i
                        idToCity = j

            #forall another vertex in each cluster
            for i in range(len(cityInClusterList[idClu1])):
                if cityInClusterList[idClu1][i].showId() != cityInClusterList[idClu1][idFromCity].showId():
                    for j in range(len(cityInClusterList[idClu2])):
                        if cityInClusterList[idClu2][j].showId() != cityInClusterList[idClu2][idToCity].showId():
                            newListNode[idClu1] = cityInClusterList[idClu1][i]
                            newListNode[idClu2] = cityInClusterList[idClu2][j]
                            newCluTree = makeCluTree(newListNode)
                            newCost = bfs(newCluTree, SoureVertex)
                            #print("k=2 " + str(newCost))
                            if newCost < bestCost:
                                move(newListNode, bestListNode)
                                bestCost = newCost
                                move(ListNodeInEachCluster, newListNode)

    return bestCluTree, bestListNode, bestCost

#-----------------VNS------------------------

def VNS():
    
    #init solution
    #ListNodeInEachCluster = initListNodeBetweenCluster()
    ListNodeInEachCluster = []
    for i in range(int(NumberOfClusters)):
        idcity = random.randint(0,len(cityInClusterList[i])-1)
        ListNodeInEachCluster.append(cityInClusterList[i][idcity])



    bestListNode = []
    for item in ListNodeInEachCluster:
        bestListNode.append(item)
    
    bestCluTree = makeCluTree(bestListNode)
    bestCost = bfs(bestCluTree, SoureVertex)

    initCost = bestCost

    k = 1

    while k < 3:
        newCluTree, newListNode, newCost = neigbourhood_structure(bestListNode,bestCluTree,k)

        if newCost < bestCost:
            move(newListNode,bestListNode)
            bestCluTree = newCluTree
            bestCost = newCost
            k = 0
        
        k = k + 1

    return bestCluTree, bestListNode, bestCost, initCost

random.seed(10)



for i in range (10):
    random.seed(i)
    print("seed: " + str(i))
    CluTree, ListNodeBetweenCluster, cost, initCost = VNS()
    print("best cost: " + str(cost))
    print("init cost: " + str(initCost))



            

