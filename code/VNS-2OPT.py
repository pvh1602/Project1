import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import time


#Open input file
infile = open('D:\\BKHN\\Project 1\\tuan_1\\eil51.tsp', 'r')

# Read instance header
Name = infile.readline().strip().split()[2] # NAME
Comment = infile.readline().strip().split()[2] # COMMENT
FileType = infile.readline().strip().split()[2] # TYPE
Dimension = infile.readline().strip().split()[2] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[2] # EDGE_WEIGHT_TYPE
infile.readline()

# Read node list
nodelist = []
N = int(Dimension)
for i in range(0, int(Dimension)):
    x,y = infile.readline().strip().split()[1:]
    nodelist.append([int(x), int(y)])

# Close input file
infile.close()

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
    
    # def routeFitness(self):
    #     if self.fitness == 0:
    #         self.fitness = 1 / float(self.routeDistance())
    #     return self.fitness

#tao ra 1 duong di(ca the)
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


cityList = []
for i in range(len(nodelist)):
    cityList.append(City(x = nodelist[i][0], y = nodelist[i][1], id = i))


def swapTwoOpt (route, i, k ):
    newRoute = []
    for j in range(0,i):
        newRoute.append(route[j])

    for j in range(k,i-1,-1):
        newRoute.append(route[j])

    for j in range(k+1,len(route)):
        newRoute.append(route[j])

    return newRoute

# newRoute = swapTwoOpt([1,2,3,4,5,6,7,8],2,4)

# for i in range(0,len(newRoute)):
#     print(newRoute[i], end = " ")

def neighborhoods_struture(route,k):
    bestDistance = Fitness(route).routeDistance()
    
    bestRoute = route
    n = len(route)
    #NeiK = []

    for i in range(0,n):
            
        newRoute = swapTwoOpt(route,i,(i+k)%n)
        #NeiK.append(newRoute)
        newDistance = Fitness(newRoute).routeDistance()
        if newDistance < bestDistance:
            bestRoute = newRoute
            bestDistance = newDistance
                           
           
    
    return bestRoute

# route = createRoute(cityList)


def runTwoOpt(route):
    bestDistance = Fitness(route).routeDistance()
    
    bestRoute = route
    improvement = 0
    while improvement < 20:
        for i in range(0,len(bestRoute)-1):
            for k in range(i+1,len(bestRoute)):
                newRoute = swapTwoOpt(bestRoute,i,k)
                newDistance = Fitness(newRoute).routeDistance()
                if newDistance < bestDistance:
                    
                    bestRoute = newRoute
                    bestDistance = newDistance
                    improvement = 0

        improvement = improvement + 1    
    
    return bestRoute

def VNSand2OPT(route):
    n = len(route)
    kmax = int(n/2)

    best_dis = Fitness(route).routeDistance()
    best_route = route
    k = 1 

    while k < (kmax+1):
        #Shaking
        new_route_in_Nk = neighborhoods_struture(best_route,k)

        #Local search 
        local_optimum = runTwoOpt(new_route_in_Nk)
        local_optimum_distance = Fitness(local_optimum).routeDistance()

        #more or not
        if local_optimum_distance < best_dis:
            best_dis = local_optimum_distance
            best_route = local_optimum
            k = 1 
        k = k + 1    
    
    return best_route

route = createRoute(cityList)

print("Initial route distance is: " + str(Fitness(route).routeDistance()))

bestRoute = VNSand2OPT(route)

for i in range(0, len(bestRoute)):
    print(City.__showDetail__(bestRoute[i]), end = " ")

print("best route distance found is: " + str(Fitness(bestRoute).routeDistance()))



        



