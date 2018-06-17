#!/usr/bin/python3
import sys
import traceback

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF








import time
import numpy as np
from TSPClasses import *
import heapq
import random
import copy


# class that handles the matrix used for the node state things
class State:
    def __init__(self, cities, numCities):
        self.listOfCities = cities
        # make the shape of the N*N matrix for the start state
        self.numCities = numCities
        self.matrix = [[self.listOfCities[j].costTo(self.listOfCities[i]) for i in range(numCities)] for j in
                       range(numCities)]

    def getCities(self):
        return self.listOfCities

    # hopefully makes it so any time i check the num cities it doesnt have to recount
    # so BigO(1)
    def size(self):
        return self.numCities

    # BigO(1)
    def getAt(self, fromNode, toNode):
        return self.matrix[self.listOfCities.index(fromNode)][self.listOfCities.index(toNode)]


'''
This class stores the matrix of costs
as well as the path of cities that came before it
as well as its own path length and its bound

By having the bound kept inside the node, It is easy to sort it in relation to other nodes
and quickly compare it to the best solution so far and see if you can prune it

it also overides the >,<,=>,<= operators so that the queue knows how to sort it
'''


class TSPNode:

    def __init__(self, state, aPath=[], pathLength=0):

        self.state = state
        self.path = copy.copy(aPath)
        self.pathLength = pathLength

        self.bound = self.computeBound()

    # O(N^2) since of double for loop
    # where N is number of cities
    def computeBound(self):

        # gotta love nesting if/for statements
        shortest = 0
        for city1 in self.state.listOfCities:
            short = math.inf
            if city1 not in self.path:
                for city2 in self.state.listOfCities:
                    if city2 not in self.path and city1 != city2:
                        # if no short path has been found(math.inf) or a cheaper path is found
                        if short == math.inf or self.state.getAt(city1, city2) < short:
                            short = self.state.getAt(city1, city2)
            if short != math.inf:
                shortest += short
        return (shortest + self.pathLength)

    # O(N^2) because it calls compute bound
    def addVertex(self, vertex):
        # add vertex, update length, and set the new bound
        if len(self.path):
            self.pathLength += self.state.getAt(self.path[-1], vertex)
        else:
            self.pathLength = 0

        self.path.append(vertex)

        self.bound = self.computeBound()

    # Make comparisons easier
    # And also makes it auto sort when put into the queue
    def __lt__(self, otherNode):
        return self.bound < otherNode.bound

    def __le__(self, otherNode):
        return self.bound <= otherNode.bound

    def __gt__(self, otherNode):
        return self.bound > otherNode.bound

    def __ge__(self, otherNode):
        return self.bound >= otherNode.bound




class Tour:
    def __init__(self, prevTour):
        self.tour = copy.deepcopy(prevTour)
        self.tourDistance = 0

        # Worst case O(N!) since the only possible path is the last one it will find. Or there will be none at all, and so it ends up
        # searching every possibility
    def recursiveFindFirstOptimal(self, currNode, startState):

        if len(currNode.path) == startState.size():
            if startState.getAt(currNode.path[-1], currNode.path[0]) != math.inf:
                # Yes it's better so it's our new optimal tour
                optimalTour = currNode
                return (True, optimalTour)

        listOCities =  startState.getCities()
        listOfCosts = []

        for city in listOCities:
            listOfCosts.append(currNode.path[-1].costTo(city))

        sortedByDistCities = [city for _,city in sorted(zip(listOfCosts,listOCities))]
        listOfCosts.sort()
        # Find first reachable city
        for i in range (0,len(sortedByDistCities)):
            node = sortedByDistCities[i]
            if node not in currNode.path and \
                    listOfCosts[i] != math.inf:
                newNode = TSPNode(currNode.state,
                                  currNode.path,
                                  currNode.pathLength)
                newNode.addVertex(node)
                # self.childStatesCreated += 1
                result = self.recursiveFindFirstOptimal(newNode, startState)
                if (result[0] == True):
                    return result
        return [False, False]


    def initShuffle(self):

        startState = State(self.tour, len(self.tour))
        optimalTour = None
        initBssfNode = TSPNode(startState)
        initBssfNode.addVertex(startState.getCities()[0])

        result = self.recursiveFindFirstOptimal(initBssfNode, startState)
        if result[0] == True:
            optimalTour = result[1]

        else:
            print("NO route exists. mesa sad man")

        self.tour = optimalTour.path
        return

        initCities = []
        random.shuffle(self.tour)
        citesLeft = copy.deepcopy(self.tour)
        initCities.append(citesLeft.pop())
        while len(citesLeft) > 0:
            random.shuffle(citesLeft)
            index = 0
            while True:
                if len(citesLeft) > 0 or initCities[-1].costTo(citesLeft[index]) != math.inf:
                    if initCities[-1].costTo(citesLeft[index]) != math.inf:
                        initCities.append(citesLeft.pop(index))
                        break
                    else:
                        index += 1
                else:
                    citesLeft.append(initCities.pop(random.int(0,len(initCities)-1)))
                    break






    def getTour(self):
        return self.tour

    def getCity(self, index):
        return self.tour[index]

    def setCity(self, newPos, city):
        self.tour[newPos] = city
        self.tourDistance = 0

    def getDistance(self):
        if self.tourDistance == 0:
            tourDist = 0
            for i in range(0, len(self.tour)):
                startCity = self.tour[i]
                if i + 1 < len(self.tour):
                    destCity = self.tour[i + 1]
                else:
                    destCity = self.tour[0]
                tourDist += startCity.costTo(destCity)
            self.tourDistance = tourDist
        return self.tourDistance

    def getTourSize(self):
        return len(self.tour)





class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario






    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour
        </summary>
        <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (
not counting initial BSSF estimate)</returns> '''
    def defaultRandomTour( self, start_time, time_allowance=60.0 ):

        results = {}


        start_time = time.time()

        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        while not foundTour:
            # create a random permutation
            perm = np.random.permutation( ncities )

            #for i in range( ncities ):
                #swap = i
                #while swap == i:
                    #swap = np.random.randint(ncities)
                #temp = perm[i]
                #perm[i] = perm[swap]
                #perm[swap] = temp

            route = []

            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )

            bssf = TSPSolution(route)
            #bssf_cost = bssf.cost()
            #count++;
            count += 1

            #if costOfBssf() < float('inf'):
            if bssf.costOfRoute() < np.inf:
                # Found a valid route
                foundTour = True
        #} while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
        #timer.Stop();

        results['cost'] = bssf.costOfRoute() #costOfBssf().ToString();                          // load results array
        results['time'] = time.time() - start_time
        results['count'] = count
        results['soln'] = bssf

       # return results;
        return results



    def greedy( self, start_time, time_allowance=60.0 ):

        start_time = time.time()
        cities = self._scenario.getCities()
        numCities = len(cities)

        intermediateSolutions = 0
        temp = 10000
        coolingRate = .01
        currentTour = Tour(cities)
        currentTour.initShuffle()

        best = Tour(currentTour.getTour())
        bssf = {'cost': best.getDistance(),
                'time': time.time() - start_time,
                'count': intermediateSolutions,
                'soln': TSPSolution(best.getTour())}

        return bssf


    def branchAndBound( self, start_time, time_allowance=60.0 ):

        start_time = time.time()
        bssf = None
        maxStoredBranches = 0
        numsolutions = 0
        branchnum = 1
        pruned_states = 0
        #Reduction
        #Time Complexiy: O(x^2), where x is the length of the list of cities, since we need to iterate
        # through the 2d matrix of values once
        try:
            #Create 2d array with edges
            matrix_length = len(self._scenario._edge_exists)
            distance_matrix = [[math.inf for x in range(matrix_length)] for y in range(matrix_length)]
            cities = self._scenario.getCities()
            route = []
            route.append(cities[0])
            for x in range(matrix_length):
                for y in range(matrix_length):
                    if x != y and self._scenario._edge_exists[x][y]:
                        cityone = cities[x]
                        distance_matrix[x][y] = cities[x].costTo(cities[y])
            #reduce matrix
            min_elements = []
            for row in range(len(distance_matrix)):
                min_element = min(distance_matrix[row])
                for element in range(matrix_length):
                    distance_matrix[row][element] = distance_matrix[row][element] - min_element
                min_elements.append(min_element)
            #Check that each node is being arrived at
            for column in range(matrix_length):
                columnvalues = [distance_matrix[x][column] for x in range(matrix_length)]
                if 0.0 not in columnvalues:
                    min_element = min(columnvalues)
                    for element in range(matrix_length):
                        distance_matrix[element][column] = distance_matrix[element][column] - min_element
                    min_elements.append(min_element)
            lowerbound = sum(min_elements)
            branches = []
            nonvisitednodes = [x for x in range(1, matrix_length)]
            currentnode = 0

            #create branches
            #TIME COMPLEXITY: O(x * x*x)
            for column in range(matrix_length):
                if distance_matrix[currentnode][column] == math.inf:
                    continue
                newlowerbound = lowerbound + distance_matrix[currentnode][column]
                newmatrix = [row[:] for row in distance_matrix]
                for row in range(matrix_length):
                    newmatrix[row][column] = math.inf
                for newcolumn in range(matrix_length):
                    newmatrix[currentnode][newcolumn] = math.inf
                newnonvisitednodes = list(nonvisitednodes)
                newnonvisitednodes.remove(column)
                newroute = list(route)
                newroute.append(cities[column])
                for row in range(matrix_length):
                    if 0.0 not in newmatrix[row]:
                        min_element = min(newmatrix[row])
                        if min_element != math.inf:
                            for element in range(matrix_length):
                                newmatrix[row][element] = newmatrix[row][element] - min_element
                            newlowerbound = newlowerbound + min_element
                for newercolumn in range(matrix_length):
                    columnvalues = [newmatrix[x][newercolumn] for x in range(matrix_length)]
                    if 0.0 not in columnvalues:
                        min_element = min(columnvalues)
                        if min_element != math.inf:
                            for element in range(matrix_length):
                                distance_matrix[element][newercolumn] = distance_matrix[element][newercolumn] - min_element
                            newlowerbound = newlowerbound + min_element
                heapq.heappush(branches, (newlowerbound, branchnum, newmatrix, column, newnonvisitednodes, newroute))
                branchnum = branchnum + 1
            #Branch creations
            while len(branches) != 0:
                if maxStoredBranches < len(branches):
                    maxStoredBranches = len(branches)
                timesofar = time.time() - start_time
                selectedbranch = heapq.heappop(branches)
                lowerbound = selectedbranch[0]
                matrix = selectedbranch[2]
                currentnode = selectedbranch[3]
                nonvisitednodes = selectedbranch[4]
                route = selectedbranch[5]
                if timesofar >= time_allowance:
                    print("out of time")
                    print("Max stored branches: ", maxStoredBranches)
                    print("number of BSSF updates", numsolutions)
                    print("total branch #", branchnum)
                    print("number of pruned branches", pruned_states)
                    if bssf == None:
                        bssf = {'cost': lowerbound,
                                'time': 60.0,
                                'count': numsolutions,
                                'soln': TSPSolution(route)}

                    return bssf
                if len(nonvisitednodes) == 0: #leaf node
                    if matrix[currentnode][0] != math.inf:
                        cost = lowerbound + matrix[currentnode][0]
                        if bssf == None or bssf['cost'] > cost:
                            numsolutions = numsolutions + 1
                            bssf = {'cost': cost,
                                    'time': time.time() - start_time,
                                    'count': numsolutions,
                                    'soln': TSPSolution(route)}
                            node = 0
                            while node < len(branches):
                                if branches[node][0] > cost:
                                    del branches[node]
                                    pruned_states = pruned_states + 1
                                else:
                                    node = node + 1
                            print("Max stored branches: ", maxStoredBranches)
                            print("number of BSSF updates", numsolutions)
                            print("total branch #", branchnum)
                            print("number of pruned branches", pruned_states)
                            return bssf
                for column in range(matrix_length):
                    if matrix[currentnode][column] == math.inf or column not in nonvisitednodes:
                        continue
                    newlowerbound = lowerbound + matrix[currentnode][column]
                    newmatrix = [row[:] for row in matrix]
                    for row in range(matrix_length):
                        newmatrix[row][column] = math.inf
                    for newcolumn in range(matrix_length):
                        newmatrix[currentnode][newcolumn] = math.inf
                    newnonvisitednodes = list(nonvisitednodes)
                    newnonvisitednodes.remove(column)
                    newroute = list(route)
                    newroute.append(cities[column])
                    for row in range(matrix_length):
                        if 0.0 not in newmatrix[row]:
                            min_element = min(newmatrix[row])
                            if min_element != math.inf:
                                for element in range(matrix_length):
                                    newmatrix[row][element] = newmatrix[row][element] - min_element
                                newlowerbound = newlowerbound + min_element
                    for newercolumn in range(matrix_length):
                        columnvalues = [distance_matrix[x][newercolumn] for x in range(matrix_length)]
                        if 0.0 not in columnvalues:
                            min_element = min(columnvalues)
                            if min_element != math.inf:
                                for element in range(matrix_length):
                                    distance_matrix[element][newercolumn] = distance_matrix[element][newercolumn] - min_element
                                newlowerbound = newlowerbound + min_element
                    heapq.heappush(branches, (newlowerbound, branchnum, newmatrix, column, newnonvisitednodes, newroute))
                    branchnum = branchnum + 1
            print("Max stored branches: ", maxStoredBranches)
            print("number of BSSF updates", numsolutions)
            print("total branch #", branchnum)
            print("number of pruned branches", pruned_states)
            return bssf
        except:
            print("Got exception")
            print(sys.exc_info()[0])
            print(traceback.print_exc())
            raise


    def acceptanceProb(self,currEnergy,newEnergy,temp):

        #CHECK IF ONE OF THEM HAS INFINITE ENERGY
        if newEnergy == math.inf:
            return -1

        if newEnergy < currEnergy:
            return 1
        else:
            return math.exp((currEnergy-newEnergy)/temp)

    def fancy( self, start_time, time_allowance=60.0 ):
        start_time = time.time()
        cities =  self._scenario.getCities()
        numCities = len(cities)

        intermediateSolutions = 0
        temp = 10000
        coolingRate = 1.5/ pow(numCities,2)
        currentTour = Tour(cities)
        currentTour.initShuffle()

        best = Tour(currentTour.getTour())
        bssf = {'cost': best.getDistance(),
                'time': time.time() - start_time,
                'count': intermediateSolutions,
                'soln': TSPSolution( best.getTour())}

        while temp > 1:
            newTour = Tour(currentTour.getTour())

            randPos1 =  random.randint(0,numCities - 1)
            randPos2 = random.randint(0, numCities - 1)

            citySwap1 = newTour.getCity(randPos1)
            citySwap2 = newTour.getCity(randPos2)

            newTour.setCity(randPos2,citySwap1)
            newTour.setCity(randPos1,citySwap2)

            currentEnergy = currentTour.getDistance()
            newEnergy = newTour.getDistance()



            if self.acceptanceProb(currentEnergy,newEnergy,temp) > random.random():
                currentTour = Tour(newTour.getTour())

            if int(currentTour.getDistance()) < int(best.getDistance()):
                best = Tour(currentTour.getTour())
                intermediateSolutions += 1
                bssf = {'cost': best.getDistance(),
                        'time': time.time() - start_time,
                        'count':   intermediateSolutions,
                        'soln':   TSPSolution(  best.getTour())}


            temp *= 1-coolingRate
        bssf = {'cost': best.getDistance(),
        'time': time.time() - start_time,
        'count': intermediateSolutions,
        'soln': TSPSolution( best.getTour())}
        return bssf





