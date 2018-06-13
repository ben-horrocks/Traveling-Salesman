#!/usr/bin/python3
import sys
import traceback

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq



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
        pass


    def branchAndBound( self, start_time, time_allowance=60.0 ):
        pass
        # start_time = time.time()
        # bssf = None
        # maxStoredBranches = 0
        # numsolutions = 0
        # branchnum = 1
        # pruned_states = 0
        # #Reduction
        # #Time Complexiy: O(x^2), where x is the length of the list of cities, since we need to iterate
        # # through the 2d matrix of values once
        # try:
        #     #Create 2d array with edges
        #     matrix_length = len(self._scenario._edge_exists)
        #     distance_matrix = [[math.inf for x in range(matrix_length)] for y in range(matrix_length)]
        #     cities = self._scenario.getCities()
        #     route = []
        #     route.append(cities[0])
        #     for x in range(matrix_length):
        #         for y in range(matrix_length):
        #             if x != y and self._scenario._edge_exists[x][y]:
        #                 cityone = cities[x]
        #                 distance_matrix[x][y] = cities[x].costTo(cities[y])
        #     #reduce matrix
        #     min_elements = []
        #     for row in range(len(distance_matrix)):
        #         min_element = min(distance_matrix[row])
        #         for element in range(matrix_length):
        #             distance_matrix[row][element] = distance_matrix[row][element] - min_element
        #         min_elements.append(min_element)
        #     #Check that each node is being arrived at
        #     for column in range(matrix_length):
        #         columnvalues = [distance_matrix[x][column] for x in range(matrix_length)]
        #         if 0.0 not in columnvalues:
        #             min_element = min(columnvalues)
        #             for element in range(matrix_length):
        #                 distance_matrix[element][column] = distance_matrix[element][column] - min_element
        #             min_elements.append(min_element)
        #     lowerbound = sum(min_elements)
        #     branches = []
        #     nonvisitednodes = [x for x in range(1, matrix_length)]
        #     currentnode = 0
        #
        #     #create branches
        #     #TIME COMPLEXITY: O(x * x*x)
        #     for column in range(matrix_length):
        #         if distance_matrix[currentnode][column] == math.inf:
        #             continue
        #         newlowerbound = lowerbound + distance_matrix[currentnode][column]
        #         newmatrix = [row[:] for row in distance_matrix]
        #         for row in range(matrix_length):
        #             newmatrix[row][column] = math.inf
        #         for newcolumn in range(matrix_length):
        #             newmatrix[currentnode][newcolumn] = math.inf
        #         newnonvisitednodes = list(nonvisitednodes)
        #         newnonvisitednodes.remove(column)
        #         newroute = list(route)
        #         newroute.append(cities[column])
        #         for row in range(matrix_length):
        #             if 0.0 not in newmatrix[row]:
        #                 min_element = min(newmatrix[row])
        #                 if min_element != math.inf:
        #                     for element in range(matrix_length):
        #                         newmatrix[row][element] = newmatrix[row][element] - min_element
        #                     newlowerbound = newlowerbound + min_element
        #         for newercolumn in range(matrix_length):
        #             columnvalues = [newmatrix[x][newercolumn] for x in range(matrix_length)]
        #             if 0.0 not in columnvalues:
        #                 min_element = min(columnvalues)
        #                 if min_element != math.inf:
        #                     for element in range(matrix_length):
        #                         distance_matrix[element][newercolumn] = distance_matrix[element][newercolumn] - min_element
        #                     newlowerbound = newlowerbound + min_element
        #         heapq.heappush(branches, (newlowerbound, branchnum, newmatrix, column, newnonvisitednodes, newroute))
        #         branchnum = branchnum + 1
        #     #Branch creations
        #     while len(branches) != 0:
        #         if maxStoredBranches < len(branches):
        #             maxStoredBranches = len(branches)
        #         timesofar = time.time() - start_time
        #         selectedbranch = heapq.heappop(branches)
        #         lowerbound = selectedbranch[0]
        #         matrix = selectedbranch[2]
        #         currentnode = selectedbranch[3]
        #         nonvisitednodes = selectedbranch[4]
        #         route = selectedbranch[5]
        #         if timesofar >= time_allowance:
        #             print("out of time")
        #             print("Max stored branches: ", maxStoredBranches)
        #             print("number of BSSF updates", numsolutions)
        #             print("total branch #", branchnum)
        #             print("number of pruned branches", pruned_states)
        #             if bssf == None:
        #                 bssf = {'cost': lowerbound,
        #                         'time': 60.0,
        #                         'count': numsolutions,
        #                         'soln': TSPSolution(route)}
        #
        #             return bssf
        #         if len(nonvisitednodes) == 0: #leaf node
        #             if matrix[currentnode][0] != math.inf:
        #                 cost = lowerbound + matrix[currentnode][0]
        #                 if bssf == None or bssf['cost'] > cost:
        #                     numsolutions = numsolutions + 1
        #                     bssf = {'cost': cost,
        #                             'time': time.time() - start_time,
        #                             'count': numsolutions,
        #                             'soln': TSPSolution(route)}
        #                     node = 0
        #                     while node < len(branches):
        #                         if branches[node][0] > cost:
        #                             del branches[node]
        #                             pruned_states = pruned_states + 1
        #                         else:
        #                             node = node + 1
        #                     print("Max stored branches: ", maxStoredBranches)
        #                     print("number of BSSF updates", numsolutions)
        #                     print("total branch #", branchnum)
        #                     print("number of pruned branches", pruned_states)
        #                     return bssf
        #         for column in range(matrix_length):
        #             if matrix[currentnode][column] == math.inf or column not in nonvisitednodes:
        #                 continue
        #             newlowerbound = lowerbound + matrix[currentnode][column]
        #             newmatrix = [row[:] for row in matrix]
        #             for row in range(matrix_length):
        #                 newmatrix[row][column] = math.inf
        #             for newcolumn in range(matrix_length):
        #                 newmatrix[currentnode][newcolumn] = math.inf
        #             newnonvisitednodes = list(nonvisitednodes)
        #             newnonvisitednodes.remove(column)
        #             newroute = list(route)
        #             newroute.append(cities[column])
        #             for row in range(matrix_length):
        #                 if 0.0 not in newmatrix[row]:
        #                     min_element = min(newmatrix[row])
        #                     if min_element != math.inf:
        #                         for element in range(matrix_length):
        #                             newmatrix[row][element] = newmatrix[row][element] - min_element
        #                         newlowerbound = newlowerbound + min_element
        #             for newercolumn in range(matrix_length):
        #                 columnvalues = [distance_matrix[x][newercolumn] for x in range(matrix_length)]
        #                 if 0.0 not in columnvalues:
        #                     min_element = min(columnvalues)
        #                     if min_element != math.inf:
        #                         for element in range(matrix_length):
        #                             distance_matrix[element][newercolumn] = distance_matrix[element][newercolumn] - min_element
        #                         newlowerbound = newlowerbound + min_element
        #             heapq.heappush(branches, (newlowerbound, branchnum, newmatrix, column, newnonvisitednodes, newroute))
        #             branchnum = branchnum + 1
        #     print("Max stored branches: ", maxStoredBranches)
        #     print("number of BSSF updates", numsolutions)
        #     print("total branch #", branchnum)
        #     print("number of pruned branches", pruned_states)
        #     return bssf
        # except:
        #     print("Got exception")
        #     print(sys.exc_info()[0])
        #     print(traceback.print_exc())
        #     raise

    def fancy( self, start_time, time_allowance=60.0 ):
        pass


