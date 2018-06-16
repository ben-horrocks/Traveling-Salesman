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

class State:
    #public double[][] matrix;
    #public int[] path;
    #public int pos;
    #public double cost;

    def __init__(self,matrix,path, pos,cost):#double[][] ,  int[] , int, double
        self.matrix = matrix;
        self.path = path;
        self.pos = pos;
        self.cost = cost;
class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario

class StatePriorityQueue:
    #public State[] values;
    #public int head;
    #public int tail;
    def __init__(self, count):
    #Constant complexity, unless there are memory complications
        self.values = [None]*(count+5);
        self.head = count+3;
        self.tail = count+3;
    def add(self, item):#State
    #Complexity n where n is the number of elements in the queue. On average, it should be n/2, but 
    #obviously it is shortened to n
        #print("Start add head=",self.head," Tail=",self.tail);
        if (self.head == self.tail):
            self.head=self.head-1;
            if(self.head<0):
                self.head=len(self.values);
            self.values[self.head] = item;
            #self.tail = (self.tail + 1) % len(self.values);
        else:
            pos = self.head;
            nextHead = self.head;
            nextHead = nextHead - 1;
            if (nextHead < 0):
                nextHead = len(self.values) - 1;
            if (nextHead == self.tail):#This accounts for growing arrays
                print("Growing Array");
                newValues = [None]*(len(self.values) * 2);
                newTail = len(newValues) - 1;
                copyIterator = newTail;
                copyIterator2 = self.tail;
                while (copyIterator2 != self.head):
                    newValues[copyIterator] = self.values[copyIterator2];
                    copyIterator-=1;
                    copyIterator2-=1;
                    if (copyIterator2 < 0):
                        copyIterator2 = len(self.values) - 1;
                newValues[copyIterator] = self.values[self.head];
                self.values = newValues;
                self.tail = newTail;
                nextHead = copyIterator-1;
            self.head = nextHead;
            posPrevious = self.head;
            while (pos != self.tail and self.values[pos].cost/self.values[pos].pos < item.cost/item.pos):#Weight based on tree depth
                self.values[posPrevious] = self.values[pos];
                posPrevious = pos;
                pos = (pos + 1) % len(self.values);
            self.values[posPrevious] = item;
            #print("End add head=",self.head," Tail=",self.tail);

    def pop(self):#Constant complexity
        result = self.values[self.head];
        self.head = (self.head + 1) % len(self.values);
        return result;
    def canPop(self):#Constant complexity
        return self.head != self.tail;

    def count(self):#Complexity n where n is the number of elements in the queue
        tempHead = self.head;
        result = 0;
        while (tempHead != self.tail):
            result+=1;
            tempHead = (tempHead + 1) % len(self.values);
        return result;