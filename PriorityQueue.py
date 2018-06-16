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

class PriorityQueue:
	    #double[] values;
        #int[] positions;
        #int head;
        #int tail;
        def __init__(self, count):#Constant complexity, unless there are memory complications
            self.values = [None]*(count+5);
            self.positions = [None]*(count+5);
            self.head = count+4;
            self.tail = count+4;

        def add(self,item,itemPos):#double int
        #Complexity n where n is the number of elements in the queue. 
        #On average, it should be n/2, but obviously it is shortened to n
            if (self.head == self.tail):
                self.values[self.head] = item;
                self.positions[self.head] = itemPos;
                self.tail = (self.tail + 1) % len(self.values);
            else:
                pos = self.head;
                self.head = self.head - 1;
                if (self.head < 0):
                	self.head = len(self.values) - 1;
                posPrevious = self.head;
                while (pos != self.tail and self.values[pos] < item):
                    self.values[posPrevious] = self.values[pos];
                    self.positions[posPrevious] = self.positions[pos];
                    posPrevious = pos;
                    pos = (pos + 1) % len(self.values);
                self.values[posPrevious] = item;
                self.positions[posPrevious] = itemPos;
        def pop(self):#Constant complexity
            result = self.positions[self.head];
            self.head = (self.head + 1) % len(self.values);
            return result;
        def prePop(self):#Constant complexity
            return self.values[self.head];
        def canPop(self):#Constant complexity
            return self.head != self.tail;
        def count(self):
        #Complexity Constant
        	if(self.tail<self.head):
        		return len(self.values)-self.head+self.tail;
        	else:
        		return self.tail-self.head;