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

#All functions in Matrix math are Complexity N^2 where N is the number of rows (or columns, its square) in the matrix
#This is because All functions iterate over each member of the matrix and perform some operation
#Or perhaps do this twice.

#Standard matrix reduction. NaN is considered an empty square, infinity,w/e
#Returns the cost
def reduceMatrix(matrix):#double[][] REQUIRES SQUARE OR IT WILL BREAK
#Complexity N^2 where N is the number of rows (or columns, its square) in the matrix
    result = 0;
    matrixLength=len(matrix);
    for i in range (matrixLength):
        min = None;
        for j in range (matrixLength):
            if (matrix[i][j]!=None and (min==None or matrix[i][j] < min)):
                min = matrix[i][j];
        
        if (min == None):
        	min = 0;
        else:
            for j in range(len(matrix[i])):
                if (matrix[i][j]!=None):
                    matrix[i][j] -= min;
            result += min;
    
    for j in range (matrixLength):#same as above, but j iterates on outside.
        min = None;
        for i in range (matrixLength):
            if (matrix[i][j]!=None and (min==None or matrix[i][j] < min)):
                min = matrix[i][j];
        if (min == None):
        	min = 0;
        else:
            for i in range (matrixLength):
                if (matrix[i][j]!=None):
                    matrix[i][j] -= min;
            result += min;
    return result;
#Reduces but stores data
def reduceMatrixReversible(matrix, storedOrig, mins, rowRemove, columnRemove):#double[][], double[], double[], int, int
#Complexity N^2 where N is the number of rows (or columns, its square) in the matrix
    matrixLength=len(matrix);
    for i in range(matrixLength):
        storedOrig[i] = matrix[rowRemove][i];
        matrix[rowRemove][i] = None;
    for j in range(matrixLength):
        storedOrig[j + matrixLength] = matrix[j][columnRemove];
        matrix[j][columnRemove] = None;

    result = 0;
    for i in range(matrixLength):
        min=None;
        for j in range(matrixLength):# REQUIRES SQUARE
            if (matrix[i][j]!=None and (min==None or matrix[i][j] < min)):
                min = matrix[i][j];
        if (min == None):
        	min = 0;
        else:
            for j in range(matrixLength):
                if (matrix[i][j]!=None):
                    matrix[i][j] -= min;
            result += min;
        mins[i] = min;
    for j in range(matrixLength):
        min = None;
        for i in range(matrixLength):
            if (matrix[i][j]!=None and (min==None or matrix[i][j] < min)):
                min = matrix[i][j];
        if (min == None):
        	min = 0;
        else:
            for i in range(matrixLength):
                if (matrix[i][j]!=None):
                    matrix[i][j] -= min;
            result += min;
        mins[j+matrixLength] = min;
    return result;
#Undoes a reduction
def unReduceMatrix(matrix,storedOrig,mins,rowRemove,columnRemove):#double[][],double[],double[],int,int
    matrixLength=len(matrix);
    for j in range(matrixLength):
        if (mins[j + matrixLength] <= 0):
        	continue;
        for i in range(matrixLength):
        	if(matrix[i][j]!=None):
        		matrix[i][j] += mins[j + matrixLength];
    for i in range(matrixLength):
        if (mins[i] <= 0):
       		continue;
        for j in range(matrixLength):
        	if(matrix[i][j]!=None):
        		matrix[i][j] += mins[i];
    for j in range(matrixLength):
        matrix[j][columnRemove] = storedOrig[j + matrixLength];
    for i in range(matrixLength):
        matrix[rowRemove][i] = storedOrig[i];
#Copies all values to a matrix. Used before, but way too slow
def copyMatrix(matrix, toMatrix):#Complexity N^2 where N is the number of rows (or columns, its square) in the matrix
    matrixLength=len(matrix);
    for i in range(matrixLength):
        toMatrix[i] = [None]*len(matrix[i]);
        for j in range(len(matrix[i])):
            toMatrix[i][j] = matrix[i][j];
#Sums all numbers in matrix, was used in debugging
def sumMatrix(matrix):#Complexity N^2 where N is the number of rows (or columns, its square) in the matrix
    matrixLength=len(matrix);
    result = 0;
    for i in range(matrixLength):
        for j in range(matrixLength):
            if (matrix[i][j]==None):
            	continue;
            result += matrix[i][j];
    return result;
def matrixToString(matrix):#double[][]
    matrixLength=len(matrix);
    result = "";
    for i in range(matrixLength):
        for j in range(len(matrix[i])):
            result = result + matrix[i][j] + " ";
        result = result + "\n";
    return result;