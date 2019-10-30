from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import time
from pyspark import SparkContext
from operator import add

# Function to make build array
def buildArray (listOfIndices):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    mysum = np.sum (returnVal)
    returnVal = np.divide (returnVal, mysum)
    return returnVal

# Function to perform one-hot-encoding based on label ID
def oneHotEncoding(ID):
    if re.compile('^AU').match(ID):
        return 1
    else:
        return 0

# Function to calculate theta for each row of RDD
def calculateTheta(x, r_prev):
    return np.dot(x, r_prev)
# Function to calculate log part of llh for each row of RDD
def getLogPart(theta):
    return np.log(1 + np.exp(theta))
# Function to calculate multiplication of x and y.
def calculateXtY(x, y):
    return -np.multiply(x, y)
# Function to calculate multiplication of x and theta.
def getXMultiplyThetaPart(x, theta):
    return np.multiply(x, (np.exp(theta) / (1.0 + np.exp(theta))))
# Function to calculate multiplication of y and theta.
def getYMultiplyTheta(y, theta):
    return -np.multiply(y, theta)
# Function to get derivative.
def getDerivative(x, y, theta):
    XtY = calculateXtY(x, y)
    XTheta = getXMultiplyThetaPart(x, theta)
    derivative = np.add(XtY, XTheta)
    return derivative
# Function to get LLH
def getLLH(x, y, theta):
    Ytheta = getYMultiplyTheta(y, theta)
    logPart = getLogPart(theta)
    cost = np.add(Ytheta, logPart)
    return cost

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="LogisticRegression")
    d_corpus = sc.textFile(sys.argv[1], 1)
    # Get important parts
    d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    # Split in to words.
    d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    # Flat to all words in corpas
    allWords = d_keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x, 1))
    # Count each words.
    allCounts = allWords.reduceByKey(add)
    # Take to 20000
    topWords = allCounts.top(20000, key=lambda x: x[1])
    twentyK = sc.parallelize(range(20000))
    # Create dictionary.
    dictionary = twentyK.map (lambda x : (topWords[x][0], x))
    # Get only words in dictionally for each document
    allWords = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    allDictionaryWords = dictionary.join(allWords)
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    # Calculate term frequence.
    tfs = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

    # Make label 0 or 1.
    data = tfs.map(lambda x: (oneHotEncoding(x[0]), x[1])).cache()
                #.map(lambda x: (x[0], np.append(1.0, x[1])))


    
    num_iteration = 100
    learningRate = 0.01
    r_prev= np.full(20000, 0.01)
    count = 0
    cost_prev = sys.float_info.max
    precision = 0.00000001
    prev_stepsize = 1
    while(num_iteration > count and precision < prev_stepsize):
        start = time.time()

        # Calculate cost and derivative
        costAndDerivative = data.map(lambda x: (x[0], x[1], calculateTheta(x[1], r_prev)))\
                                .map(lambda x: (1, np.append(getLLH(x[1], x[0], x[2]), getDerivative(x[1], x[0], x[2]))))\
                                .reduceByKey(lambda x1, x2:np.add(x1, x2)).collect()
    # data.map(lambda x: (1, getCostAndDerivative(x[1], x[0], r_prev)))\
    #                         .reduceByKey(lambda x1, x2: np.add(x1, x2)).collect()

        cost_cur = costAndDerivative[0][1][0]
        dr = costAndDerivative[0][1][1:]


        # Set current weight 
        r_cur = np.subtract(r_prev, learningRate*dr)



        # Bold Driver logic
        if cost_cur < cost_prev:
            learningRate *= 1.05
        else:
            learningRate *= 0.5

        print("Iteration: ", count)
        print("W: " ,r_cur)
        print("Loss: ", cost_cur)

        count += 1

        # Calculate stepsize
        prev_stepsize = np.linalg.norm(np.subtract(r_prev, r_cur), 2)

        r_prev = r_cur

        cost_prev = cost_cur

        stop = time.time()
        duration = stop - start
        print("Duration", duration)
        
    sc.stop()