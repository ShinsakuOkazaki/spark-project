from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import time
from pyspark import SparkContext
from operator import add

def oneHotEncoding(ID):
    if re.compile('^AU').match(ID):
        return 1
    else:
        return 0

def buildArray (listOfIndices):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    mysum = np.sum (returnVal)
    returnVal = np.divide (returnVal, mysum)
    return returnVal

def getType1AndType2(true, pred):
    if true == 1.0 and pred == 1.0:
        TP = 1.0
    else:
        TP = 0.0
    if true == 0.0 and pred == 0.0:
        TN = 1.0
    else:
        TN = 0.0
    if true == 1.0 and pred == 0.0:
        FN = 1.0
    else:
        FN = 0.0
    if true == 0.0 and pred == 1.0:
        FP = 1.0
    else:
        FP = 0.0
    
    return np.array([TP, TN, FN, FP])

def getPreMargin(scores, yi_scores):
    # max(0, xw - xw_true) + delta
    return np.maximum(0.0, np.subtract(scores, yi_scores) + 1)

def changeToZero(preMargins, y):
    preMargins[y] = 0.0
    return preMargins

def getBinary(margins):
    margins[margins > 0.0] = 1.0
    return margins

def changeToRowSum(binary, y, row_sum):
    binary[y] = - row_sum
    return binary

def getDw(x, b):
    return np.array([x * b[0], x * b[1]])

def getF1score(TP, TN, FN, FP):
    if TP == 0:
        return 0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="LogisticRegression")
    read_start = time.time()
    d_corpus = sc.textFile(sys.argv[1], 1)
    read_stop = time.time()
    read_duration = read_stop - read_start
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
    data = tfs.map(lambda x: (oneHotEncoding(x[0]), x[1]))

    # Get sample number of train data
    num_train = data.count()

    # Calculate mu.
    mu = data.map(lambda x: (1, x[1]))\
        .reduceByKey(lambda x1, x2: np.add(x1, x2)).collect()[0][1]
    mu = np.divide(mu, num_train)
    # Calculate variance.
    variance = data.map(lambda x: (1, np.square(np.subtract(x[1], mu))))\
                .reduceByKey(lambda x1, x2: np.add(x1, x2)).collect()[0][1]
    variance = np.divide(variance, num_train)

    # Get the column index whose variance is top 10000
    label = [(i, variance[i]) for i in range(len(variance))]
    sortedColumn = sorted(label, key = lambda x: x[1], reverse=True)
    topColumn = sortedColumn[:10000]
    columnIdx = [topColumn[i][0] for i in range(len(topColumn))]

    # Reduce dimention.
    data = data.map(lambda x: (x[0], x[1][columnIdx])).cache()

    def svm_loss(W, reg):
        # Set initial dW as zero
        dW = np.zeros(W.shape)
        # To get scores do dot product of X (n, 200000) and W (200000, 2)
        scores = data.map(lambda x: (x[0], x[1], np.dot(x[1], W)))
        # Select one correctly categorized point.
        yi_scores = scores.map(lambda x: (x[0], x[1], x[2], x[2][x[0]]))
        # Calculate margin and ignore the correctlly classified points by setting them 0.
        preMargins = yi_scores.map(lambda x: (x[0], x[1], getPreMargin(x[2], x[3])))
        margins = preMargins.map(lambda x: (x[0], x[1], changeToZero(x[2], x[0])))
        # Calculate loss by taking mean of distance from mergin to point
        loss = margins.map(lambda x: (1, (np.sum(x[2]), 1.0)))\
                    .reduceByKey(lambda x1, x2: (x1[0] + x2[0], x1[1] + x2[1]))
        loss = loss.collect()[0][1]
        loss = loss[0] / loss[1]
        # Set binary by setting one if the distance from margin is over than zero.
        binary = margins.map(lambda x: (x[0], x[1], getBinary(x[2])))
        # Set the correctly classified case to negative one or zero
        row_sum = binary.map(lambda x: (x[0], x[1], x[2], np.sum(x[2])))
        trueBinary = row_sum.map(lambda x: (x[0], x[1], changeToRowSum(x[2], x[0], x[3])))
        # Get derivative of yx part, if j = yi, -x, otherwise x 
        dW = trueBinary.map(lambda x: (1,  getDw(x[1], x[2])))\
                        .reduceByKey(lambda x1, x2: np.add(x1, x2))
        dW = dW.collect()[0][1].reshape(-1, 2)
        # Average
        dW /= num_train
        dW += reg * W
        
        return loss, dW

    # Set initial weights (10000, 2)
    W = np.full((10000, 2), 0.1)
    num_iteration = 100
    learningRate = 0.01
    count = 0
    reg = 0.1
    loss_prev =  sys.float_info.max
    train_start = time.time()
    while num_iteration > count:
        
        loss, dW = svm_loss(W, reg)
        
        W = np.subtract(W, learningRate*dW)
        
        # Bold Driver logic
        if loss < loss_prev:
            learningRate *= 1.05
        else:
            learningRate *= 0.5

        print("Iteration: ", count)
        print("W: " ,W)
        print("Loss: ", loss)
        
        loss_prev = loss

        count += 1

    #W_true = W[np.arange(len(W)), np.argmax(W, axis = 1)]
    train_stop = time.time()
    train_duration = train_stop - train_start

    ################ Test ################
    # Get test data ready for prediction.
    test_start = time.time()
    test_corpus = sc.textFile(sys.argv[2], 1)
    test_keyAndText = test_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    test_keyAndListOfWords = test_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    test_allWords = test_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    test_allDictionaryWords = dictionary.join(test_allWords)
    test_justDocAndPos = test_allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
    test_allDictionaryWordsInEachDoc = test_justDocAndPos.groupByKey()
    test_tfs = test_allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
    test = test_tfs.map(lambda x: (oneHotEncoding(x[0]), x[1][columnIdx], x[0]))
    prediction = test.map(lambda x: (x[0], np.dot(x[1], W), x[2]))\
                 .map(lambda x: (x[0], np.argmax(x[1]), x[2]))

    type1AndType2 = prediction.map(lambda x: (x[2], getType1AndType2(x[0], x[1])))
    matric = type1AndType2.map(lambda x: (1, x[1])).reduceByKey(lambda x1, x2: np.add(x1, x2)).collect()

    TP = matric[0][1][0]
    TN = matric[0][1][1]
    FN = matric[0][1][2]
    FP = matric[0][1][3]

    f1_score = getF1score(TP, TN, FN, FP)
    test_stop = time.time()
    test_duration = test_stop - test_start
    print("TP: ", TP)
    print("TN: ", TN)
    print("FN: ", FN)
    print("FP: ", FP)
    print("F score: ",f1_score)
    print("Read time: ", read_duration)
    print("Train time: ", train_duration)
    print("Test time: ", test_duration)