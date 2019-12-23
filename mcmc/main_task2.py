from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import time
from pyspark import SparkContext
from operator import add
import time

# Function to make build array
def buildArray (listOfIndices):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    mysum = np.sum (returnVal)
    returnVal = np.divide (returnVal, mysum)
    return returnVal
def getCntSum(x1, x2):
    return [x1[0]+x2[0], np.add(x1[1], x2[1])]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="LogisticRegression")
    d_corpus = sc.textFile(sys.argv[1], 1)
    # Get important parts
    d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 18 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
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
    pre_tfs = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
    
    top100 = pre_tfs.filter(lambda x: x[0] == 'comp.graphics/37261').map(lambda x: (x[0], np.sort(x[1])[::-1][:100] * 20000.0)).collect()[0][1]

    print(top100)