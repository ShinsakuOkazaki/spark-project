from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import time
from pyspark import SparkContext
from operator import add

if __name__ == "__main__":
    if len(sys.argv) != 2:
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
    
    pairApplicant = dictionary.filter(lambda x: x[0] == "applicant").collect()
    pairAnd = dictionary.filter(lambda x: x[0] == "and").collect()
    pairAttack = dictionary.filter(lambda x: x[0] == "attack").collect()
    pairProtein= dictionary.filter(lambda x: x[0] == "protein").collect()
    pairCar = dictionary.filter(lambda x: x[0] == "car").collect()

    print("Position of applicant: ", pairApplicant[0][1])
    print("Position of and: ", pairAnd[0][1])
    print("Position of attack: ", pairAttack[0][1])
    print("Position of protein: ", pairProtein[0][1])
    print("Position of car: ", pairCar[0][1])

    sc.stop()