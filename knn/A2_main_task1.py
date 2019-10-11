from __future__ import print_function

import sys
import re

import numpy as np
from numpy import dot
from numpy.linalg import norm

from pyspark import SparkContext


def stringVector (x):
	returnVal= str(x[0])
	for j in x[1]:
  		returnVal += ','+ str(j)
   	return returnVal
      

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: job <inputFile> <outputFolder>", file=sys.stderr)
        print(len(sys.argv))
        print(sys.argv[1] + '  ' + sys.argv[2])
        exit(-1)
	
    
    sc = SparkContext(appName="Assignment-2")

	# First load up all of the documents in the corpus
    corpus = sc.textFile(sys.argv[1])

	# Assumption: Each document is stored in one line of the text file
	# We need this count later ... 
    numberOfDocs = corpus.count()

	# Each entry in validLines will be a line from the text file
    validLines = corpus.filter(lambda x : 'id' in x and 'url=' in x)

	# Now, we transform it into a set of (docID, text) pairs
    keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 

	# Now, we split the text in each (docID, text) pair into a list of words
	# After this step, we have a data set with 
	# (docID, ["word1", "word2", "word3", ...])
	# We use a regular expression here to make 
	# sure that the program does not break down on some of the documents

    regex = re.compile('[^a-zA-Z]')
	# # remove all non letter characters
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split())) 

	# Now get the top 20,000 words... 
	# first change (docID, ["word1", "word2", "word3", ...])
	# to ("word1", 1) ("word2", 1)...
    allWords = ???

	# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
    allCounts = allWords.???

	# Get the top 20,000 words in a local array in a sorted format based on frequency 
    topWords = allCounts.???

	# We'll create a RDD that has a set of (word, dictNum) pairs
	# start by creating an RDD that has the number 0 through 20000
	# 20000 is the number of words that will be in our dictionary
    twentyK = sc.parallelize(range(20000))

	# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1) 
	# ("NextMostCommon", 2), ...
	# the number will be the spot in the dictionary used to tell us 
	# where the word is located
    dictionary = twentyK.map (lambda x : (topWords[x][0], x))

	# Next step, we get a RDD that has, for each 
	# (docID, ["word1", "word2", "word3", ...]),
	# ("word1", docID), ("word2", docId), ...
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

	# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = ???.join(???)

	# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.???

	# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
	
    # Now, we can store this data for later use. 
    forCSV= allDictionaryWordsInEachDoc.map(lambda x: (x[0], np.array(map(str, x[1]))))
	
    forCSV= forCSV.map(lambda x: stringVector(x))
	
    print(forCSV.take(2))
	# ['418384,27,8,8,8,13,10449,640,15243,46,3213,45,224,489,183,273,5841,63,32,28,12,12,4062,4062,4062,221,2406,604,2673,2673,161,88,989,228,228,228,228,228,241,241,241,241,241,241,406,25,450,450,114,12279,272,11,11,39,39,281,365,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14508,14508,14508,14508,1285,5,5,5,5,4665,10618,10618,10618,1077,76,5004,38,283,176,2,2,2,2,2,1258,1258,1258,16,22,22,434,20,2998,24,335,15800,247,103,4085,14,14,14,61,15,101,546,1676,381,229,984,16489,6748,134,134,16170,16170,16170,16170,2845,23,1081,33,316,2990,85,242,2761,4126,2858,1049,1352,94,14559,10,10,10,10,10,14650,4256,3646,3646,3646,75,5126,4,4,4,4,2843,2281,3684,53,601,270,3,3,3,3,3,3,3,474,2046,6153,3819,1,1,1,1,1,1,1,1,1,1,1,1,1,2615']

	# TODO: Uncomment this line to save the file. 
    forCSV.saveAsTextFile(sys.argv[2]+"_allDictionaryWordsInEachDoc")
	
    sc.stop()
