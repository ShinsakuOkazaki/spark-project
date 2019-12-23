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


def getProbs (checkParams, x, pi, log_allMus):
    if checkParams == True:
            if x.shape [0] != log_allMus.shape [1]:
                    raise Exception ('Number of words in doc does not match')
            if pi.shape [0] != log_allMus.shape [0]:
                    raise Exception ('Number of document classes does not match')
            if not (0.999 <= np.sum (pi) <= 1.001):
                    raise Exception ('Pi is not a proper probability vector')
            for i in range(log_allMus.shape [0]):
                    if not (0.999 <= np.sum (np.exp (log_allMus[i])) <= 1.001):
                            raise Exception ('log_allMus[' + str(i) + '] is not a proper probability vector')
    
    # to ensure that we don't have any underflows, we will do
    # all of the arithmetic in "log space". Specifically, according to
    # the Multinomial distribution, in order to compute
    # Pr[x | class j] we need to compute:
    #
    #       pi[j] * prod_w allMus[j][w]^x[w]
    #
    # If the doc has a lot of words, we can easily underflow. So
    # instead, we compute this as:
    #
    #       log_pi[j] + sum_w x[w] * log_allMus[j][w]
    #
    allProbs = np.log (pi)
    #
    # consider each of the classes, in turn
    for i in range(log_allMus.shape [0]):
            product = np.multiply (x, log_allMus[i])
            allProbs[i] += np.sum (product)
    #
    # so that we don't have an underflow, we find the largest
    # logProb, and then subtract it from everyone (subtracting
    # from everything in an array of logarithms is like dividing
    # by a constant in an array of "regular" numbers); since we
    # are going to normalize anyway, we can do this with impunity
    #
    biggestLogProb = np.amax (allProbs)
    allProbs -= biggestLogProb
    #
    # finally, get out of log space, and return the answer
    #
    allProbs = np.exp (allProbs)
    return allProbs / np.sum (allProbs)

def getCntSum(x1, x2):
    return [x1[0]+x2[0], np.add(x1[1], x2[1])]

def mapToLabelVec(label):
    label_vec = np.zeros(20)
    label_idx = np.argwhere(labels == label)
    label_vec[label_idx] = 1.0
    return label_vec

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
    tfs = allDictionaryWordsInEachDoc.map(lambda x: (x[0][:x[0].index("/")], buildArray(x[1]))).cache()
    labels = tfs.map(lambda x: x[0]).distinct().collect()
    labels = np.array(labels)

    alpha = np.full(20, 0.5)
    beta = np.full((20, 20000), 0.5)
    total = tfs.count()
    mu = np.zeros((20, 20000))
    pi = np.random.dirichlet(alpha)
    for j in range(mu.shape[0]):
            mu[j] = np.random.dirichlet(beta[j])
    
    train_time_start = time.time()
    for i in range(200):
        iter_time_start = time.time()
        
        # Calculate probality vector of c
        prob_c = tfs.map(lambda x: (x[0], x[1],getProbs(False, x[1], pi, np.log(mu)))).cache()
        # Take mu from categorical destribution
        prob_c_categorical = prob_c.map(lambda x: (x[0], x[1], np.random.multinomial(1, x[2])))
        # Assign label to each document
        prob_c_assigned = prob_c_categorical.map(lambda x: (x[0], x[1], np.nonzero(x[2])[0][0])).cache()
        # Get count of each assigned class and sum of corresponding tfs.
        cntc_sumx = prob_c_assigned.map(lambda x: (x[2], [1.0 ,x[1]])).reduceByKey(getCntSum)

        cnt_sum = cntc_sumx.collect()

        # Update alpha
        # updata with mu
        cnt_true = [0.0] * 20 
        for k in range(len(cnt_sum)):
            cnt_true[cnt_sum[k][0]] = cnt_sum[k][1][0] / total
        cnt_true = np.array(cnt_true)
        alpha = np.add(alpha, cnt_true)
        
        # Update beta
        sum_x_true = [np.zeros(20000)] * 20 
        for l in range(len(cnt_sum)):
            sum_x_true[cnt_sum[l][0]] = cnt_sum[l][1][1]
        sum_x_true = np.array(sum_x_true)
        beta = np.add(beta, sum_x_true)

        # Hyper Parameter 
        pi = np.random.dirichlet(alpha)
        # mu = np.array([np.random.dirichlet(beta) for i in range(20)])
        for j in range(mu.shape[0]):
            mu[j] = np.random.dirichlet(beta[j])
        
        print("Iteration: ", i)
        print("Count of each category: ", cnt_true * total )
        print("Iteration time: ", time.time() - iter_time_start)
    
    # Get top 50 words belonging to predicted class.
    top_idx = np.argsort(-np.log(mu))[:,:50]
    dic = dictionary.collect()
    words = np.array(dic)[:, 0]
    top_50_words = np.apply_along_axis(lambda x: words[x], 0, top_idx)

    # Get top 3 real label classified to predicted class.
    labels_category = prob_c_assigned.map(lambda x: (x[2], mapToLabelVec(x[0]))).reduceByKey(np.add)
    label_map = labels_category.collect()
    top_threes = {}
    for l in label_map:
        percent = l[1] / np.sum(l[1])
        top_idxs = np.argsort(-percent)[:3]
        pairs = zip(labels[top_idxs].tolist(), percent[top_idxs].tolist())
        top_threes[l[0]] = tuple(pairs)

    print("Tope 50 words for each category")
    print(top_50_words)


    print("Top 3 real category for each predicted category")
    for k, v in top_threes.items():
        print("Category ", k)
        print(v)

    print("Training time: ", time.time() - train_time_start)