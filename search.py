import numpy as np
from datetime import datetime
import random
import cProfile
import re
import multiprocessing as mp

chance = random.seed(datetime.now())
# Using when there is different line length
with open('connect.dat') as f:
    data = []
    for line in f:  # read rest of lines
        data.append([int(x) for x in line.split()])
# Format data of a file to list ( uniform data )
#data = np.genfromtxt('mushroom.dat', delimiter=" ",dtype=None)
#data = data.tolist()

def format_data():
    global data
    # Initializing weights for FrequencyBased and AreaBased Algorithm
    wFrequencyBased = []
    wAreaBased = []
    for i in data:
        # Calculating weights
        wFrequencyBased.append(pow(2,len(i)))
        wAreaBased.append(len(i)*pow(2,len(i)-1))

    return wFrequencyBased,wAreaBased,data


def getTransactionData(w,data,n):

    # Given a number of iteration, choose randomly n "transaction"
    D = random.choices(data, w, k=n)
    print("TRANSACTION PRISE :",D)
    return D


def frequencyBasedSampling(wFrequencyBased,data,n):
    # List of Transaction (n equals the number of iterations)
    D = getTransactionData(wFrequencyBased, data,n)
    # Motifs chosen at first
    motifs = []
    # Motifs chosen without duplicate
    allMotifs = []

    for i in D:
        isIn = False
        for j in i:
            # Choosing randomly a int number between 0 and 1
            global chance
            chance = random.randint(0, 1)
            # Add it to the motifs if chance equals 1
            if chance == 1:
                motifs.append(j)
        print("MOTIFS : ",motifs)
        # Checking if this motifs is already in allMotifs
        for z in allMotifs:
            if z == motifs:
                isIn = True
        if isIn == False:
            allMotifs.append(motifs)
        motifs = []
    print("ALL MOTIFS : ",allMotifs)
    # Call to the frequency function
    frequencyMotifs(allMotifs,data)

def frequencyMotifs(allMotifs,data):
    print("Len allMotifs : ", len(allMotifs))
    # Optimize the speed
    cpus = parallelizeCode()
    # Calling multiple agent depending on how many cpu (2 by default)
    pool = mp.Pool(processes=cpus)
    # Calculating nb frequency in parellel for each motif
    result = pool.map(contains,allMotifs)
    print("NBFrequency : ", result)


def areaBasedSampling(wAreaBased,data,n):
    # List of Transaction (n equals the number of iterations)
    D = getTransactionData(wAreaBased, data,n)
    # Iterator
    iterate = 0
    # Size of motifs chosen
    size = [1]*n
    # Motifs chosen at first
    motifs = []
    # Motifs chosen without duplicate
    allMotifs = []

    for i in D:
        cpt = 0
        # Choosing randomly a float number between 0 and 1
        chooseMotifs = random.uniform(0, 1)
        # Choosing size of the motifs (> importance to bigger motifs)
        for j in range(len(i)):
            cpt = cpt + j + 1
            if chooseMotifs <= cpt/len(i):
                size[iterate] = len(str(i[j]))
                break
        iterate = iterate +1

    iterate = 0

    for i in D:
        isIn = False
        for j in i:
            # If it matches the size of the motifs, we create our list
            if len(str(j)) == size[iterate]:
                motifs.append(j)
        iterate = iterate + 1
        # Checking if this motifs is already in allMotifs
        for z in allMotifs:
            if z == motifs:
                isIn = True
        # If not we are adding it
        if isIn == False:
            allMotifs.append(motifs)
        print("MOTIFS : ", motifs)
        motifs = []
    print("ALL MOTIFS :",allMotifs)
    # Call to the frequency function
    frequencyMotifs(allMotifs, data)

#######################################################################

def wrapperC(small_big):
    return contains(*small_big)
def contains(small):
    count = 0
    global data
    for i in data:
        if all(elem in i for elem in small):
            count = count + 1
    return count


########################################################################

def parallelizeCode ():
    try:
        cpus = mp.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default
    return cpus

if __name__ == '__main__':

    #######################
    # Show the time passed in each function
    pr = cProfile.Profile()
    pr.enable()
    #######################

    # Number of iterations
    n = 1000
    # Format data
    wFrequencyBased,wAreaBased,data = format_data()
    # FrenquencyBased Algorithm
    frequencyBasedSampling(wFrequencyBased,data,n)
    # AreaBased Algorithm
    #areaBasedSampling(wAreaBased,data,n)

    ########################
    # Close and print result
    pr.disable()
    pr.print_stats()
    ########################