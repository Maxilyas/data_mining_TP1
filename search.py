import numpy as np
from datetime import datetime
import random
import cProfile
import multiprocessing as mp
import matplotlib.pyplot as plt # pour l'affichage
import sys
from itertools import combinations

chance = random.seed(datetime.now())
def format_data():
    # Initializing weights for FrequencyBased and AreaBased Algorithm
    wFrequencyBased = []
    wAreaBased = []
    # Use it when there is different line length in the file
    with open('connect.dat') as f:
        data = []
        for line in f:  # read rest of lines
            data.append([int(x) for x in line.split()])
    # Format data of a file to list ( uniform data )
    # data = np.genfromtxt('mushroom.dat', delimiter=" ",dtype=None)
    # data = data.tolist()
    for i in data:
        # Calculating weights ( divide by arbitrary number in case len(i) is too big
        wFrequencyBased.append(pow(2, len(i))/pow(2,len(i)-2))
        wAreaBased.append((len(i)*pow(2, len(i)-1))/pow(2,len(i)-2))

    return wFrequencyBased, wAreaBased, data

# First Algorithm
def frequencyBasedSampling(D):
    # Motifs chosen at first
    motifs = []
    # Motifs chosen without duplicate
    allMotifs = []
    for i in D:
        for j in i:
            # Choosing randomly a int number between 0 and 1
            global chance
            chance = random.randint(0, 1)
            # Add it to the motifs if chance equals 1
            if chance == 1:
                motifs.append(j)
        print("MOTIFS : ",motifs)
        # Checking if this motifs is already in allMotifs
        # If not we are adding it
        if not isInAllMotifs(allMotifs, motifs):
            allMotifs.append(motifs)
        motifs = []
    print("ALL MOTIFS : ",allMotifs)
    # Call to the frequency function
    frequencyMotifs(allMotifs)

# Second Algorithm
def areaBasedSampling(D):
    # Iterator
    iterate = 0
    # Size of motifs chosen
    size = [1]*len(D)
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
        iterate = iterate + 1

    iterate = 0

    for i in D:
        for j in i:
            # If it matches the size of the motifs, we create our list
            if len(str(j)) == size[iterate]:
                motifs.append(j)
        iterate = iterate + 1
        # Checking if this motifs is already in allMotifs
        # If not we are adding it
        if not isInAllMotifs(allMotifs, motifs):
            allMotifs.append(motifs)
        print("MOTIFS : ", motifs)
        motifs = []
    print("ALL MOTIFS :", allMotifs)
    # Call to the frequency function
    frequencyMotifs(allMotifs)

# Get Transaction from both algorithm
def getTransactionData(w,data,n):
    # Max size you can have
    #print(sys.maxsize)
    # Given a number of iteration, choose randomly n "transaction"
    D = random.choices(data, w, k=n)
    print("TRANSACTION PRISE :", D)
    return D

# Get Frequency Motifs from both algorithm
def frequencyMotifs(allMotifs):
    print("Len allMotifs : ", len(allMotifs))
    # Optimize the speed
    cpus = parallelizeCode()
    # Calling multiple agent depending on how many cpu (2 by default)
    pool = mp.Pool(cpus, init_pool, [D])
    # Calculating nb frequency in parellel for each motif
    result = pool.map(contains, allMotifs)
    print("NBFrequency : ", result)

    # List of len of allMotifs
    x = [len(l) for l in allMotifs]
    # Create a graph with relation between len of motifs and the number of frequency
    showGraph(x,result)

def init_pool(D):
    global Df
    Df = D

########################################################################

def isInAllMotifs(allMotifs,motifs):
    for z in allMotifs:
        if z == motifs:
            return True
    return False

def contains(small):
    count = 0
    for i in Df:
        #if len(i) < 80:
        if all(elem in i for elem in small):
            count = count + 1
    return count


########################################################################
def showGraph(x,y):
    GraphFreq = Graph()
    GraphFreq.x_plot = x
    GraphFreq.y_plot = y
    GraphFreq.showGraphScatter()
    #GraphFreq.showGraphPlot()

class Graph:
    def __init__(self):
        self.x_plot = []
        self.y_plot = []

    def showGraphScatter(self):
        plt.scatter(self.x_plot, self.y_plot)
        plt.show()

    def showGraphPlot(self):
        plt.plot(self.x_plot, self.y_plot)
        plt.show()


def parallelizeCode ():
    try:
        cpus = mp.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default
    return cpus


if __name__ == '__main__':
    checkTime = False
    algo = 1
    #######################
    # Show the time passed in each function
    if checkTime:
        pr = cProfile.Profile()
        pr.enable()
    #######################
    # Number of iterations
    n = 1000
    # Format data
    wFrequencyBased,wAreaBased,data = format_data()

    if algo == 1:
        # List of Transaction (n equals the number of iterations)
        D = getTransactionData(wFrequencyBased,data,n)
        # FrenquencyBased Algorithm
        frequencyBasedSampling(D)
    if algo == 2:
        D = getTransactionData(wAreaBased,data,n)
        # AreaBased Algorithm
        areaBasedSampling(D)

    ########################
    # Close and print result
    if checkTime:
        pr.disable()
        pr.print_stats()
    ########################