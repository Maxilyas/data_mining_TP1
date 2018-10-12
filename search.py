import numpy as np
from datetime import datetime
import random
import cProfile
import multiprocessing as mp
import matplotlib.pyplot as plt # pour l'affichage
from sklearn.linear_model import LinearRegression
import collections as c

chance = random.seed(datetime.now())

##################################################################
#########################DATA#####################################

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

# Get Transaction from both algorithm
def getTransactionData(w,data,n):
    # Given a number of iteration, choose randomly n "transaction"
    D = random.choices(data, w, k=n)
    print("Ensemble des Transactions :", D)
    return D

##################################################################
##########################ALGORITHM###############################

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
        #print("MOTIFS : ",motifs)
        # Checking if this motifs is already in allMotifs
        # If not we are adding it
        if not isInAllMotifs(allMotifs, motifs):
            allMotifs.append(motifs)
        motifs = []
    print("Motifs Retenu : ", allMotifs)
    return allMotifs

# Second Algorithm
def areaBasedSampling(D):
    # Iterator
    iterate = 0
    # Size of motifs chosen
    size = [0]*len(D)
    # Motifs chosen without duplicate
    allMotifs = []

    for i in D:
        cpt = 0
        # Choosing randomly a float number between 0 and 1
        chooseMotifs = random.uniform(0, 1)
        # Choosing size of the motifs (> importance to bigger motifs)
        for j in range(len(i)):
            cpt = cpt + j + 1
            if chooseMotifs <= cpt/len(i)*2:
                size[iterate] = cpt
                break
        # Choosing element depending on the length of the motifs (random in transaction)
        motifs = random.sample(i, size[iterate])
        # Checking if this motifs is already in allMotifs
        # If not we are adding it
        if not isInAllMotifs(allMotifs, motifs):
            allMotifs.append(motifs)
        iterate = iterate + 1
        #print("MOTIFS : ", motifs)
    print("Motifs Retenu :", allMotifs)
    return allMotifs

def isInAllMotifs(allMotifs,motifs):
    for z in allMotifs:
        if z == motifs:
            return True
    return False

##################################################################
#############################EVAL##DIVERSITY######################

def evalDiversity(epoch,algo):
    print("Evaluation de la diversité...")
    count = 0
    nbEpoch = epoch
    # Compare two sets of motifs (epoch times)
    while epoch > 0:
        print("Epoch ", epoch)
        D = getTransactionData(wFrequencyBased, data, n)
        if algo == 1:
            motifs1 = frequencyBasedSampling(D)
        else:
            motifs1 = areaBasedSampling(D)
        D = getTransactionData(wFrequencyBased, data, n)
        if algo == 1:
            motifs2 = frequencyBasedSampling(D)
        else:
            motifs2 = areaBasedSampling(D)
        for i in motifs1:
            for j in motifs2:
                if c.Counter(i) == c.Counter(j):
                    count = count + 1
        epoch = epoch - 1
    print("Nombre de motifs récurrent : ", count)
    print("Diversité : ", count/(nbEpoch*n))
    print("Evaluation de la diversité terminé !")

##################################################################
###############MULTI_PROCESSING####COMPUTE FREQ###################

# Get Frequency Motifs from both algorithm
def frequencyMotifs(allMotifs):
    print("Nombre de motifs : ", len(allMotifs))
    # Optimize the speed
    cpus = parallelizeCode()
    # Calling multiple agent depending on how many cpu (2 by default)
    pool = mp.Pool(cpus, init_pool, [D])
    # Calculating nb frequency in parellel for each motif
    result = pool.map(contains, allMotifs)
    print("Fréquence de chaque motifs : ", result)

    return result

# Get Frequency Motifs in all DB
def frequencyMotifsInAllDB(allMotifs):
    print("Nombre de motifs : ", len(allMotifs))
    # Optimize the speed
    cpus = parallelizeCode()
    # Calling multiple agent depending on how many cpu (2 by default)
    pool = mp.Pool(cpus, init_pool_data, [data])
    # Calculating nb frequency in parellel for each motif
    result = pool.map(checkAllDB, allMotifs)
    print("Fréquence de chaque motifs : ", result)

    return result

def init_pool(D):
    global Df
    Df = D
def init_pool_data(db):
    global data
    data = db

def contains(small):
    count = 0
    for i in Df:
        if all(elem in i for elem in small):
            count = count + 1
    return count

def checkAllDB(small):
    count = 0
    for i in data:
        if all(elem in i for elem in small):
            count = count + 1
    return count

def parallelizeCode ():
    try:
        cpus = mp.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default
    return cpus

##################################################################
###########################GRAPH##################################

def showGraphFrequency(x,y):

    # Give good arguments to the graph
    GraphFreq = Graph()
    GraphFreq.x_plot = x
    GraphFreq.y_plot = y
    GraphFreq.X = np.array(x).reshape(-1,1)

    # Create a graph with relation between Freq Data and Freq sample
    GraphFreq.showGraphScatterAndRansac()


class Graph:
    def __init__(self):
        self.x_plot = []
        self.y_plot = []
        self.X = np.array([])

    def showGraphScatterAndRansac(self):
        # Fit line using all data
        lr = LinearRegression()
        lr.fit(self.X, self.y_plot)
        # Predict data of estimated models
        line_X = np.arange(self.X.min(), self.X.max())[:, np.newaxis]
        line_y = lr.predict(line_X)
        lw = 2

        plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
        plt.xlabel("FreqSample")
        plt.ylabel("FreqData")
        plt.scatter(self.x_plot, self.y_plot)
        plt.show()

    def showGraphScatterOnly(self):
        plt.scatter(self.x_plot, self.y_plot)
        plt.show()


##################################################################
###########################MAIN###################################

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
    epoch = 1
    # Format data
    wFrequencyBased, wAreaBased, data = format_data()

    if algo == 1:
        evalDiversity(epoch, algo)
        # List of Transaction (n equals the number of iterations)
        D = getTransactionData(wFrequencyBased, data, n)
        # FrenquencyBased Algorithm
        motifs = frequencyBasedSampling(D)
        # Freq of each motifs in sample
        freqSample = frequencyMotifs(motifs)
        # Freq of each motifs in DB
        freqDB = frequencyMotifsInAllDB(motifs)
        # Show the graph between freqSample and freqDB
        showGraphFrequency(freqSample, freqDB)

    if algo == 2:
        evalDiversity(epoch, algo)
        # List of Transaction (n equals the number of iterations)
        D = getTransactionData(wAreaBased, data, n)
        # AreaBased Algorithm
        motifs = areaBasedSampling(D)
        # Freq of each motifs in sample
        freqSample = frequencyMotifs(motifs)
        # Freq of each motifs in DB
        freqDB = frequencyMotifsInAllDB(motifs)
        # Show the graph between freqSample and freqDB
        showGraphFrequency(freqSample, freqDB)

    ########################
    # Close and print result
    if checkTime:
        pr.disable()
        pr.print_stats()
    ########################