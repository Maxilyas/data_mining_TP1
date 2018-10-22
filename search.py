import numpy as np
from datetime import datetime
import random
import cProfile
import multiprocessing as mp
import matplotlib.pyplot as plt # pour l'affichage
from sklearn.linear_model import LinearRegression
import collections as c
import os
from decimal import *

chance = random.seed(datetime.now())

##################################################################
#########################DATA#####################################

def format_data():
    # Initializing weights for FrequencyBased and AreaBased Algorithm
    wFrequencyBased = []
    wAreaBased = []
    lenghtMax = []
    data = []
    # Use it when there is different line length in the file
    with open('mushroom.dat') as f:
        temp = []
        for line in f:  # read rest of
            transac = [int(x) for x in line.split()]
            lenghtMax.append(len(transac))
            temp.append(transac)

    # Solution Question 8
    avg = sum(lenghtMax) / len(temp)
    var = np.var(lenghtMax)
    et = np.sqrt(var)
    print("Average Length : ", avg)
    print("Variance : ", var)
    print("EcartType : ", et)
    print("Maximum length ",max(lenghtMax))

    # Check if the biggest transaction is bigger than avg+2*et
    if max(lenghtMax) <= avg + 2*et:
        # If not we are not touching the algorithm
        data = temp
        for i in data:
            # Calculating weights
            wFrequencyBased.append(pow(2, len(i)))
            wAreaBased.append((len(i) * pow(2, len(i) - 1)))
    else:
        # We are putting equivalent weights because otherwise the same transaction will be picked.
        for i in temp:
            if len(i) < avg + 2*et:
                data.append(i)
                wFrequencyBased.append()
                wAreaBased.append(1)

    return wFrequencyBased, wAreaBased, data

# Get Transaction from both algorithm
def getTransactionData(w,data,n):
    # Given a number of iteration, choose randomly n "transaction"
    D = random.choices(data, w, k=n)
    #print("Ensemble des Transactions :", D)
    return D

##################################################################
##########################ALGORITHM###############################

# First Algorithm
def frequencyBasedSampling(D):
    print("Frequency Based Sampling Algorithm engaged...")
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
    #print("Motifs Retenu : ", allMotifs)
    print("DONE !")
    return allMotifs

# Second Algorithm
def areaBasedSampling(D):
    print("Area Based Sampling algorithm engaged...")
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
    #print
    print("DONE !")
    return allMotifs

def isInAllMotifs(allMotifs,motifs):
    for z in allMotifs:
        if z == motifs:
            return True
    return False

##################################################################
#############################EVAL##DIVERSITY######################

# Check the equality of each motifs and return val between 0 and 1 (0=high diversity,1=bad diversity)
def evalDiversity(algo):
    print("Evaluation de la diversité...")
    count = 0
    global D
    # Compare sets of motifs
    if algo == 1:
        D = getTransactionData(wFrequencyBased, data, n)
        allMotifs = frequencyBasedSampling(D)
    if algo == 2:
        D = getTransactionData(wAreaBased, data, n)
        allMotifs = areaBasedSampling(D)
    for i in allMotifs:
        for j in allMotifs:
            if set(j) == set(i):
                count = count + 1
    count = count - len(allMotifs)
    print("Diversité : ", count/n)
    print("Evaluation de la diversité terminé !")


##################################################################
###############MULTI_PROCESSING####COMPUTE FREQ###################

# Get frequency motifs in D from both algorithm without multiprocessing.
def frequencyMotifsWithoutMPInTransac(allMotifs, D):
    result = []
    print("Calcul de la fréquence des motifs en cours...")
    for i in allMotifs:
        result.append(checkMotifs(i,D))
    resultFreq = [(x / len(D))*100 for x in result]
    print("Fréquence de chaque motifs (%) : ", resultFreq)
    print("DONE !")
    return resultFreq

# Get frequency motifs in Data from both algorithm without multiprocessing.
def frequencyMotifsWithoutMPInDB(allMotifs, data):
    result = []
    print("Calcul de la fréquence des motifs en cours...")
    for i in allMotifs:
        result.append(checkMotifs(i,data))
    resultFreq = [(x / len(data))*100 for x in result]
    print("Fréquence de chaque motifs (%) : ", resultFreq)
    print("DONE !")
    return resultFreq

# Get Frequency Motifs from both algorithm
def frequencyMotifs(allMotifs):
    print("Calcul de la fréquence des motifs dans transaction...")
    #print("Nombre de motifs : ", len(allMotifs))
    # Optimize the speed
    cpus = parallelizeCode()
    # Calling multiple agent depending on how many cpu (2 by default)
    pool = mp.Pool(cpus, init_pool, [D])
    # Calculating nb frequency in parellel for each motif
    result = pool.map(contains, allMotifs)

    resultFreq = [(x / len(D))*100 for x in result]
    print("Fréquence de chaque motifs (%) : ", resultFreq)
    print("DONE !")
    return resultFreq

# Get Frequency Motifs in all DB
def frequencyMotifsInAllDB(allMotifs):
    print("Calcul de la fréquence des motifs dans la BD...")
    #print("Nombre de motifs : ", len(allMotifs))
    # Optimize the speed
    cpus = parallelizeCode()
    # Calling multiple agent depending on how many cpu (2 by default)
    pool = mp.Pool(cpus, init_pool_data, [data])
    # Calculating nb frequency in parellel for each motif
    result = pool.map(checkAllDB, allMotifs)
    resultFreq = [(x / len(data))*100 for x in result]
    print("Fréquence de chaque motifs (%) : ", resultFreq)
    print("DONE !")
    return resultFreq

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

def checkMotifs(small,sets):
    count = 0
    for i in sets:
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

def showGraphFrequency(x,y,saveGraph):

    # Give good arguments to the graph
    GraphFreq = Graph()
    GraphFreq.x_plot = x
    GraphFreq.y_plot = y
    GraphFreq.X = np.array(x).reshape(-1, 1)

    # Create a graph with relation between Freq Data and Freq sample
    GraphFreq.showGraphScatterAndRL(saveGraph)

def showGraphDistrib(x, y,name):
    # Give good arguments to the graph
    GraphFreq = Graph()
    GraphFreq.x_plot = x
    GraphFreq.y_plot = y

    # Create a graph with relation between Freq Data and Freq sample
    GraphFreq.showGraphScatter(name)


class Graph:
    def __init__(self):
        self.x_plot = []
        self.y_plot = []
        self.X = np.array([])

    def showGraphScatterAndRL(self, saveGraph):
        # Fit line using all data
        lr = LinearRegression()
        lr.fit(self.X, self.y_plot)
        # Predict data of estimated models
        line_X = np.arange(self.X.min(), self.X.max())[:, np.newaxis]
        line_y = lr.predict(line_X)
        lw = 2

        plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
        plt.xlabel("FreqSample (%)")
        plt.ylabel("FreqData (%)")
        plt.scatter(self.x_plot, self.y_plot)
        if saveGraph:
            plt.savefig("Question7Graph.png")
        plt.show()

    def showGraphScatter(self, name):
        plt.xlabel("Length Motifs")
        plt.ylabel("Freq Motifs (%)")
        plt.scatter(self.x_plot, self.y_plot)
        plt.savefig("fig_"+name+".png")
        plt.show()



##################################################################
##########################DISTRIBUTION############################

def distribFile(algo):

    print("SAVING DISTRIB GRAPH IN PROGRESS ...")
    inputdir = "/home/antoine/Documents/data_mining/data_mining_TP1"
    filelist = os.listdir(inputdir)
    global D
    # Use it when there is different line length in the file
    for input in filelist:
        if input.endswith(".dat"):
            with open(input) as f:
                name = os.path.splitext(input)[0]
                data = []
                # Initializing weights for FrequencyBased and AreaBased Algorithm
                wFrequencyBased = []
                wAreaBased = []
                for line in f:  # read rest of lines
                    data.append([int(x) for x in line.split()])
                # Format data of a file to list ( uniform data )
                # data = np.genfromtxt('mushroom.dat', delimiter=" ",dtype=None)
                # data = data.tolist()
            if algo == 1:
                for i in data:
                    # Calculating weights ( divide by arbitrary number in case len(i) is too big
                    wFrequencyBased.append(pow(2, len(i)))
                # List of Transaction
                D = getTransactionData(wFrequencyBased, data, n)
                # FrenquencyBased Algorithm
                motifs = frequencyBasedSampling(D)
                # length of motifs
                x = [len(l) for l in motifs]
                if multiProcessing:
                    # Freq of each motifs in sample
                    freqSample = frequencyMotifs(motifs)
                else:
                    freqSample = frequencyMotifsWithoutMPInTransac(motifs,D)
                showGraphDistrib(x, freqSample, name)

            if algo == 2:
                for i in data:
                    wAreaBased.append((len(i)*pow(2, len(i)-1)))
                # List of Transaction
                D = getTransactionData(wAreaBased, data, n)
                # areaBased Algorithm
                motifs = areaBasedSampling(D)
                # length of motifs
                x = [len(l) for l in motifs]
                if multiProcessing:
                    # Freq of each motifs in sample
                    freqSample = frequencyMotifs(motifs)
                else:
                    freqSample = frequencyMotifsWithoutMPInTransac(motifs,D)
                showGraphDistrib(x, freqSample, name)

    print("DONE !")



##################################################################
###########################MAIN###################################

if __name__ == '__main__':
    checkTimeInFunction = False
    saveGraph = False
    algo = 1
    multiProcessing = False
    #######################
    # Show the time passed in each function
    if checkTimeInFunction:
        pr = cProfile.Profile()
        pr.enable()
    #######################
    # Number of iterations
    n = 1000
    D = []
    # UNCOMMENT THIS ONLY IF YOU WANT TO CREATE GRAPH TO SEE DISTRIB
    # /!\ You have to change the file directory path in the function
    #distribFile(algo)
    # Format data
    wFrequencyBased, wAreaBased, data = format_data()


    if algo == 1:
        # Randomize selectors ensure high diversity !
        evalDiversity(algo)
        # List of Transaction (n equals the number of iterations)
        D = getTransactionData(wFrequencyBased, data, n)
        # FrenquencyBased Algorithm
        motifs = frequencyBasedSampling(D)
        if multiProcessing:
            # Freq of each motifs in sample
            freqSample = frequencyMotifs(motifs)
            # Freq of each motifs in DB
            freqDB = frequencyMotifsInAllDB(motifs)
        else:
            # Freq of each motifs in sample
            freqSample = frequencyMotifsWithoutMPInTransac(motifs,D)
            # Freq of each motifs in DB
            freqDB = frequencyMotifsWithoutMPInDB(motifs,data)
        # Show the graph between freqSample and freqDB
        showGraphFrequency(freqSample, freqDB, saveGraph)

    if algo == 2:
        # Randomize selectors ensure high diversity !
        evalDiversity(algo)
        # List of Transaction (n equals the number of iterations)
        D = getTransactionData(wAreaBased, data, n)
        # AreaBased Algorithm
        motifs = areaBasedSampling(D)
        if multiProcessing:
            # Freq of each motifs in sample
            freqSample = frequencyMotifs(motifs)
            # Freq of each motifs in DB
            freqDB = frequencyMotifsInAllDB(motifs)
        else:
            # Freq of each motifs in sample
            freqSample = frequencyMotifsWithoutMPInTransac(motifs,D)
            # Freq of each motifs in DB
            freqDB = frequencyMotifsWithoutMPInDB(motifs,data)
        # Show the graph between freqSample and freqDB
        showGraphFrequency(freqSample, freqDB, saveGraph)

    ########################
    # Close and print result
    if checkTimeInFunction:
        pr.disable()
        pr.print_stats()
    ########################
