import numpy as np
from datetime import datetime
import random
import pickle

chance = random.seed(datetime.now())

def format_data():
    # Format data of a file to list ( uniform data )
    data = np.genfromtxt('mushroom.dat', delimiter=" ",dtype=None)
    data = data.tolist()

    # Initializing wait for FrequencyBased and AreaBased Algorithm
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
    nbFrequent = [i for i in range(len(allMotifs))]
    print("Len allMotifs : ", len(allMotifs))

    # Browse through data
    for i in data:
        count = 0
        for j in allMotifs:
            # Simple counter
            if contains(j,i) == True:
                nbFrequent[count] = nbFrequent[count] + 1
            count = count +1
    print("NBFrequency : ", nbFrequent)
    # Returning a list of frequency
    return nbFrequent


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

def contains(small, big):
    # Checking if a motif is in a "transaction"
    result = all(elem in big for elem in small)
    if result:
        return True
    return False

########################################################################

if __name__ == '__main__':
    # Number of iterations
    n = 1
    # Format data
    wFrequencyBased,wAreaBased,data = format_data()
    # FrenquencyBased Algorithm
    frequencyBasedSampling(wFrequencyBased,data,n)
    # AreaBased Algorithm
    areaBasedSampling(wAreaBased,data,n)