import numpy as np
from datetime import datetime
import random
import pickle
chance = random.seed(datetime.now())

def format_data():
    data = np.genfromtxt('mushroom.dat', delimiter=" ",dtype=None)
    data = data.tolist()
    wFrequencyBased = []
    wAreaBased = []
    for i in data:
        wFrequencyBased.append(pow(2,len(i)))
        wAreaBased.append(len(i)*pow(2,len(i)-1))

    return wFrequencyBased,wAreaBased,data


def getTransactionData(w,data,n):

    if n == 1:
        D = random.choices(data,w)
    else:
        D = random.choices(data, w, k=n)
    print("TRANSACTION PRISE :",D)
    return D


def frequencyBasedSampling(wFrequencyBased,data,n):
    D = getTransactionData(wFrequencyBased, data,n)
    motifs = []
    allMotifs = []
    for i in D:
        isIn = False
        for j in i:
            chance = random.randint(0, 1)
            if chance == 1:
                motifs.append(j)
        print("MOTIFS : ",motifs)
        for z in allMotifs:
            if z == motifs:
                isIn = True
        if isIn == False:
            allMotifs.append(motifs)
        motifs = []
    print("ALL MOTIFS : ",allMotifs)
    frequencyMotifs(allMotifs,data)

def frequencyMotifs(allMotifs,data):
    nbFrequent = [i for i in range(len(allMotifs))]
    print("Len allMotifs : ", len(allMotifs))

    for i in data:
        count = 0
        for j in allMotifs:
            if contains(j,i) == True:
                nbFrequent[count] = nbFrequent[count] + 1
            count = count +1
    print("NBFrequency : ", nbFrequent)
    return nbFrequent


def areaBasedSampling(wAreaBased,data,n):
    D = getTransactionData(wAreaBased, data,n)
    iterate = 0
    taille = [1]*n
    motifs = []
    allMotifs = []

    for i in D:
        cpt = 0
        chooseMotifs = random.uniform(0, 1)
        for j in range(len(i)):
            cpt = cpt + j + 1
            if chooseMotifs <= cpt/len(i):
                taille[iterate] = len(str(i[j]))
                break
        iterate = iterate +1

    iterate = 0

    for i in D:
        isIn = False
        for j in i:
            if len(str(j)) == taille[iterate]:
                motifs.append(j)
        iterate = iterate + 1
        for z in allMotifs:
            if z == motifs:
                isIn = True
        if isIn == False:
            allMotifs.append(motifs)
        print("MOTIFS : ", motifs)
        motifs = []
    print("ALL MOTIFS :",allMotifs)

    frequencyMotifs(allMotifs, data)
#######################################################################

def contains(small, big):
    result = all(elem in big for elem in small)
    if result:
        return True
    return False

########################################################################

if __name__ == '__main__':
    n = 1000
    wFrequencyBased,wAreaBased,data = format_data()
    #frequencyBasedSampling(wFrequencyBased,data,n)
    areaBasedSampling(wAreaBased,data,n)