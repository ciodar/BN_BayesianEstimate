from __future__ import division
from __future__ import print_function
import numpy as np
import csv
from BayesianNetwork import BayesianNet


def static_MLE(csvFile, network):
    # array to collect data from .csv file
    A = []
    B = []
    C = []
    D = []
    E = []
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter = ';')
        for row in reader:
            A.append(row['A'])
            B.append(row['B'])
            C.append(row['C'])
            D.append(row['D'])
            E.append(row['E'])

    for i in range(len(network.nodes['A'].domain)):
        partialData = 0
        totalData = 0
        #print ('A =', network.nodes['A'].domain[i], '| {} = ',)
        for j in range(len(A)):
            if A[j] == network.nodes['A'].domain[i]:
                partialData += 1
            totalData += 1
        #Calculate likelihood with Laplace Smoothing for k=1, where +2 is due to (lenght of the domain)*k
        likelihoodA = (partialData+1)/(totalData+2)
        #print (likelihoodA)
        if i == 0:
            saveCPT(network.nodes['A'], likelihoodA, i)
            #print(network.nodes['A'].getCPT())
    #print ('')


    for i in range(len(network.nodes['C'].domain)):
        for j in range(len(network.nodes['E'].domain)):
            partialData = 0
            totalData = 0
            #print ('(E =', network.nodes['E'].domain[j],') | (C =', network.nodes['C'].domain[i], ') = ',)
            for k in range(len(E)):
                if C[k] == network.nodes['C'].domain[i]:
                    if E[k] == network.nodes['E'].domain[j]:
                        partialData += 1
                    totalData += 1
            likelihoodEgivenC = (partialData+1)/(totalData+2)
            #print (likelihoodEgivenC)
            if (j==0):
                saveCPT(network.nodes['E'], likelihoodEgivenC, i)
                #saveCPT(network.nodes['E'], 1-likelihoodEgivenC, i+2)
                #print(network.nodes['E'].getCPT())
    #print ('')

    for i in range(len(network.nodes['A'].domain)):
        for j in range(len(network.nodes['C'].domain)):
            partialData = 0
            totalData = 0
            #print ('(C =', network.nodes['C'].domain[j],') | (A =', network.nodes['A'].domain[i], ') = ',)
            for k in range(len(C)):
                if A[k] == network.nodes['A'].domain[i]:
                    if C[k] == network.nodes['C'].domain[j]:
                        partialData += 1
                    totalData += 1
            likelihoodCgivenA = (partialData+1)/(totalData+2)
            #print (likelihoodCgivenA)
            if (j==0):
                saveCPT(network.nodes['C'], likelihoodCgivenA, i)
                #saveCPT(network.nodes['C'], 1- likelihoodCgivenA, i+2)
                #print(network.nodes['C'].getCPT())
    #print ('')

    for i in range(len(network.nodes['A'].domain)):
        for j in range(len(network.nodes['B'].domain)):
            partialData = 0
            totalData = 0
            #print ('(B =', network.nodes['B'].domain[j],') | (A =', network.nodes['A'].domain[i], ') = ',)
            for k in range(len(B)):
                if A[k] == network.nodes['A'].domain[i]:
                    if B[k] == network.nodes['B'].domain[j]:
                        partialData += 1
                    totalData += 1
            likelihoodBgivenA = (partialData+1)/(totalData+2)
            #print (likelihoodBgivenA)
            if (j==0):
                saveCPT(network.nodes['B'], likelihoodBgivenA, i)
                #saveCPT(network.nodes['B'], 1- likelihoodBgivenA, i+2)
                #print(network.nodes['B'].getCPT())
    #print ('')

    for i in range(len(network.nodes['B'].domain)):
        for j in range(len(network.nodes['C'].domain)):
            for k in range(len(network.nodes['D'].domain)):
                partialData = 0
                totalData = 0
                #print('(D =', network.nodes['D'].domain[k], ') | (B =', network.nodes['B'].domain[i], ', C =', network.nodes['C'].domain[j], ') =', )
                for l in range(len(D)):
                    if B[l] == network.nodes['B'].domain[i]:
                        if C[l] == network.nodes['C'].domain[j]:
                            if D[l] == network.nodes['D'].domain[k]:
                                partialData += 1
                            totalData += 1
                likelihoodDgivenBC = (partialData+1)/(totalData+2)
                #print(likelihoodDgivenBC)
                if (k==0 and i==0):
                    saveCPT(network.nodes['D'], likelihoodDgivenBC, j)
                    #saveCPT(network.nodes['D'], 1-likelihoodDgivenBC, j+4)
                    #print(network.nodes['D'].getCPT())
                elif (k==0 and i==1):
                    saveCPT(network.nodes['D'], likelihoodDgivenBC, j+2)
                    #saveCPT(network.nodes['D'],1 - likelihoodDgivenBC, j + 6)
                    #print(network.nodes['D'].getCPT())
    #print('')
#calculate parameter with Maximum Likelihood for the given network
def MLE(csvFile, network):
    # array to collect data from .csv file
    #Reads each line of csv file and puts data in each array
    d = dict()
    f =  open(csvFile)
    reader = csv.DictReader(f, delimiter = ';')
    row_dict = []
    for row in reader:
        row_dict.append(row)

    for v in network.getNodes():
        for i,n in enumerate(network.nodes[v].domain):
            partialData = 0
            totalData = 0
            #print(v,'=', n, '| {} = ', )
            for j in range(len(d[v])):
                if row_dict[v][j] == n:
                    partialData += 1
                totalData += 1
            likelihood = (partialData +1)/(totalData + 2)
            #print(likelihood)
            saveCPT(network.nodes[v],likelihood,i)
        print('')


def saveCPT(node, likelihood, i):
    node.getCPT()[i] = [likelihood]

#Create a BayesianNet() object for every array cell and calculates MLE for each one
def arrayMLE(arr):
    for i in range(len(arr)):
        bn = BayesianNet()
        bn.readNetwork("resources/cancer3.bn")
        arr[i][0] = bn
        with open(arr[i][1]) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            row_dict = []
            for row in reader:
                row_dict.append(row)
        static_MLE(arr[i][1],arr[i][0])


