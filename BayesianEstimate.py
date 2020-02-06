import csv
import numpy as np
import copy
import scipy.stats as stats
from matplotlib import pyplot as plt

from BayesianNetwork import BayesianNet

def bayesEstimate(csvFile,bn):
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        data = []
        for row in reader:
            data.append(row)
    sampleSize = len(data)

    obs_dict = dict([(rv, []) for rv in bn.getNodeKeys()])
    # set empty conditional probability table for each RV
    for rv in bn.getNodes():
        # get number of values in the CPT
        p_idx = int(np.prod([bn.nodes[p].card() for p in rv.parents]) * rv.card())
        if rv.alphas != list(np.zeros(shape=(p_idx,1))):
            rv.cpt = copy.deepcopy(rv.alphas)
        else:
        #Sets uniform distribution for each cell of cpt
            rv.cpt = [sampleSize / p_idx] * p_idx

    # loop through each row of data
    for row in data:
        # store the observation of each variable in the row
        obs_dict = dict([(rv, row[rv]) for rv in bn.getNodeKeys()])
        # loop through each RV and increment its observed value
        for rv in bn.getNodes():
            rv_dict = {n: obs_dict[n] for n in obs_dict if n in rv.scope()}
            offset = bn.cpt_indices(target=rv, val_dict=rv_dict)
            #increments the virtual count
            rv.alphas[offset] +=1
        for rv in bn.getNodes():
            for i in range(0, len(rv.cpt), rv.card()):
                temp_sum = float(np.sum(rv.cpt[i:(i + rv.card())]))
                alphas_sum = float(np.sum(rv.alphas[i:(i + rv.card())]))
                for j in range(rv.card()):
                    rv.cpt[i + j] = (rv.cpt[i + j] + rv.alphas[i+j]+1) / (temp_sum + alphas_sum + rv.card())
                    # Rounds the calculated posterior probability to 2nd decimal
                    rv.cpt[i + j] = round(rv.cpt[i + j], 5)
    plotGamma(bn.getNode('A').alphas[0],bn.getNode('A').alphas[1])
def arrayBE(arr):
    for i in range(len(arr)):
        bn = BayesianNet()
        bn.readNetwork("resources/cancer3.bn")
        arr[i][0] = bn
        bayesEstimate(arr[i][1], arr[i][0])
def plotGamma(alpha,beta):
    x = np.linspace (0,1, 200)
    y1 = stats.gamma.pdf(x, a=alpha, scale=1/beta) #a is alpha, 1/scale is beta
    plt.plot(x, y1, "y-", label=(r'$\alpha=29, \beta=3$'))
    plt.show()

if __name__ == "__main__":
    bn = BayesianNet()
    bn.readNetwork("resources/cancer3.bn")
    csvFile ='resources/datasets/10Cases.csv'
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        tuples = []
        for row in reader:
            tuples.append(tuple(row[v] for v in bn.getNodeKeys()))
        print(tuples)