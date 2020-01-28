import math
import itertools
#import BayesianNetwork
#Calculates Kullback-Leibler divergency
def calculateKLDivergency(bayesianNet):
    KLdivergence = 0.0
    for t in combinations(bayesianNet.getNodes()):
        d = dict()
        for i,w in enumerate(bayesianNet.getNodeKeys()):
            d[w] = t[i]
        KLdivergence += bayesianNet.original_full_joint(d) * (math.log(bayesianNet.original_full_joint(d)) - math.log(bayesianNet.full_joint(d)))
    bayesianNet.setDivergenceKL(KLdivergence)
    return KLdivergence

#estimates mean for al KL calculated in the netwok of the array
def meanKL(arr):
    mean = 0.0
    for i in range(len(arr)):
        calculateKLDivergency(arr[i][0])
        mean += arr[i][0].getDivergenceKL()
    return mean/len(arr)

def combinations(nodes):
    comb_list = []
    for v in nodes:
        comb_list.append(v.domain)
    return list(itertools.product(*comb_list))