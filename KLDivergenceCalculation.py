import math
import itertools
#Calculates Kullback-Leibler divergency
def calculateKLDivergency(bn):
    KLdivergence = 0.0
    for t in combinations(bn.getNodes()):
        d = dict()
        for i,w in enumerate(bn.getNodeKeys()):
            d[w] = t[i]
        try:
            KLdivergence += bn.original_full_joint(d) * (math.log(bn.original_full_joint(d)) - math.log(bn.full_joint(d)))
        except ValueError:
            print("Math domain error")
    bn.setDivergenceKL(KLdivergence)
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