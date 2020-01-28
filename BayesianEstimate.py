import csv
import numpy as np
import copy
from BayesianNetwork import BayesianNet
from KLDivergenceCalculation import combinations

def bayes_estimator(csvFile,bn):
    """
    Bayesian Estimation method of parameter learning.
    This method proceeds by either 1) assuming a uniform prior
    over the parameters based on the Dirichlet distribution
    with an equivalent sample size = csvFile size if the prior distribution is equal to 0,
    or 2) assuming a prior as specified by the user within the *prior_dict* argument.
    The prior distribution is then updated from observations in the data based on the
    Multinomial distribution - for which the Dirichlet
    is a "conjugate prior."
    Arguments
    ---------
    *bn* : a BayesianNetwork object
    *csvFile* : a csv file, with RV name in the first line. The order of the variables must follow the first line.
    """
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        data = []
        for row in reader:
            data.append(row)
    equiv_sample = len(data)

    obs_dict = dict([(rv, []) for rv in bn.getNodeKeys()])
    # set empty conditional probability table for each RV
    for rv in bn.getNodes():
        # get number of values in the CPT = product of scope vars' cardinalities
        p_idx = int(np.prod([bn.nodes[p].card() for p in rv.parents]) * rv.card())
        if rv.alphas != list(np.zeros(shape=(p_idx,1))):
            rv.cpt = copy.deepcopy(rv.alphas)
        else:
        #Sets uniform distribution for each cell of cpt
            rv.cpt = [equiv_sample / p_idx] * p_idx

    # loop through each row of data
    for row in data:
        # store the observation of each variable in the row
        obs_dict = dict([(rv, row[rv]) for rv in bn.getNodeKeys()])
        # loop through each RV and increment its observed value
        for rv in bn.getNodes():
            rv_dict = {n: obs_dict[n] for n in obs_dict if n in rv.scope()}
            offset = bn.cpt_indices(target=rv, val_dict=rv_dict)
            rv.cpt[offset] += 1
            rv.alphas[offset] +=1

    for rv in bn.getNodes():
        cpt = rv.cpt
        alphas = rv.alphas
        for i in range(0, len(rv.cpt), rv.card()):
            temp_sum = float(np.sum(cpt[i:(i + rv.card())]))
            #alphas_sum = float(np.sum(alphas[i:(i+rv.card())]))
            for j in range(rv.card()):
                #
                cpt[i + j] /= (temp_sum)
                #Rounds the calculated posterior probability to 2nd decimal
                cpt[i + j] = round(cpt[i + j], 5)
                #if(bn.nodes[rv].parents):
                    #print('P(',bn.nodes[rv].name,'=')
        #print('CPT for ',rv,': ',cpt)
def arrayBE(arr):
    for i in range(len(arr)):
        bn = BayesianNet()
        bn.readNetwork("resources/cancer3.bn")
        arr[i][0] = bn
        bayes_estimator(arr[i][1], arr[i][0])

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