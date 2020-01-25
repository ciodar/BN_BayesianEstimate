import csv
import numpy as np
from BayesianNetwork import BayesianNet
from KLDivergenceCalculation import combinations

def bayes_estimator(bn, csvFile):
    """
    Bayesian Estimation method of parameter learning.
    This method proceeds by either 1) assuming a uniform prior
    over the parameters based on the Dirichlet distribution
    with an equivalent sample size = *sample_size*, or
    2) assuming a prior as specified by the user with the
    *prior_dict* argument. The prior distribution is then
    updated from observations in the data based on the
    Multinomial distribution - for which the Dirichlet
    is a "conjugate prior."
    Note that the Bayesian and MLE estimators essentially converge
    to the same set of values as the size of the dataset increases.
    Also note that, unlike the structure learning algorithms, the
    parameter learning functions REQUIRE a passed-in BayesNet object
    because there MUST be some pre-determined structure for which
    we can actually learn the parameters. You can't learn parameters
    without structure - so structure must always be there first!
    Finally, note that this function can be used to calculate only
    ONE conditional probability table in a BayesNet object by
    passing in a subset of random variables with the "nodes"
    argument - this is mostly used for score-based structure learning,
    where a single cpt needs to be quickly recalculate after the
    addition/deletion/reversal of an arc.
    Arguments
    ---------
    *bn* : a BayesNet object
    *data* : a nested numpy array
        Data from which to learn parameters
    *equiv_sample* : an integer
        The "equivalent sample size" (see function summary)
    *prior_dict* : a dictionary, where key = random variable
        and for each key the value is another dictionary where
        key = an instantiation for the random variable and the
        value is its FREQUENCY (an integer value, NOT its relative
        proportion/probability).

    *nodes* : a list of strings
        Which nodes to learn the parameters for - if None,
        all nodes will be used as expected.

    Returns
    -------
    None
    Effects
    -------
    - modifies/sets bn.data to the learned parameters
    Notes
    -----
    """
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        data = []
        for row in reader:
            data.append(row)
    equiv_sample = len(data)

    obs_dict = dict([(rv, []) for rv in bn.getNodes()])
    # set empty conditional probability table for each RV
    for rv in bn.getNodes():
        # get number of values in the CPT = product of scope vars' cardinalities
        p_idx = int(np.prod([bn.nodes[p].card() for p in bn.nodes[rv].parents]) * bn.nodes[rv].card())
        bn.nodes[rv].cpt = [equiv_sample / p_idx] * p_idx

    # loop through each row of data
    for row in data:
        # store the observation of each variable in the row
        obs_dict = dict([(rv, row[rv]) for rv in bn.getNodes()])
        # loop through each RV and increment its observed parent-self value
        for rv in bn.getNodes():
            rv_dict = {n: obs_dict[n] for n in obs_dict if n in bn.nodes[rv].scope()}
            offset = bn.cpt_indices(target=bn.nodes[rv], val_dict=rv_dict)
            bn.nodes[rv].cpt[offset] += 1

    for rv in bn.getNodes():
        cpt = bn.nodes[rv].cpt
        for i in range(0, len(bn.nodes[rv].cpt), bn.nodes[rv].card()):
            temp_sum = float(np.sum(cpt[i:(i + bn.nodes[rv].card())]))
            for j in range(bn.nodes[rv].card()):
                cpt[i + j] /= (temp_sum)
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
            tuples.append(tuple(row[v] for v in bn.getNodes()))
        print(tuples)