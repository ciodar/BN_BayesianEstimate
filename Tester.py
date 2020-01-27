from BayesianNetwork import BayesianNet
import numpy as np
#from MaximumLikelihood import MLE, arrayMLE,static_MLE
from KLDivergenceCalculation import calculateKLDivergency, meanKL,combinations
from BayesianEstimate import bayes_estimator,arrayBE
bn = BayesianNet()
bn.readNetwork("resources/cancer3uniform.bn")
#comb = combinations(bn)
#x = bn.cpt_indices(bn.nodes["B"],dict({'A': 'Present', 'B':'Not increased','C':'Present','D':'Absent','E':'Present'}))

#Calculate KL divergency for more datasets of lenght 10 learning parameter from ech trhough Maximum Likelihood
# array10 = np.array([BayesianNet, 'resources/datasets/10Cases.csv'], [BayesianNet, 'resources/datasets/10_1Cases.csv'], [BayesianNet, 'resources/datasets/10_2Cases.csv'],
#                         [BayesianNet, 'resources/datasets/10_3Cases.csv'], [BayesianNet, 'resources/datasets/10_4Cases.csv'], [BayesianNet, 'resources/datasets/10_5Cases.csv'],
#                         [BayesianNet, 'resources/datasets/10_6Cases.csv'], [BayesianNet, 'resources/datasets/10_7Cases.csv'], [BayesianNet, 'resources/datasets/10_8Cases.csv'],
#                         [BayesianNet, 'resources/datasets/10_9Cases.csv'], [BayesianNet, 'resources/datasets/10_10Cases.csv'])
#
# #bayes_estimator(bn, 'resources/datasets/10000Cases.csv')
# #bn.plot()
# arrayBE(array10)
# # arrayMLE(array10)
# # #arrayBE(array10)
# #
# # #Calculate KL divergency for more datasets of lenght 50 learning parameter from ech trhough Maximum Likelihood
# array50 = np.array([[BayesianNet, '50Cases.csv'], [BayesianNet, '50_1Cases.csv'], [BayesianNet, '50_2Cases.csv'],
#                         [BayesianNet, '50_3Cases.csv'], [BayesianNet, '50_4Cases.csv'], [BayesianNet, '50_5Cases.csv'],
#                         [BayesianNet, '50_6Cases.csv'], [BayesianNet, '50_7Cases.csv'], [BayesianNet, '50_8Cases.csv'],
#                         [BayesianNet, '50_9Cases.csv'], [BayesianNet, '50_10Cases.csv']])
#
# #arrayMLE(array50)
# arrayBE(array50)
#
# #Calculate KL divergency for more datasets of lenght 100 learning parameter from ech trhough Maximum Likelihood
# array100 = np.array([[BayesianNet, 'resources/datasets/100Cases.csv'], [BayesianNet, 'resources/datasets/100_1Cases.csv'], [BayesianNet, 'resources/datasets/100_2Cases.csv'],
#                         [BayesianNet, 'resources/datasets/100_3Cases.csv'], [BayesianNet, 'resources/datasets/100_4Cases.csv'], [BayesianNet, 'resources/datasets/100_5Cases.csv'],
#                         [BayesianNet, 'resources/datasets/100_6Cases.csv'], [BayesianNet, 'resources/datasets/100_7Cases.csv'], [BayesianNet, 'resources/datasets/100_8Cases.csv'],
#                         [BayesianNet, 'resources/datasets/100_9Cases.csv'], [BayesianNet, 'resources/datasets/100_10Cases.csv']])
#
# #arrayMLE(array100)
# arrayBE(array100)
bNet10 = BayesianNet()
bNet10.readNetwork("resources/cancer3uniform.bn")
csvfile10 = 'resources/datasets/10Cases.csv'
bayes_estimator(csvfile10, bNet10)
print('KL divergence for n = 10', calculateKLDivergency(bNet10))
#
#Calculate MLE for all other datasets
# bNet250 = BayesianNet()
# bNet250.readNetwork("resources/cancer3uniform.bn")
csvfile250 = 'resources/datasets/250Cases.csv'
#bayes_estimator(csvfile250, bNet250)
bayes_estimator(csvfile250, bNet10)
print('KL divergence for n = 250', calculateKLDivergency(bNet10))
#
# bNet500 = BayesianNet()
# bNet500.readNetwork("resources/cancer3uniform.bn")
csvfile500 = 'resources/datasets/500Cases.csv'
#bayes_estimator(csvfile500, bNet500)
bayes_estimator(csvfile500, bNet10)
print('KL divergence for n = 500', calculateKLDivergency(bNet10))
# bNet750 = BayesianNet()
# bNet750.readNetwork("resources/cancer3uniform.bn")
csvfile750 = 'resources/datasets/750Cases.csv'
#bayes_estimator(csvfile750, bNet750)
bayes_estimator(csvfile750, bNet10)
print('KL divergence for n = 750', calculateKLDivergency(bNet10))

# bNet1000 = BayesianNet()
# bNet1000.readNetwork("resources/cancer3uniform.bn")
csvfile1000 = 'resources/datasets/1000Cases.csv'
#bayes_estimator(csvfile1000, bNet1000)
bayes_estimator(csvfile1000, bNet10)
print('KL divergence for n = 1000', calculateKLDivergency(bNet10))
# bNet1250 = BayesianNet()
# bNet1250.readNetwork("resources/cancer3uniform.bn")
csvfile1250 = 'resources/datasets/1250Cases.csv'
#bayes_estimator(csvfile1250, bNet1250)
bayes_estimator(csvfile1250, bNet10)
print('KL divergence for n = 1250', calculateKLDivergency(bNet10))
# bNet1500 = BayesianNet()
# bNet1500.readNetwork("resources/cancer3uniform.bn")
csvfile1500 = 'resources/datasets/1500Cases.csv'
#bayes_estimator(csvfile1500, bNet1500)
bayes_estimator(csvfile1500, bNet10)
print('KL divergence for n = 1500', calculateKLDivergency(bNet10))
# bNet1750 = BayesianNet()
# bNet1750.readNetwork("resources/cancer3uniform.bn")
csvfile1750 = 'resources/datasets/1750Cases.csv'
#bayes_estimator(csvfile1750, bNet1750)
bayes_estimator(csvfile1750, bNet10)
print('KL divergence for n = 1750', calculateKLDivergency(bNet10))
#bNet2000 = BayesianNet()
#bNet2000.readNetwork("resources/cancer3uniform.bn")
csvfile2000 = 'resources/datasets/2000Cases.csv'
#bayes_estimator(csvfile2000, bNet2000)
bayes_estimator(csvfile2000, bNet10)
print('KL divergence for n = 2000', calculateKLDivergency(bNet10))
# bNet2250 = BayesianNet()
# bNet2250.readNetwork("resources/cancer3uniform.bn")
csvfile2250 = 'resources/datasets/2250Cases.csv'
# bayes_estimator(csvfile2250, bNet2250)
bayes_estimator(csvfile2250, bNet10)
print('KL divergence for n = 2250', calculateKLDivergency(bNet10))
# bNet2500 = BayesianNet()
# bNet2500.readNetwork("resources/cancer3uniform.bn")
csvfile2500 = 'resources/datasets/2500Cases.csv'
#bayes_estimator(csvfile2500, bNet2500)
bayes_estimator(csvfile2500, bNet10)
print('KL divergence for n = 2500', calculateKLDivergency(bNet10))
# bNet3000 = BayesianNet()
# bNet3000.readNetwork("resources/cancer3uniform.bn")
csvfile3000 = 'resources/datasets/3000Cases.csv'
#bayes_estimator(csvfile3000, bNet3000)
bayes_estimator(csvfile3000, bNet10)
print('KL divergence for n = 3000', calculateKLDivergency(bNet10))
# bNet3500 = BayesianNet()
# bNet3500.readNetwork("resources/cancer3uniform.bn")
# csvfile3500 = '3500Cases.csv'
# bayes_estimator(csvfile3500, bNet3500)

# bNet4000 = BayesianNet()
# bNet4000.readNetwork("resources/cancer3uniform.bn")
#csvfile4000 = 'resources/datasets/4000Cases.csv'
#bayes_estimator(csvfile4000, bNet4000)

#bNet4000 = BayesianNet()
#bNet4000.readNetwork("resources/cancer3uniform.bn")
csvfile4000 = 'resources/datasets/4000Cases.csv'
bayes_estimator(csvfile4000, bNet10)
print('KL divergence for n = 4000', calculateKLDivergency(bNet10))
#bNet4500 = BayesianNet()
#bNet4500.readNetwork("resources/cancer3uniform.bn")
csvfile4500 = 'resources/datasets/4500Cases.csv'
bayes_estimator(csvfile4500, bNet10)
print('KL divergence for n = 4500', calculateKLDivergency(bNet10))
#bNet5000 = BayesianNet()
#bNet5000.readNetwork("resources/cancer3uniform.bn")
csvfile5000 = 'resources/datasets/5000Cases.csv'
#bayes_estimator(csvfile5000, bNet5000)
bayes_estimator(csvfile5000, bNet10)
print('KL divergence for n = 5000', calculateKLDivergency(bNet10))
# bNet6000 = BayesianNet()
# bNet6000.readNetwork("resources/cancer3uniform.bn")
csvfile6000 = 'resources/datasets/6000Cases.csv'
#bayes_estimator(csvfile6000, bNet6000)
bayes_estimator(csvfile6000, bNet10)
print('KL divergence for n = 6000', calculateKLDivergency(bNet10))
# bNet7000 = BayesianNet()
# bNet7000.readNetwork("resources/cancer3uniform.bn")
csvfile7000 = 'resources/datasets/7000Cases.csv'
#bayes_estimator(csvfile7000, bNet7000)
bayes_estimator(csvfile7000, bNet10)
print('KL divergence for n = 7000', calculateKLDivergency(bNet10))
# bNet8000 = BayesianNet()
# bNet8000.readNetwork("resources/cancer3uniform.bn")
csvfile8000 = 'resources/datasets/8000Cases.csv'
#bayes_estimator(csvfile8000, bNet8000)
bayes_estimator(csvfile8000, bNet10)
print('KL divergence for n = 8000', calculateKLDivergency(bNet10))

# bNet9000 = BayesianNet()
# bNet9000.readNetwork("resources/cancer3uniform.bn")
csvfile9000 = 'resources/datasets/9000Cases.csv'
#bayes_estimator(csvfile9000, bNet9000)
bayes_estimator(csvfile9000, bNet10)
print('KL divergence for n = 9000', calculateKLDivergency(bNet10))

# bNet10000 = BayesianNet()
# bNet10000.readNetwork("resources/cancer3uniform.bn")
csvfile10000 = 'resources/datasets/10000Cases.csv'
#bayes_estimator(csvfile10000, bNet10000)
bayes_estimator(csvfile10000, bNet10)
print('KL divergence for n = 10000', calculateKLDivergency(bNet10))
#print All Kullback-Leibler for all learned network
# print('KL divergence for n = 10',meanKL(array10))
# print('KL divergence for n = 50', meanKL(array50))
# print('KL divergence for n = 100',meanKL(array100))

#print('KL divergence for n = 10', calculateKLDivergency(bNet10))
# print('KL divergence for n = 500', calculateKLDivergency(bNet500))
# print('KL divergence for n = 750', calculateKLDivergency(bNet750))
# print('KL divergence for n = 1000', calculateKLDivergency(bNet1000))
# print('KL divergence for n = 1250', calculateKLDivergency(bNet1250))
# print('KL divergence for n = 1500', calculateKLDivergency(bNet1500))
# print('KL divergence for n = 1750', calculateKLDivergency(bNet1750))
# print('KL divergence for n = 2000', calculateKLDivergency(bNet2000))
# print('KL divergence for n = 2250', calculateKLDivergency(bNet2250))
# print('KL divergence for n = 2500', calculateKLDivergency(bNet2500))
# print('KL divergence for n = 3000', calculateKLDivergency(bNet3000))
# # print('KL divergence for n = 3500', calculateKLDivergency(bNet3500))
# print('KL divergence for n = 4000', calculateKLDivergency(bNet4000))
# print('KL divergence for n = 4500', calculateKLDivergency(bNet4500))
# print('KL divergence for n = 5000', calculateKLDivergency(bNet5000))
# print('KL divergence for n = 6000', calculateKLDivergency(bNet6000))
# print('KL divergence for n = 7000', calculateKLDivergency(bNet7000))
# print('KL divergence for n = 8000', calculateKLDivergency(bNet8000))
# print('KL divergence for n = 9000', calculateKLDivergency(bNet9000))
# print('KL divergence for n = 10000', calculateKLDivergency(bNet10000))
#


