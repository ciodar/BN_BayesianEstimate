from BayesianNetwork import BayesianNet
import numpy as np
from MaximumLikelihood import MLE, arrayMLE,static_MLE
from KLDivergenceCalculation import calculateKLDivergency, meanKL,combinations
from BayesianEstimate import arrayBE,BE
bn = BayesianNet()
bn.readNetwork("resources/cancer3.bn")
comb = combinations(bn)
x = bn.cpt_indices(bn.nodes["B"],dict({'A': 'Present', 'B':'Not increased','C':'Present','D':'Absent','E':'Present'}))
bn.plot()
#Calculate KL divergency for more datasets of lenght 10 learning parameter from ech trhough Maximum Likelihood
array10 = np.array([[BayesianNet, '10Cases.csv'], [BayesianNet, '10_1Cases.csv'], [BayesianNet, '10_2Cases.csv'],
                        [BayesianNet, '10_3Cases.csv'], [BayesianNet, '10_4Cases.csv'], [BayesianNet, '10_5Cases.csv'],
                        [BayesianNet, '10_6Cases.csv'], [BayesianNet, '10_7Cases.csv'], [BayesianNet, '10_8Cases.csv'],
                        [BayesianNet, '10_9Cases.csv'], [BayesianNet, '10_10Cases.csv']])
#arrayBE(array10)
arrayMLE(array10)
#arrayBE(array10)

#Calculate KL divergency for more datasets of lenght 50 learning parameter from ech trhough Maximum Likelihood
array50 = np.array([[BayesianNet, '50Cases.csv'], [BayesianNet, '50_1Cases.csv'], [BayesianNet, '50_2Cases.csv'],
                        [BayesianNet, '50_3Cases.csv'], [BayesianNet, '50_4Cases.csv'], [BayesianNet, '50_5Cases.csv'],
                        [BayesianNet, '50_6Cases.csv'], [BayesianNet, '50_7Cases.csv'], [BayesianNet, '50_8Cases.csv'],
                        [BayesianNet, '50_9Cases.csv'], [BayesianNet, '50_10Cases.csv']])

arrayMLE(array50)
#arrayBE(array50)

#Calculate KL divergency for more datasets of lenght 100 learning parameter from ech trhough Maximum Likelihood
array100 = np.array([[BayesianNet, '100Cases.csv'], [BayesianNet, '100_1Cases.csv'], [BayesianNet, '100_2Cases.csv'],
                        [BayesianNet, '100_3Cases.csv'], [BayesianNet, '100_4Cases.csv'], [BayesianNet, '100_5Cases.csv'],
                        [BayesianNet, '100_6Cases.csv'], [BayesianNet, '100_7Cases.csv'], [BayesianNet, '100_8Cases.csv'],
                        [BayesianNet, '100_9Cases.csv'], [BayesianNet, '100_10Cases.csv']])

arrayMLE(array100)
#arrayBE(array100)

#Calculate MLE for all other datasets
bNet250 = BayesianNet()
bNet250.readNetwork("resources/cancer3.bn")
csvfile250 = '250Cases.csv'
BE(csvfile250, bNet250)
#
bNet500 = BayesianNet()
bNet500.readNetwork("resources/cancer3.bn")
csvfile500 = '500Cases.csv'
BE(csvfile500, bNet500)

bNet750 = BayesianNet()
bNet750.readNetwork("resources/cancer3.bn")
csvfile750 = '750Cases.csv'
BE(csvfile750, bNet750)

bNet1000 = BayesianNet()
bNet1000.readNetwork("resources/cancer3.bn")
csvfile1000 = '1000Cases.csv'
BE(csvfile1000, bNet1000)

bNet1250 = BayesianNet()
bNet1250.readNetwork("resources/cancer3.bn")
csvfile1250 = '1250Cases.csv'
BE(csvfile1250, bNet1250)

bNet1500 = BayesianNet()
bNet1500.readNetwork("resources/cancer3.bn")
csvfile1500 = '1500Cases.csv'
BE(csvfile1500, bNet1500)

bNet1750 = BayesianNet()
bNet1750.readNetwork("resources/cancer3.bn")
csvfile1750 = '1750Cases.csv'
BE(csvfile1750, bNet1750)

bNet2000 = BayesianNet()
bNet2000.readNetwork("resources/cancer3.bn")
csvfile2000 = '2000Cases.csv'
BE(csvfile2000, bNet2000)

bNet2250 = BayesianNet()
bNet2250.readNetwork("resources/cancer3.bn")
csvfile2250 = '2250Cases.csv'
BE(csvfile2250, bNet2250)

bNet2500 = BayesianNet()
bNet2500.readNetwork("resources/cancer3.bn")
csvfile2500 = '2500Cases.csv'
BE(csvfile2500, bNet2500)

bNet3000 = BayesianNet()
bNet3000.readNetwork("resources/cancer3.bn")
csvfile3000 = '3000Cases.csv'
BE(csvfile3000, bNet3000)

# bNet3500 = BayesianNet()
# bNet3500.readNetwork("resources/cancer3.bn")
# csvfile3500 = '3500Cases.csv'
# BE(csvfile3500, bNet3500)

bNet4000 = BayesianNet()
bNet4000.readNetwork("resources/cancer3.bn")
csvfile4000 = '4000Cases.csv'
BE(csvfile4000, bNet4000)

bNet4500 = BayesianNet()
bNet4500.readNetwork("resources/cancer3.bn")
csvfile4500 = '4500Cases.csv'
BE(csvfile4500, bNet4500)

bNet5000 = BayesianNet()
bNet5000.readNetwork("resources/cancer3.bn")
csvfile5000 = '5000Cases.csv'
BE(csvfile5000, bNet5000)

bNet6000 = BayesianNet()
bNet6000.readNetwork("resources/cancer3.bn")
csvfile6000 = '6000Cases.csv'
BE(csvfile6000, bNet6000)

bNet7000 = BayesianNet()
bNet7000.readNetwork("resources/cancer3.bn")
csvfile7000 = '7000Cases.csv'
BE(csvfile7000, bNet7000)

bNet8000 = BayesianNet()
bNet8000.readNetwork("resources/cancer3.bn")
csvfile8000 = '8000Cases.csv'
BE(csvfile8000, bNet8000)

bNet9000 = BayesianNet()
bNet9000.readNetwork("resources/cancer3.bn")
csvfile9000 = '9000Cases.csv'
BE(csvfile9000, bNet9000)

bNet10000 = BayesianNet()
bNet10000.readNetwork("resources/cancer3.bn")
csvfile10000 = '10000Cases.csv'
BE(csvfile10000, bNet10000)


#print All Kullback-Leibler for all learned network
print('KL divergence for n = 10',meanKL(array10))
print('KL divergence for n = 50', meanKL(array50))
print('KL divergence for n = 100',meanKL(array100))
print('KL divergence for n = 250', calculateKLDivergency(bNet250))
print('KL divergence for n = 500', calculateKLDivergency(bNet500))
print('KL divergence for n = 750', calculateKLDivergency(bNet750))
print('KL divergence for n = 1000', calculateKLDivergency(bNet1000))
print('KL divergence for n = 1250', calculateKLDivergency(bNet1250))
print('KL divergence for n = 1500', calculateKLDivergency(bNet1500))
print('KL divergence for n = 1750', calculateKLDivergency(bNet1750))
print('KL divergence for n = 2000', calculateKLDivergency(bNet2000))
print('KL divergence for n = 2250', calculateKLDivergency(bNet2250))
print('KL divergence for n = 2500', calculateKLDivergency(bNet2500))
print('KL divergence for n = 3000', calculateKLDivergency(bNet3000))
# print('KL divergence for n = 3500', calculateKLDivergency(bNet3500))
print('KL divergence for n = 4000', calculateKLDivergency(bNet4000))
print('KL divergence for n = 4500', calculateKLDivergency(bNet4500))
print('KL divergence for n = 5000', calculateKLDivergency(bNet5000))
print('KL divergence for n = 6000', calculateKLDivergency(bNet6000))
print('KL divergence for n = 7000', calculateKLDivergency(bNet7000))
print('KL divergence for n = 8000', calculateKLDivergency(bNet8000))
print('KL divergence for n = 9000', calculateKLDivergency(bNet9000))
print('KL divergence for n = 10000', calculateKLDivergency(bNet10000))



