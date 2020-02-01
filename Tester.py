from BayesianNetwork import BayesianNet
from KLDivergenceCalculation import calculateKLDivergency
from BayesianEstimate import bayesEstimate
bn = BayesianNet()
bn.readNetwork("resources/cancer3uniform.bn")
bn.plot()

bNet10 = BayesianNet()
bNet10.readNetwork("resources/cancer3uniform.bn")
csvfile10 = 'resources/datasets/10Cases.csv'
bayesEstimate(csvfile10, bNet10)
print('KL divergence for n = 10', calculateKLDivergency(bNet10))

csvfile50 = 'resources/datasets/50Cases.csv'
bayesEstimate(csvfile50, bNet10)
print('KL divergence for n = 50', calculateKLDivergency(bNet10))

csvfile100 = 'resources/datasets/100Cases.csv'
bayesEstimate(csvfile100, bNet10)
print('KL divergence for n = 100', calculateKLDivergency(bNet10))
#
#Calculate MLE for all other datasets
# bNet250 = BayesianNet()
# bNet250.readNetwork("resources/cancer3uniform.bn")
csvfile250 = 'resources/datasets/250Cases.csv'
#bayesEstimate(csvfile250, bNet250)
bayesEstimate(csvfile250, bNet10)
print('KL divergence for n = 250', calculateKLDivergency(bNet10))
#
# bNet500 = BayesianNet()
# bNet500.readNetwork("resources/cancer3uniform.bn")
csvfile500 = 'resources/datasets/500Cases.csv'
#bayesEstimate(csvfile500, bNet500)
bayesEstimate(csvfile500, bNet10)
print('KL divergence for n = 500', calculateKLDivergency(bNet10))
# bNet750 = BayesianNet()
# bNet750.readNetwork("resources/cancer3uniform.bn")
csvfile750 = 'resources/datasets/750Cases.csv'
#bayesEstimate(csvfile750, bNet750)
bayesEstimate(csvfile750, bNet10)
print('KL divergence for n = 750', calculateKLDivergency(bNet10))

# bNet1000 = BayesianNet()
# bNet1000.readNetwork("resources/cancer3uniform.bn")
csvfile1000 = 'resources/datasets/1000Cases.csv'
#bayesEstimate(csvfile1000, bNet1000)
bayesEstimate(csvfile1000, bNet10)
print('KL divergence for n = 1000', calculateKLDivergency(bNet10))
# bNet1250 = BayesianNet()
# bNet1250.readNetwork("resources/cancer3uniform.bn")
csvfile1250 = 'resources/datasets/1250Cases.csv'
#bayesEstimate(csvfile1250, bNet1250)
bayesEstimate(csvfile1250, bNet10)
print('KL divergence for n = 1250', calculateKLDivergency(bNet10))
# bNet1500 = BayesianNet()
# bNet1500.readNetwork("resources/cancer3uniform.bn")
csvfile1500 = 'resources/datasets/1500Cases.csv'
#bayesEstimate(csvfile1500, bNet1500)
bayesEstimate(csvfile1500, bNet10)
print('KL divergence for n = 1500', calculateKLDivergency(bNet10))
# bNet1750 = BayesianNet()
# bNet1750.readNetwork("resources/cancer3uniform.bn")
csvfile1750 = 'resources/datasets/1750Cases.csv'
#bayesEstimate(csvfile1750, bNet1750)
bayesEstimate(csvfile1750, bNet10)
print('KL divergence for n = 1750', calculateKLDivergency(bNet10))
#bNet2000 = BayesianNet()
#bNet2000.readNetwork("resources/cancer3uniform.bn")
csvfile2000 = 'resources/datasets/2000Cases.csv'
#bayesEstimate(csvfile2000, bNet2000)
bayesEstimate(csvfile2000, bNet10)
print('KL divergence for n = 2000', calculateKLDivergency(bNet10))
# bNet2250 = BayesianNet()
# bNet2250.readNetwork("resources/cancer3uniform.bn")
csvfile2250 = 'resources/datasets/2250Cases.csv'
# bayesEstimate(csvfile2250, bNet2250)
bayesEstimate(csvfile2250, bNet10)
print('KL divergence for n = 2250', calculateKLDivergency(bNet10))
# bNet2500 = BayesianNet()
# bNet2500.readNetwork("resources/cancer3uniform.bn")
csvfile2500 = 'resources/datasets/2500Cases.csv'
#bayesEstimate(csvfile2500, bNet2500)
bayesEstimate(csvfile2500, bNet10)
print('KL divergence for n = 2500', calculateKLDivergency(bNet10))
# bNet3000 = BayesianNet()
# bNet3000.readNetwork("resources/cancer3uniform.bn")
csvfile3000 = 'resources/datasets/3000Cases.csv'
#bayesEstimate(csvfile3000, bNet3000)
bayesEstimate(csvfile3000, bNet10)
print('KL divergence for n = 3000', calculateKLDivergency(bNet10))
# bNet3500 = BayesianNet()
# bNet3500.readNetwork("resources/cancer3uniform.bn")
# csvfile3500 = '3500Cases.csv'
# bayesEstimate(csvfile3500, bNet3500)

# bNet4000 = BayesianNet()
# bNet4000.readNetwork("resources/cancer3uniform.bn")
#csvfile4000 = 'resources/datasets/4000Cases.csv'
#bayesEstimate(csvfile4000, bNet4000)

#bNet4000 = BayesianNet()
#bNet4000.readNetwork("resources/cancer3uniform.bn")
csvfile4000 = 'resources/datasets/4000Cases.csv'
bayesEstimate(csvfile4000, bNet10)
print('KL divergence for n = 4000', calculateKLDivergency(bNet10))
#bNet4500 = BayesianNet()
#bNet4500.readNetwork("resources/cancer3uniform.bn")
csvfile4500 = 'resources/datasets/4500Cases.csv'
bayesEstimate(csvfile4500, bNet10)
print('KL divergence for n = 4500', calculateKLDivergency(bNet10))
#bNet5000 = BayesianNet()
#bNet5000.readNetwork("resources/cancer3uniform.bn")
csvfile5000 = 'resources/datasets/5000Cases.csv'
#bayesEstimate(csvfile5000, bNet5000)
bayesEstimate(csvfile5000, bNet10)
print('KL divergence for n = 5000', calculateKLDivergency(bNet10))
# bNet6000 = BayesianNet()
# bNet6000.readNetwork("resources/cancer3uniform.bn")
csvfile6000 = 'resources/datasets/6000Cases.csv'
#bayesEstimate(csvfile6000, bNet6000)
bayesEstimate(csvfile6000, bNet10)
print('KL divergence for n = 6000', calculateKLDivergency(bNet10))
# bNet7000 = BayesianNet()
# bNet7000.readNetwork("resources/cancer3uniform.bn")
csvfile7000 = 'resources/datasets/7000Cases.csv'
#bayesEstimate(csvfile7000, bNet7000)
bayesEstimate(csvfile7000, bNet10)
print('KL divergence for n = 7000', calculateKLDivergency(bNet10))
# bNet8000 = BayesianNet()
# bNet8000.readNetwork("resources/cancer3uniform.bn")
csvfile8000 = 'resources/datasets/8000Cases.csv'
#bayesEstimate(csvfile8000, bNet8000)
bayesEstimate(csvfile8000, bNet10)
print('KL divergence for n = 8000', calculateKLDivergency(bNet10))

# bNet9000 = BayesianNet()
# bNet9000.readNetwork("resources/cancer3uniform.bn")
csvfile9000 = 'resources/datasets/9000Cases.csv'
#bayesEstimate(csvfile9000, bNet9000)
bayesEstimate(csvfile9000, bNet10)
print('KL divergence for n = 9000', calculateKLDivergency(bNet10))

# bNet10000 = BayesianNet()
# bNet10000.readNetwork("resources/cancer3uniform.bn")
csvfile10000 = 'resources/datasets/10000Cases.csv'
#bayesEstimate(csvfile10000, bNet10000)
bayesEstimate(csvfile10000, bNet10)
print('KL divergence for n = 10000', calculateKLDivergency(bNet10))

