import numpy as np

#create Node class that rapresents the random variable in Bayesian Network
class Node:

    def __init__(self, name):
        self.name = name
        self.originalcpt = np.array
        self.cpt = np.array
        self.parents = []
        self.children = []
        self.domain = []
        self.caption = None

    def getChildren(self):
        return self.children

    def getParents (self):
        return self.parents

    def getCPT(self):
        return self.cpt

    def card(self):
        return len(self.domain)
    def scope(self):
        scope = [self.name]
        scope.extend(self.parents)
        return scope
    def value_idx(self, val):
        try:
            return self.domain.index(val)
        except ValueError:
            print("Value Index Error")
            return -1