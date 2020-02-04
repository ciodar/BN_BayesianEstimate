from Node import Node
import numpy as np
#import networkx as nx
#import matplotlib.pyplot as plt
import json
import csv


#Create the Graph Network on the basis of the graph generated on Hugin
class BayesianNet:
    def __init__(self):
        self.nodes = dict()

    def readNetwork(self,path):
        #creates a Bayesian Network from the provided input text file.
        #the text file must be a valid json with
        #V --> List of vertices of the network
        #E --> Dictionary, containing for each node, its children.
        #F --> Dictionary, containing for each node, its parents
        # - A cpt as a 1-dimensional array. The columns must follow the order
        # - The node's parents
        # - An original pseudocount. If it is not known, an array containg zeros must be provided.
        with open(path, 'r') as f:
            inputNetwork = json.load(f)
        self.nodes = dict()
        #Reads nodes
        for x in inputNetwork['V']:
            self.nodes[x] = Node(x)
        #Reads childrens for each node
        for e in inputNetwork['E']:
            for c in inputNetwork['E'][e]:
                self.nodes[e].children.append(c)
        #For each node, reads domain,parent and shapes cpt
        for v in inputNetwork['F']:
            self.nodes[v].domain=inputNetwork['F'][v]['values']
            for p in inputNetwork['F'][v]['parents']:
                self.nodes[v].parents.append(p)
            self.nodes[v].cpt=np.zeros(shape=(len(inputNetwork['F'][v]['cpt']),1))
            self.nodes[v].alphas = inputNetwork['F'][v]['alphas']
            self.nodes[v].originalcpt = inputNetwork['F'][v]['cpt']
    #Setter and getter for KL divergency value
    def getDivergenceKL(self):
        return self.divergenceKL
    def setDivergenceKL(self, div):
        self.divergenceKL = div
    #Getter for nodes
    def getNode(self,key):
        return self.nodes[key]
    def getNodes(self):
        return list(self.nodes.values())
    def getNodeKeys(self):
        return list(self.nodes.keys())
    #
    def stride(self, rv, n):
        if n==rv.name:
            return 1
        else:
            card_list = [rv.card()]
            card_list.extend([self.nodes[p].card() for p in rv.parents])
            n_idx = rv.parents.index(n) + 1
            return int(np.prod(card_list[0:n_idx]))
    #Returns the full joint as product of the factors for the specified values of the
    def full_joint(self,val_dict):
        prod = 1
        for n in self.getNodes():
            t = dict((k, val_dict[k]) for k in n.scope())
            prod*= n.cpt[self.cpt_indices(n,t)]
        return prod

    def original_full_joint(self,val_dict):
        prod = 1
        for n in self.getNodes():
            t = dict((k, val_dict[k]) for k in n.scope())
            prod*= n.originalcpt[self.cpt_indices(n,t)]
        return prod

    def cpt_indices(self, target, val_dict):
        """
        Get the index of the CPT which corresponds
        to a dictionary of rv=val sets. This can be
        used for parameter learning to increment the
        appropriate cpt frequency value based on
        observations in the data.

        There is definitely a fast way to do this.
            -- check if (idx - rv_stride*value_idx) % (rv_card*rv_stride) == 0

        Arguments
        ---------
        *target* : a string
            Main RV

        *val_dict* : a dictionary, where
            key=rv,val=rv value

        """
        stride = dict([(n,self.stride(target,n)) for n in target.scope()])
        #if len(val_dict)==len(self.parents(target)):
        #    idx = sum([self.value_idx(rv,val)*stride[rv] \
        #            for rv,val in val_dict.items()])
        #else:
        card = dict([(n, self.nodes[n].card()) for n in target.scope()])
        idx = set(range(len(target.cpt)))
        for rv, val in val_dict.items():
            val_idx = self.nodes[rv].value_idx(val)
            rv_idx = []
            try:
                s_idx = val_idx*stride[rv]
            except KeyError:
                print("Key Error")
            while s_idx < len(target.cpt):
                rv_idx.extend(range(s_idx,(s_idx+stride[rv])))
                s_idx += stride[rv]*card[rv]
            idx = idx.intersection(set(rv_idx))
        if len(list(idx)) !=1:
            print("Len != 1")
        return list(idx)[0]
    #Plots the network
    # def plot(self):
    #     G = nx.Graph()
    #     G.add_nodes_from(self.getNodeKeys())
    #     for k, v in self.nodes.items():
    #         for vv in v.getChildren():
    #             G.add_edge(k, vv)
    #     pos = nx.spring_layout(G)  # positions for all nodes
    #     nx.draw_networkx_nodes(G, pos, node_size=500)
    #     nx.draw_networkx_edges(G, pos)
    #     nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    #     plt.show()

if __name__ == "__main__":

    bn = BayesianNet()
    bn.readNetwork("resources/cancer3.bn")
    for v in bn.getNodes():
        print(v,v.scope())
        d = []
        for s in v.scope():
            d.append(bn.nodes[s].domain)
        #comb = list(itertools.product(*d))
        #print(v,comb)
        csvFile = 'resources/datasets/10Cases.csv'
        with open(csvFile) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            tuples = []
            for row in reader:
                print("row:",row)
                t = tuple(row[n] for n in bn.nodes[v].scope())
                idx = [i for i, n in enumerate(comb) if comb[i] == t][0]
                print("index in the cpt:",idx)




