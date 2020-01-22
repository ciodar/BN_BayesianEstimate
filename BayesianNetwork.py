from Node import Node
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json

#Create the Graph Network on the basis of the graph generated on Hugin
class BayesianNet:
    def __init__(self):
        self.nodes = dict()
    def getDivergenceKL(self):
        return self.divergenceKL
    def setDivergenceKL(self, div):
        self.divergenceKL = div
    def getNodes(self):
        return list(self.nodes.keys())

    #def parents(self,rv):
    #    return self.nodes[rv].parents
    #def cpt(self,target):
    #    return self.nodes[target].cpt

    def rec_idx(self,rv,val_dict):
        self_idx = self.nodes[rv].value_idx(val_dict[rv])
        prt_card = 0
        idx = 0
        for i,p in enumerate(self.nodes[rv].parent):
            prt_card += self.nodes[p].card()
            idx += self.rec_idx(p,val_dict)
        return (self_idx*prt_card) + idx

    def factor(self,rv,val_dict):
        return rv.cpt[self.cpt_indices(rv,val_dict)]
    def original_factor(self,rv,val_dict):
        return rv.originalcpt[self.original_cpt_indices(rv,val_dict)]

    def combination_size(self):
        size = 1
        for v in self.getNodes():
            size *= self.nodes[v].card()
        return size

    def full_joint(self,val_dict):
        prod = 1
        for i in self.nodes.values():
            prod*= self.factor(i,val_dict)
        return prod

    def original_full_joint(self,val_dict):
        prod = 1
        for i in self.nodes.values():
            prod*= self.original_factor(i,val_dict)
        return prod

    def stride(self, rv, n):
        if n == rv.name:
            return 1
        else:
            card_list = [rv.card()]
            card_list.extend([self.nodes[p].card() for p in rv.parents])
            n_idx = rv.parents.index(n) + 1
            return int(np.prod(card_list[0:n_idx]))

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
        *target* : a Node object instance
            Main RV
        *val_dict* : a dictionary, where
            key=rv,val=rv value
        """
        stride = dict([(n, self.stride(target, n)) for n in target.scope()])
        # if len(val_dict)==len(self.parents(target)):
        #    idx = sum([self.value_idx(rv,val)*stride[rv] \
        #            for rv,val in val_dict.items()])
        # else:
        card = dict([(n, self.nodes[n].card()) for n in target.scope()])
        idx = set(range(len(target.cpt)))
        for rv, val in val_dict.items():
            if rv in target.scope():
                val_idx = self.nodes[rv].value_idx(val)
                rv_idx = []
                s_idx = val_idx * stride[rv]
                while s_idx < len(target.cpt):
                    rv_idx.extend(range(s_idx, (s_idx + stride[rv])))
                    s_idx += stride[rv] * self.nodes[rv].card()
                idx = idx.intersection(set(rv_idx))
        if len(idx) != 1:
            print(target.name, " ", val_dict)
        return list(idx)[0]
    def original_cpt_indices(self, target, val_dict):
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
        *target* : a Node object instance
            Main RV
        *val_dict* : a dictionary, where
            key=rv,val=rv value
        """
        stride = dict([(n, self.stride(target, n)) for n in target.scope()])
        # if len(val_dict)==len(self.parents(target)):
        #    idx = sum([self.value_idx(rv,val)*stride[rv] \
        #            for rv,val in val_dict.items()])
        # else:
        card = dict([(n, self.nodes[n].card()) for n in target.scope()])
        idx = set(range(len(target.originalcpt)))
        for rv, val in val_dict.items():
            if rv in target.scope():
                val_idx = self.nodes[rv].value_idx(val)
                rv_idx = []
                s_idx = val_idx * stride[rv]
                while s_idx < len(target.originalcpt):
                    rv_idx.extend(range(s_idx, (s_idx + stride[rv])))
                    s_idx += stride[rv] * self.nodes[rv].card()
                idx = idx.intersection(set(rv_idx))
        if len(idx) != 1:
            print(target.name," ",val_dict)
        return list(idx)[0]

    def readNetwork(self,path):
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
            self.nodes[v].originalcpt = inputNetwork['F'][v]['cpt']

    def plot(self):
        G = nx.Graph()
        G.add_nodes_from(self.getNodes())
        for k, v in self.nodes.items():
            for vv in v.getChildren():
                G.add_edge(k, vv)
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        plt.show()






