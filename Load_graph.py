import json
import os
import glob
from statistics import mean
import networkx as nx
from tqdm  import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.readwrite import json_graph
import numpy as np


def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key


map_path = "company_matcher2.json"
with open(map_path, "r", encoding="utf-8") as file:
    data1 = json.load(file)


dir = "final_graph/graph*"
rep=[]
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        set1=[x for x, y in g.nodes(data=True) if y["bipartite"] == 0]
        set2=[x for x, y in g.nodes(data=True) if y["bipartite"] == 1]
        rep.append(g)

adj1=nx.to_numpy_matrix(rep[0])
adj2 = nx.to_scipy_sparse_matrix(rep[0])
adj1+adj2
        # dir1="final_graph/"
        # dataset="graph"+g.name
        # res = json_graph.node_link_data(g)
        #
        #
        # with open(dir1+dataset+".json","w") as outfile:
        #     json.dump(res,outfile)
