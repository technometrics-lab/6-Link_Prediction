# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:49:53 2021

@author: Santiago
"""

import json
import networkx as nx
from networkx.readwrite import json_graph
from viz_graph import *
import random
import glob

def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key


def clean_reduce(g) -> None:
    map_path = "company_matcher2.json"
    with open(map_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for comp in list(g):
        if comp in data.keys():
            g.nodes()[comp]["name"] = data[comp]
        else:
            g.nodes()[comp]["name"] = comp

    set1 = [n for n in list(g) if g.nodes[n]["bipartite"]==0]
    set2 = [n for n in list(g) if g.nodes[n]["bipartite"]==1]

    deg_comp = g.degree(set1)
    print(deg_comp)
    deg_val=[deg_comp[n] for n in set1]
    top3=[get_key(deg_comp,n) for n in sorted(deg_val)[0:3]]
    print(top3)
    # visualize_graph(G_min1)

dir = "final_graph/graph*"
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        #read each graph
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        clean_reduce(g)
