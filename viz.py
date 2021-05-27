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


def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key
        
result_path = "test_graphs/graph11.json"
with open(result_path, 'r', encoding = 'utf-8') as file:
    data = json.load(file)
    g = json_graph.node_link_graph(data)

map_path = "company_matcher.json"
with open(map_path, "r", encoding="utf-8") as file:
    data = json.load(file)

for key, comp in nx.get_node_attributes(g, "content").items():
    if comp[0] in data.keys():
        g.nodes()[key]["name"] = data[comp[0]]
    else:
        if g.nodes()[key]["bipartite"] == 1:
            tech = g.nodes()[key]["content"][0].split("_")
            tech = " ".join(tech)
            g.nodes()[key]["name"] = tech

set1 = [n for n in list(g) if g.nodes[n]["bipartite"]==0]
set2 = [n for n in list(g) if g.nodes[n]["bipartite"]==1]

degcomp = dict(g.degree(set1))
degtech = dict(g.degree(set2))
sortcomp = sorted(degcomp.values(), reverse=True)
sortindexcomp = [get_key(degcomp, item) for item in sortcomp]
top3comp = [g.nodes[n]["content"][0] for n in sortindexcomp[0:3]]
setn1 = list(g[sortindexcomp[0]])
setn1.append(sortindexcomp[0])
setn2 = list(g[sortindexcomp[2]])
setn2.append(sortindexcomp[1])
setn3 = list(g[sortindexcomp[3]])
setn3.append(sortindexcomp[3])
setn4 = list(g[sortindexcomp[5]])
setn4.append(sortindexcomp[5])
setn1.extend(setn2)
setn1.extend(setn3)
setn1.extend(setn4)
setn1 = list(set(setn1))
G_min = nx.subgraph(g,setn1)
print(G_min.degree())
for n,d in G_min.degree():
    print(n," ",d)
    if d == 1:
        if random.random() < 0.9:
            setn1.remove(n)
    elif d == 2:
        if random.random() < 0.4:
            setn1.remove(n)
    elif d == 3:
        if random.random() < 0.4:
            setn1.remove(n)
    elif d == 4:
        if random.random() < 0.4:
            setn1.remove(n)

G_min1 = nx.subgraph(g, setn1)
G_min1.nodes()[2734]["name"]="IBM"
visualize_graph(G_min1)


