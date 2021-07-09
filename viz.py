# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:49:53 2021

@author: Santiago
"""

import json
import random
import glob
import math
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import numpy as np

def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key

def describe_graph(G):
    print(nx.info(G))
    if nx.is_connected(G):
        print("Diameter: %.4f" %nx.diameter(G)) # Longest shortest path
    else:
        print("Diameter and Avg shortest path length are not defined!")

    print("Sparsity: %.4f" %nx.density(G))  # #edges/#edges-complete-graph

    print("Global clustering coefficient aka Transitivity: %.4f" %nx.transitivity(G))

def visualize_graph(G, with_labels=True, k=None, alpha=0.4, node_shape="o", pos = None):
    #nx.draw_spring(G, with_labels=with_labels, alpha = alpha)
    set2 = [n for n in G.nodes if G.nodes(data="bipartite")[n]==1]
    set1 = [n for n in G.nodes if G.nodes(data="bipartite")[n]==0]

    companyDegree = nx.degree(G, set1)
    valueDegree = nx.degree(G, set2)

    plt.figure(1, figsize=(25, 15))
    if pos==None:
        k = 2.3/math.sqrt(G.order())
        pos = nx.spring_layout(G, k=k)

    if with_labels:
        lab1 = nx.draw_networkx_labels(G, pos, labels=dict([(n, G.nodes(data="name")[n]) for n in G.nodes()]), font_size=20)

    nc2 = nx.draw_networkx_nodes(G, pos, nodelist=set2, node_color='r', node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 200 for v in dict(valueDegree).values()])

    nc1 = nx.draw_networkx_nodes(G, pos, nodelist=set1, node_color='g', node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 100 for v in dict(companyDegree).values()])

    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)

    plt.axis('off')
    axis = plt.gca()
    # axis.set_title("Graph of "+G.name[0:4]+" "+G.name[4:6], fontsize=20)
    axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    axis.set_ylim([1.2 * y for y in axis.get_ylim()])
    plt.tight_layout()
    plt.savefig("figures/indeedgraphs/graph"+G.name+".png",format="png")
    plt.close()
    return pos

def clean_reduce(g) -> None:
    map_path = "company_matcher2.json"
    with open(map_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for comp in list(g):
        if comp in data.keys():
            if data[comp] == "Personalvorsorgestiftung der IBM Corporation, Forschungslaboratorium Zürich":
                g.nodes()[comp]["name"] = "IBM"
            else:
                g.nodes()[comp]["name"] = data[comp]
        else:
            g.nodes()[comp]["name"] = comp.replace("_"," ").title()

    set1 = [n for n in list(g) if g.nodes[n]["bipartite"]==0]
    set2 = [n for n in list(g) if g.nodes[n]["bipartite"]==1]

    deg_comp = dict(g.degree(set1))
    top3=[get_key(deg_comp,n) for n in sorted(set(deg_comp.values()),reverse=True)]
    n1=list(set([n for n in g.neighbors(top3[0])]))
    n2=list(set([n for n in g.neighbors(top3[1])]))
    n3=list(set([n for n in g.neighbors(top3[2])]))
    n1.extend(n2)
    n1.extend(n3)
    n1.extend(top3)
    n1=set(n1)
    G_min=nx.subgraph(g,n1)

    visualize_graph(G_min)
    return n1
dir = "indeed_graph/graph*"
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        #read each graph
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        clean_reduce(g)
