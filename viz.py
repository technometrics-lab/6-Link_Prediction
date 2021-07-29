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

# usefull function to get key name from value in a dict/ returns the first match
def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key

# graph drawing function
def visualize_graph(G, with_labels=True, k=None, alpha=0.4, node_shape="o", pos = None):

    # get company and technology node set
    set2 = [n for n in G.nodes if G.nodes(data="bipartite")[n]==1]
    set1 = [n for n in G.nodes if G.nodes(data="bipartite")[n]==0]
    # get node:degree dict for adjusting node size
    companyDegree = nx.degree(G, set1)
    valueDegree = nx.degree(G, set2)

    plt.figure(1, figsize=(25, 15))
    # get node positioning in plot
    if pos==None:
        k = 2.3/math.sqrt(G.order())
        pos = nx.spring_layout(G, k=k)
    # draw node labels here name is real name of companies and technologies
    if with_labels:
        lab1 = nx.draw_networkx_labels(G, pos, labels=dict([(n, G.nodes(data="name")[n]) for n in G.nodes()]), font_size=20)
    # draw tech nodes
    nc2 = nx.draw_networkx_nodes(G, pos, nodelist=set2, node_color='r', node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 200 for v in dict(valueDegree).values()])
    # draw company nodes
    nc1 = nx.draw_networkx_nodes(G, pos, nodelist=set1, node_color='g', node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 100 for v in dict(companyDegree).values()])
    #draw edges
    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)

    #final plot adjustments
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    axis.set_ylim([1.2 * y for y in axis.get_ylim()])
    plt.tight_layout()
    plt.savefig("figures/indeedgraphs/graph"+G.name+".png",format="png")
    plt.close()
    # this return only if we want same positioning for further plotting
    return pos


# function that selects the top 3 company in the graph and draws the subgraphs of
# the neighborhood of them
def clean_reduce(g) -> None:
    # get the zefix code to real company name map
    map_path = "company_matcher2.json"
    with open(map_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # set the name attribute for each node could be set later only on the
    # selected company
    for comp in list(g):
            g.nodes()[comp]["name"] = data[comp]
        else:
            g.nodes()[comp]["name"] = comp.replace("_"," ").title()

    # gets the company and technology set
    set1 = [n for n in list(g) if g.nodes[n]["bipartite"]==0]
    set2 = [n for n in list(g) if g.nodes[n]["bipartite"]==1]

    #get the top3 companies
    deg_comp = dict(g.degree(set1))
    top3=[get_key(deg_comp,n) for n in sorted(set(deg_comp.values()),reverse=True)]
    # get their neigbors
    n1=list(set([n for n in g.neighbors(top3[0])]))
    n2=list(set([n for n in g.neighbors(top3[1])]))
    n3=list(set([n for n in g.neighbors(top3[2])]))
    n1.extend(n2)
    n1.extend(n3)
    n1.extend(top3)
    #eliminate duplicates
    n1=set(n1)
    test = list(n1)
    #dont know why but some companies get into neighborhoods even though they
    # are not connected to the top3 comapnies so we get rid of them
    for elem in test:
        if elem.startswith("CH") and elem not in [top3[0], top3[1], top3[2]]:
            n1.remove(elem)
    G_min=nx.subgraph(g,n1)
    # draw the subgraph
    visualize_graph(G_min)

if __name__ == '__main__':
    dir = "indeed_graph/graph*"
    # load each graph and plot them
    for path in glob.glob(dir,recursive=True):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            g=json_graph.node_link_graph(data)
            clean_reduce(g)
