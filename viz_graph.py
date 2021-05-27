# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:59:33 2021

@author: Santiago
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

def describe_graph(G):
    print(nx.info(G))
    if nx.is_connected(G):
        print("Diameter: %.4f" %nx.diameter(G)) # Longest shortest path
    else:
        print("Diameter and Avg shortest path length are not defined!")

    print("Sparsity: %.4f" %nx.density(G))  # #edges/#edges-complete-graph

    print("Global clustering coefficient aka Transitivity: %.4f" %nx.transitivity(G))

def visualize_graph(G, with_labels=True, k=None, alpha=0.4, node_shape='o'):
    #nx.draw_spring(G, with_labels=with_labels, alpha = alpha)
    set2 = [n for n in G.nodes if G.nodes(data="bipartite")[n]==1]
    set1 = [n for n in G.nodes if G.nodes(data="bipartite")[n]==0]
    companyDegree = nx.degree(G, set1)
    valueDegree = nx.degree(G, set2)

    plt.figure(1, figsize=(25, 15))
    k = 2.3/math.sqrt(G.order())
    pos = nx.spring_layout(G, k=k)
    if with_labels:
        lab1 = nx.draw_networkx_labels(G, pos, labels=dict([(n, G.nodes(data="name")[n]) for n in G.nodes()]), font_size=20)

    nc2 = nx.draw_networkx_nodes(G, pos, nodelist=set2, node_color='r',
                                 node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 200 for v in dict(valueDegree).values()])
    nc1 = nx.draw_networkx_nodes(G, pos, nodelist=set1, node_color='g', node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 100 for v in dict(companyDegree).values()])
    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    axis.set_ylim([1.2 * y for y in axis.get_ylim()])
    plt.tight_layout()
    plt.savefig("graph1.png")


# =============================================================================
# print("The graph is connected: " + str(nx.is_connected(G)))
# comp = list(nx.connected_components(G))
# print('The graph contains', len(comp), 'connected components')
#
# largest_comp = max(comp, key=len)
# percentage_lcc = len(largest_comp)/G.number_of_nodes() * 100
# G_con=nx.subgraph(G,largest_comp)
# describe_graph(G_con)
# print('The largest component has', len(largest_comp), 'nodes', 'accounting for %.2f'% percentage_lcc, '% of the nodes')
#
# print("the second largest component has", len(sorted(comp,key=len)[-2]), "nodes")
#
# G_connected=nx.subgraph(G,largest_comp)
# print("The largest component is bipartite: "+ str(bipartite.is_bipartite(G_connected)))
#
# print("the graph contains "+ str(len(list(nx.isolates(G))))+" isolated nodes.")
#
# set2=[n for n in G.nodes if G.nodes(data="bipartite")[n]==1]
# set1=[n for n in G.nodes if G.nodes(data="bipartite")[n]==0]
#
#
# valueDegree = dict(nx.degree(G, set2))
# =============================================================================
