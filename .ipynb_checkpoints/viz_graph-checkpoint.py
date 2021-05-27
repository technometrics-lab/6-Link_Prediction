# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:59:33 2021

@author: Santiago
"""

import json
import os
import networkx as nx
from tqdm  import tqdm
from networkx.readwrite import json_graph
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np
import math

path = ["01","02","03","04","05","06","07","08","09","10","11"]
arr = []
for p in path:
    result_path = "test_graphs/graph" + p + ".json"
    with open(result_path, 'r', encoding = 'utf-8') as file:
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        g.name=p
        arr.append(g)

G=arr[0]
def describe_graph(G):
    print(nx.info(G))
    if nx.is_connected(G):
        #print("Avg. Shortest Path Length: %.4f" %nx.average_shortest_path_length(G))
        print("Diameter: %.4f" %nx.diameter(G)) # Longest shortest path
    else:
        print("Graph is not connected")
        print("Diameter and Avg shortest path length are not defined!")
    print("Sparsity: %.4f" %nx.density(G))  # #edges/#edges-complete-graph
    # #closed-triplets(3*#triangles)/#all-triplets
    print("Global clustering coefficient aka Transitivity: %.4f" %nx.transitivity(G))
    
def visualize_graph(G, with_labels=True, k=None, alpha=0.4, node_shape='o'):
    #nx.draw_spring(G, with_labels=with_labels, alpha = alpha)
    set2=[n for n in G.nodes if G.nodes(data="bipartite")[n]==1]
    set1=[n for n in G.nodes if G.nodes(data="bipartite")[n]==0]
    companyDegree = nx.degree(G, set1) 
    valueDegree = nx.degree(G, set2)
    
    plt.figure(1,figsize=(25,15))
    k=2.3/math.sqrt(G.order())
    pos = nx.spring_layout(G,k=k)
    if with_labels:
        lab1 = nx.draw_networkx_labels(G, pos, labels=dict([(n, G.nodes(data="name")[n]) for n in G.nodes()]), font_size=20)

    nc2 = nx.draw_networkx_nodes(G, pos, nodelist=set2, node_color='r', node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 200 for v in dict(valueDegree).values()])
    nc1 = nx.draw_networkx_nodes(G, pos, nodelist=set1, node_color='g', node_shape=node_shape,alpha=0.25,
                                 node_size=[v * 100 for v in dict(companyDegree).values()])
    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
    plt.tight_layout()
    plt.savefig("graph1.png",dpi=250)



def filter_dict(G, percentage, set1, set2):
    """Selects values to delete from the graph G according to ...
    Args:
        - G: graph 
        - percentage: percentage of entities to keep
        - set1: first group of nodes
        - set2: second group of nodes
    Return:
        - to_delete: list of values to delete
    """

    degree_set2 = list(dict(nx.degree(G, set2)).values())
    
    threshold_companies = math.ceil( len(set2)/percentage )
    
    
    if threshold_companies > np.max(degree_set2): # not okay because we would not plot anything
        threshold_companies=np.mean(degree_set2)
    
    dict_nodes = nx.degree(G, set1) 
    
    to_delete= []
    
    # Iterate over all the items in dictionary
    for (key, value) in dict(dict_nodes).items():
        
        if value <= threshold_companies:
            to_delete.append(key)
    
    return to_delete

# =============================================================================
# print("The graph is connected: " + str(nx.is_connected(G)))
# comp = list(nx.connected_components(G))
# print('The graph contains', len(comp), 'connected components')
# 
# largest_comp = max(comp, key=len)
# percentage_lcc = len(largest_comp)/G.number_of_nodes() * 100
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
