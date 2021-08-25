# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:36:25 2021

@author: Santiago
"""

import json
import os
import networkx as nx
from tqdm  import tqdm
from networkx.readwrite import json_graph
from multiprocessing import Process
import glob

# script so that each graph have the same node set even though there is isolated nodes
def Cleaner(arr):

    comp_list=[]
    dir = "final_graph/"
    # get the full node set by combining each node set
    for G in arr:
        comp_list = comp_list + list(G)
    # get rid of duplicates
    comp_set=set(comp_list)
    # for each graph add the missing nodes
    for G in arr:
        for comp in comp_set:
            if
            if comp not in list(G):
                G.add_node(comp,bipartite=0)

        #save them in final_graph
        dataset="graph"+G.name
        res = json_graph.node_link_data(G)
        if not os.path.exists(dir):
            os.makedirs(dir)

        #write new cleaned graphs
        with open(dir+dataset+".json","w") as outfile:
            json.dump(res,outfile)



if __name__ == '__main__':
    # get all the uncleaned graphs in a list for Cleaner
    arr=[]
    for result_path in glob.glob("graphs/graph*",recursive=True):
        with open(result_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            g=json_graph.node_link_graph(data)
            #set graph name if not done already
            g.name=result_path[12:]
            arr.append(g)
    # Clean the graphs and save them
    Cleaner(arr)
