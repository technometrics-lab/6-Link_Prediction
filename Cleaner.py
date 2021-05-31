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


def Cleaner(arr):
    comp_list=[]
    dir = "final_graph/"
    for G in arr:
        comp_list = comp_list + list(G)

    comp_set=set(comp_list)
    for G in arr:
        for comp in comp_set:
            if
            if comp not in list(G):
                G.add_node(comp,bipartite=0)

        dataset="graph"+G.name
        res = json_graph.node_link_data(G)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir+dataset+".json","w") as outfile:
            json.dump(res,outfile)



if __name__ == '__main__':

    arr=[]
    for result_path in glob.glob("graphs/graph*",recursive=True):
        with open(result_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            g=json_graph.node_link_graph(data)
            g.name=result_path[12:]
            arr.append(g)
    Cleaner(arr)
