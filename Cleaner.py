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


def Cleaner(g,real_label):

    label = nx.get_node_attributes(g, "content")
    val = real_label.values()
    for key, item in tqdm(label.items()):
        if item not in val:
           g.remove_node(key)
           continue
    
        key_item = get_key(real_label,item)
        
        if not key_item == key:
             nx.relabel_nodes(g, {key : key_item},copy=False)
             
    res = json_graph.node_link_data(g)
    dir = "relabeled_graphs/"
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    dataset = "graph"+g.name

    with open(dir + dataset + ".json", 'w') as outfile:
        json.dump(res, outfile)
    

def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key


class Labeler(Process):
    
    def __init__(self,graph,label):
        Process.__init__(self)
        #path that caracterizes one graph
        self.graph = graph
        self.label = label
        
    def run(self):
        print("start")
        Cleaner(self.graph, self.label)
        print("stop")

if __name__ == '__main__':
    
    path = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    arr = []
    for p in path:
        result_path = "graphs/graph" + p + ".json"
        with open(result_path, 'r', encoding = 'utf-8') as file:
            data = json.load(file)
            g=json_graph.node_link_graph(data)
            g.name=p
            arr.append(g)
        
    label=nx.get_node_attributes(arr[0],"content")
    labeler_list=[]

    for g in arr[1:-1]:
        labeler=Labeler(g,label)
        labeler_list.append(labeler)
        labeler.start()
    for labeler in labeler_list:
        labeler.join()
            
          
            
            
            
#     iso_set = set(nx.isolates(arr[0]))
#     for n in range(1, len(arr)):
#         print(n)
#         iso_set = iso_set & set(nx.isolates(arr[n]))
#     for G in arr:
#         G.remove_nodes_from(iso_set)
#     for g in arr:
#         l=list(nx.isolates(g))
#         g.remove_nodes_from(l)