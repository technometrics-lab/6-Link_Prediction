# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:36:25 2021

@author: Santiago
"""

import json
import time
import os
import argparse
import networkx as nx
from tqdm  import tqdm
import glob
from networkx.readwrite import json_graph
import pickle


def Cleaner(arr):
    for n in range(0,len(arr)):
        print(n)
        arr[0].graph["partition"]=[]
    for n in range(1, len(arr)):
        print(n)
        if n == 1:
            real_label = nx.get_node_attributes(arr[0], "content")
        label = nx.get_node_attributes(arr[n], "content")
        for key, item in label.items():
            if item not in real_label.values():

                arr[n].remove_node(key)
                continue
            
            
            key_item = get_key(real_label,item)
            
            if not key_item == key:
                arr[n] = nx.relabel_nodes(arr[n], {key : key_item},copy=False)

                
    iso_set = set(nx.isolates(arr[0]))
    for n in range(1, len(arr)):
        print(n)
        iso_set = iso_set & set(nx.isolates(arr[n]))
    for G in arr:
        G.remove_nodes_from(iso_set)
    return arr

def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key
        
if __name__ == '__main__':
    
    path = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    arr = []
    for p in path:
        result_path = "graphs/graph" + p + ".json"
        with open(result_path, 'r', encoding = 'utf-8') as file:
            data = json.load(file)
            arr.append(json_graph.node_link_graph(data))
    
    arrnew = Cleaner(arr)
    
    n = 0
    for g in arrnew:
        res = json_graph.node_link_data(g)
        dir = "clean_graphs/"
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        dataset = "graph"+path[n]
        n += 1
    
        with open(dir + dataset + ".json", 'w') as outfile:
            json.dump(res, outfile)