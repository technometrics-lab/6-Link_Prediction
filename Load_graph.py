import json
import os
import glob
from statistics import mean
import networkx as nx
from tqdm  import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.readwrite import json_graph



def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key


map_path = "company_matcher2.json"
with open(map_path, "r", encoding="utf-8") as file:
    data1 = json.load(file)

flipped = {}

for key, item in data1.items():
    if item not in flipped:
        flipped[item]=[key]
    else:
        flipped[item].append(key)
for k, i in flipped.items():
    if len(i)>1:
        print(k," ",len(i))

dir = "final_graph/graph*"
rep=[]
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        a=[item for key, item in flipped.items() if len(item)>1]
        for conc in a:
            true=conc[0]
            for item in conc[1:-1]:
                nx.contracted_nodes(g,true,item,copy=False)

        dir1="final_graph/"
        dataset="graph"+g.name
        res = json_graph.node_link_data(g)


        with open(dir1+dataset+".json","w") as outfile:
            json.dump(res,outfile)
