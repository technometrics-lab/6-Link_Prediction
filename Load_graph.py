import json
import os
import glob
import random
import math
from statistics import mean
import networkx as nx
from tqdm  import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.readwrite import json_graph
import numpy as np
from graphic_base import GraphicBase
from viz import clean_reduce, visualize_graph

def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key


map_path = "company_matcher2.json"
with open(map_path, "r", encoding="utf-8") as file:
    data1 = json.load(file)


dir = "indeed_graph/graph*"
res = []
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        res.append(g)


print(nx.info(res[0]))
key=["ka", "pa", "sh", "svm"]
label={"ka":"Katz Index", "pa":"Preferential Attachment Index",
       "sh":"Hyperbolic Sine Index", "svm":"SVM"}
ka = [0.853, 0.759, 0.689, 0.626, 0.622, 0.612]
pa = [0.849, 0.754, 0.683, 0.65, 0.633, 0.619]
hs = [0.853, 0.759, 0.689, 0.626, 0.621, 0.612]
svm = [0.856, 0.762, 0.695, 0.641, 0.631, 0.621]

graphic = GraphicBase("",
                      "",
                      "Forecast range",
                      "Mean AUC",
                      date_format=False)
graphic.ax.plot(ka, label=label["ka"], lw =5)
graphic.ax.plot(pa, label = label["pa"], lw = 5)
graphic.ax.plot(hs, label =label["sh"], lw = 5)
graphic.ax.plot(svm, label=label["svm"], lw = 5)
plt.legend(loc ="upper right", prop={"size": 40})
plt.xticks(range(0,len(ka)),["1 month","2 months","3 months","4 months","5 months","6 months"])
graphic.save_graph("figures/","Mean_Auc_evo6.pdf")
