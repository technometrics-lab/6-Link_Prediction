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

for g in res:
    nx.write_gexf(g,"indeed_graph/graph"+g.name+".gexf")
# del_edge = []
# new_edge = []
# for n in range(len(res)-6):
#     g1 = res[n]
#     g2 = res[n+6]
#     adj1 = nx.to_numpy_matrix(g1)
#     adj2  =nx.to_numpy_matrix(g2, list(g1))
#     new_edge.append(np.count_nonzero((adj1-adj2)==-1))
#     del_edge.append(np.count_nonzero((adj1-adj2)==1))
#
#
#
# graphic = GraphicBase("New edges through time",
#                       "",
#                       "",
#                       "Number of new edges",
#                       date_format=False)
# graphic.ax.plot(new_edge)
# graphic.save_graph("figures/","new_edge6.pdf")
#
#
# graphic = GraphicBase("Edge disappearance through time",
#                       "",
#                       "",
#                       "Number of edge disapearing",
#                       date_format=False)
# graphic.ax.plot(del_edge)
# graphic.save_graph("figures/","del_edge6.pdf")


# arr1 = [0.953, 0.919, 0.89, 0.872, 0.859, 0.849]
# arr2 = [0.952, 0.917, 0.888, 0.87, 0.856, 0.846]
# arr3 = [0.929, 0.901, 0.875, 0.856, 0.84, .827]
# graphic = GraphicBase("Mean AUC for different forecast range",
#                       "",
#                       "Forecast range",
#                       "Mean AUC",
#                       date_format=False)
# graphic.ax.plot(arr2, label = "Katz Index",lw = 7)
# graphic.ax.plot(arr3, label = "Preferential Attachment Index", lw = 7)
# graphic.ax.plot(arr2, label = "Hyperbolic Sine Index", lw = 7)
# graphic.ax.plot(arr1, label = "SVM", lw = 7)
#
# plt.legend(loc="upper right", prop={"size":40})
# plt.xticks(range(6),["1 Month", "2 Months","3 Months",
#                      "4 Months","5 Months","6 Months"])
# graphic.save_graph("figures/","test12345.pdf")
# patent_d = dict(nx.get_edge_attributes(res[0], "pw"))
#
# d_sum = {}
# for g in res[1:-1]:
#     sum_p = 0
#     for edge, dat in patent_d.items():
#         dat1 = g.get_edge_data(edge[0],edge[1], default=0)
#         if not dat1 == 0:
#             dat1 = dat1["pw"]
#         if not dat1 == dat:
#             sum_p += dat-dat1
#             print(g.name," ",edge[0]," to ",edge[1]," ",dat-dat1)
#     d_sum[g.name] = sum_p
#
# for date, sum in d_sum.items():
#     print(date, " ",sum)

# set1 = clean_reduce(res[0])
# for g in res:
#     __ = clean_reduce(g)
#     G_min = nx.subgraph(g,set1)
#     if g.name=="201803":
#         pos1 = visualize_graph(G_min)
#     else:
#         ___ = visualize_graph(G_min, pos=pos1)
# G = nx.random_partition_graph([3,2],0,0.5)
#
# set1 = G.graph["partition"][0]
# set2 = G.graph["partition"][1]
#
# G_comp = nx.Graph()
# arr = []
# edge1 = []
# edge2 = []
# for u in set1:
#     for v in set2:
#         G_comp.add_edge(u, v)
#         arr.append((u, v))
#         if u in G.neighbors(v):
#             edge1.append((u,v))
#         else:
#             edge2.append((u,v))
# pos = nx.spring_layout(G_comp)
# label_pa = nx.preferential_attachment(G, arr)
#
# dict_pa1 = {(u,v):p for u,v,p in label_pa}
# max1 = max(list(dict_pa1.values()))
# dict_pa = {}
# for index, p in dict_pa1.items():
#     dict_pa[index] = round(p/max1,3)
# nc2 = nx.draw_networkx_nodes(G_comp, pos, nodelist=set2, node_color='r', node_shape="o",alpha=0.25)
#
# nc1 = nx.draw_networkx_nodes(G_comp, pos, nodelist=set1, node_color='g', node_shape="o",alpha=0.25)
#
# ec1 = nx.draw_networkx_edges(G_comp, pos,edgelist = edge1, alpha=0.25)
# ec2 = nx.draw_networkx_edges(G_comp, pos,edgelist = edge2, style="dashed", alpha=0.25)
# lab1 =nx.draw_networkx_edge_labels(G_comp, pos,label_pos=0.4,alpha=0.75, edge_labels=dict_pa, font_size=12, font_color="r")
#
#
# plt.axis('off')
# axis = plt.gca()
# axis.set_xlim([1.2 * x for x in axis.get_xlim()])
# axis.set_ylim([1.2 * y for y in axis.get_ylim()])
# plt.tight_layout()
# plt.savefig("figures/Graphs/score_example1.png",format="png")
# plt.close()
