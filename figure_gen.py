import json
import os
import glob
import numpy as np
from statistics import mean
import networkx as nx
from tqdm  import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.readwrite import json_graph
from graphic_base import GraphicBase


def get_key(dict, search_item):
    for key, item in dict.items():
        if item == search_item:
            return key


map_path = "company_matcher2.json"
with open(map_path, "r", encoding="utf-8") as file:
    data1 = json.load(file)

dir = "indeed_graph/graph*"
arrdeg = []
arrcomp = []
arrtop = []
arrtopc = []
key_tech = []
key_comp = []
iw_max = []
pw_max = []
iw_mean = []
pw_mean = []
iw_dict = {}
pw_dict = {}
new_edge = []
del_edge = []
same_edge = []
perc_edge = []
g_old = 0
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        #read each graph
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        adj1 = nx.to_numpy_matrix(g)


        #get number of edges for each tim
        arrdeg.append(g.number_of_edges())
        #inspect new edge/ edge deletion and edge keeping
        if not g_old == 0:
            adj2 = nx.to_numpy_matrix(g_old, list(g))
            new_edge.append(np.count_nonzero((adj1-adj2)==-1)/2)
            same_edge.append(np.count_nonzero(np.logical_and((adj1==1),(adj2==1)))/2)
            del_edge.append(np.count_nonzero((adj1-adj2)==1)/2)
            perc_edge.append((del_edge[-1]+new_edge[-1])/arrdeg[-1])
        #get size of largest connected component
        comp = list(nx.connected_components(g))
        largest_comp = len(max(comp, key=len))
        arrcomp.append(largest_comp)

        #get most linked technology
        tech=[n for n in list(g) if g.nodes[n]["bipartite"]==1]
        deg_tech=dict(g.degree(tech))
        sort_tech=sorted(deg_tech.values(), reverse=True)
        arrtop.append(sort_tech[0])
        key_tech.append(get_key(deg_tech, sort_tech[0]))

        #get most linked company by month
        comp=[n for n in list(g) if g.nodes[n]["bipartite"]==0]
        deg_comp=dict(g.degree(comp))
        sort_comp=sorted(deg_comp.values(), reverse=True)
        arrtopc.append(sort_comp[0])
        key_comp.append(data1[get_key(deg_comp, sort_comp[0])])

        #analysis of IW and PW
        iw=nx.get_edge_attributes(g,"iw")
        iw_max.append(max(list(iw.values())))

        pw=nx.get_edge_attributes(g,"pw")
        pw_max.append(max(list(pw.values())))

        iw_mean.append(mean(list(iw.values())))
        pw_mean.append(mean(list(pw.values())))

        iw_dict[g.name] = list(iw.values())
        pw_dict[g.name] = list(pw.values())

        g_old=g

dir="figures/indeed_stats/"
graphic = GraphicBase("Number of edges through time",
                      "",
                      "",
                      "Number of edges",
                      date_format=False)
graphic.ax.plot(arrdeg)
graphic.save_graph("figures/indeed_stats/","number_edges.pdf")


graphic = GraphicBase("Size of the largest connected component through time",
                      "",
                      "",
                      "Size of largest connected component",
                      date_format=False)
graphic.ax.plot(arrcomp)
graphic.save_graph("figures/indeed_stats/","number_conncomp.pdf")


graphic = GraphicBase("Degree of the most linked technology through time",
                      "",
                      "",
                      "degree of most linked technology",
                      date_format=False)
graphic.ax.plot(range(len(arrtop)),arrtop)
for x,y in zip(range(len(arrtop)), arrtop):
    label = key_tech[x].replace("_"," ")
    plt.annotate(label,
                 (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha="center")
graphic.save_graph("figures/indeed_stats/","number_tech.pdf")

graphic = GraphicBase("Degree of the most linked company through time",
                      "",
                      "",
                      "degree of most linked company",
                      date_format=False)
graphic.ax.plot(arrtopc)
for x,y in zip(range(len(arrtopc)), arrtopc):
    label = key_comp[x].replace("_"," ")
    plt.annotate(label,
                 (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha="center")
graphic.save_graph("figures/indeed_stats/","number_company.pdf")


graphic = GraphicBase("Maximum number of job opening linking a company and technology through time",
                      "",
                      "",
                      "Number of job openings",
                      date_format=False)
graphic.ax.plot(iw_max)
graphic.save_graph("figures/indeed_stats/","iw_max.pdf")


graphic = GraphicBase("Maximum number of patents linking a company and technology through time",
                      "",
                      "",
                      "Number of patents",
                      date_format=False)
graphic.ax.plot(pw_max)
graphic.save_graph("figures/indeed_stats/","pw_max.pdf")


graphic = GraphicBase("Mean number of job openings linking a company and technology through time",
                      "",
                      "",
                      "Number of job openings",
                      date_format=False)
graphic.ax.plot(iw_mean)
graphic.save_graph("figures/indeed_stats/","iw_mean.pdf")


graphic = GraphicBase("Mean number of patents linking a company and technology through time",
                      "",
                      "",
                      "Number of patents",
                      date_format=False)
graphic.ax.plot(pw_mean)
graphic.save_graph("figures/indeed_stats/","pw_mean.pdf")

# for key in iw_dict.keys():
#     sns.displot(iw_dict[key], kind="kde")
#     plt.title("density of job openings number for " + key)
#     plt.savefig(dir+"job_opening/iw_hist"+key+".pdf",
#                 format = 'pdf',
#                 dpi = 1000,
#                 bbox_inches = 'tight')
#     plt.close()
#
#     sns.displot(pw_dict[key], kind="kde")
#     plt.title("density of patents number for " + key)
#     plt.savefig(dir+"patent/pw_hist" + key + ".pdf",
#                 format = 'pdf',
#                 dpi = 1000,
#                 bbox_inches = 'tight')
#     plt.close()

graphic = GraphicBase("New edges through time",
                      "",
                      "",
                      "Number of new edges",
                      date_format=False)
graphic.ax.plot(new_edge)
plt.xticks(range(0,len(del_edge), 6),["04-2018","10-2018","04-2019","10-2019","04-2020","10-2020"])
graphic.save_graph("figures/indeed_stats/","new_edge.pdf")


graphic = GraphicBase("Edge disappearance through time",
                      "",
                      "",
                      "Number of edge disapearing",
                      date_format=False)
graphic.ax.plot(del_edge)
plt.xticks(range(0,len(del_edge), 6),["04-2018","10-2018","04-2019","10-2019","04-2020","10-2020"])
graphic.save_graph("figures/indeed_stats/","del_edge.pdf")


graphic = GraphicBase("Number of edges that survived from one month to another",
                      "",
                      "",
                      "Number of edges",
                      date_format=False)
graphic.ax.plot(same_edge)
graphic.save_graph("figures/indeed_stats/","same_edge.pdf")

graphic = GraphicBase("percentage of edge that appeared or dissapeared",
                      "",
                      "",
                      "\% of edges",
                      date_format=False)
graphic.ax.plot(perc_edge)
plt.xticks(range(0,len(del_edge), 6),["04-2018","10-2018","04-2019","10-2019","04-2020","10-2020"])
graphic.save_graph("figures/indeed_stats/","perc_edge.pdf")
