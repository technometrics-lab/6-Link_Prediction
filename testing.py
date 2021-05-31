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
dir = "final_graph/graph*"
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
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        #read each graph
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        #get number of edges for each tim
        arrdeg.append(g.size())
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

dir="figures/"
plt.plot(arrdeg)
plt.ylabel("Number of edges")
plt.title("evolution of the number of edges in time")

plt.savefig(dir+"number_edges.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()

plt.plot(arrcomp)
plt.ylabel("Size of largest connected component")
plt.title("evolution of the size of the largest connected component in time")

plt.savefig(dir+"number_conncomp.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()


plt.plot(arrtop)
plt.ylabel("degree of most linked technology")
plt.title("evolution of the degree of the most linked technology")

plt.savefig(dir+"number_tech.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()


plt.plot(arrtopc)
plt.ylabel("degree of most linked company")
plt.title("evolution of the degree of the most linked company")

plt.savefig(dir+"number_company.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()

plt.plot(iw_max)
plt.ylabel("number of job openings")
plt.title("Maximum number of job openings linking a company and technology by month")

plt.savefig(dir+"iw_max.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()

plt.plot(pw_max)
plt.ylabel("number of patents")
plt.title("Maximum number of patents linking a company and technology by month")

plt.savefig(dir+"pw_max.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()

plt.plot(iw_mean)
plt.ylabel("number of job openings")
plt.title("Mean number of patents linking a company and technology by month")

plt.savefig(dir+"iw_mean.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()

plt.plot(pw_mean)
plt.ylabel("number of patents")
plt.title("Mean number of patents linking a company and technology by month")

plt.savefig(dir+"pw_mean.pdf",
            format = 'pdf',
            dpi = 1000,
            bbox_inches = 'tight')
plt.close()



for key in iw_dict.keys():
    sns.displot(iw_dict[key], kind="kde")
    plt.title("density of job openings number for " + key)
    plt.savefig(dir+"job_opening/iw_hist"+key+".pdf",
                format = 'pdf',
                dpi = 1000,
                bbox_inches = 'tight')
    plt.close()
    sns.displot(pw_dict[key], kind="kde")
    plt.title("density of patents number for " + key)
    plt.savefig(dir+"patent/pw_hist" + key + ".pdf",
                format = 'pdf',
                dpi = 1000,
                bbox_inches = 'tight')
    plt.close()
