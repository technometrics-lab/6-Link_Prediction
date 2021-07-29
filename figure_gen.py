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
tech_list=["cloud_computing","cryptography","quantum_computing","trusted_computing","internet_of_things",
         "intrusion_detection_system","digital_signature","electronic_signature","machine_learning",
         "feature_(machine_learning)","deep_learning","linear_predictive_analysis","automated_machine_learning",
         "file_sharing","computer_vision","5g","penetration_test","superminicomputer","authentic_learning","network_model",
         "visual_cryptography","minisupercomputer","modular_neural_network","overlay_network","padding_(cryptography)",
         "security_token","distributed_artificial_intelligence","id-based_cryptography","speech_analytics",
         "salt_(cryptography)","adversarial_machine_learning","message_authentication","technical_intelligence",
         "omega_network","pre-boot_authentication","email_privacy","data_analysis","explainable_artificial_intelligence",
         "distributed_ledger","ipfirewall","authentication_server","mobile_virtual_private_network",
         "quantum_machine","contact_scraping","blockchain","time-based_authentication","data_security",
         "statistical_relational_learning","wi-fi","reinforcement_learning","authenticated_encryption","security_log",
         "network_monitoring","titan_(supercomputer)","distributed_algorithm","electronic_authentication",
         "smtp_authentication","open_security","spiking_neural_network","feature_learning",
         "strategic_intelligence","attack_(computing)","bluetooth","strong_cryptography","form-based_authentication",
         "semantic_computing","quantum_neural_network","deep_linking","code_(cryptography)","deception_technology",
         "deep_content_inspection","proxy_server","information_security_management","nanocomputer","trusted_path",
         "deep_packet_inspection","high-performance_technical_computing","key_(cryptography)","trusted_platform_module",
         "security_modes","network_mapping","security_level","private_network","quantum_cryptography","financial_cryptography",
         "smtps","authentication","artificial_intelligence_system","anticipation_(artificial_intelligence)",
         "security_bug","mutual_authentication","anchor_modeling","high-throughput_computing","internetworking",
         "pmac_(cryptography)","https","email_authentication","closed-loop_authentication","packet_injection","hru_(security)",
         "one-way_quantum_computer","circuit-level_gateway","virtual_private_server","quantum_artificial_intelligence_lab",
         "neural_cryptography","strong_authentication","artificial_intelligence","risk-based_authentication","tsmp",
         "reliance_authentication","ccmp_(cryptography)","virtual_private_network","ntfs","eager_learning","supercomputer",
         "key_authentication","predictive_informatics","federated_search","encryption","predictive_analytics",
         "cryptovirology","hybrid_neural_network","data_fusion","mqtt"]

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
perc_comp = []
g_old = 0
techdeg = {}
for t in tech_list:
    techdeg[t] = []
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
        perc_comp.append(largest_comp/len(list(g)))

        #get most linked technology
        tech=[n for n in list(g) if g.nodes[n]["bipartite"]==1]
        deg_tech=dict(g.degree(tech))
        sort_tech=sorted(deg_tech.values(), reverse=True)
        arrtop.append(sort_tech[1])
        key_tech.append(get_key(deg_tech, sort_tech[0]))
        # get degree evolution for each technology some will only be zero all the way
        # but i am too lazy to adapt it
        for t in tech:
            techdeg[t].append(g.degree(t))

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

        g_old=g

# basically we plot everything
# sometimes added good xticks for time but not on the graphs i deemed unnecessary
dir="figures/"
graphic = GraphicBase("Number of edges through time",
                      "",
                      "",
                      "Number of edges",
                      date_format=False)
graphic.ax.plot(arrdeg)
plt.xticks(range(0,len(arrdeg), 6),["03-2018","09-2018","03-2019","09-2019","03-2020","09-2020"])
graphic.save_graph("figures/","number_edges.pdf")


graphic = GraphicBase("Size of the largest connected component through time",
                      "",
                      "",
                      "Size of largest connected component",
                      date_format=False)
graphic.ax.plot(arrcomp)
plt.xticks(range(0,len(arrcomp), 6),["03-2018","09-2018","03-2019","09-2019","03-2020","09-2020"])
graphic.save_graph("figures/","number_conncomp.pdf")

graphic = GraphicBase("size percentage of the largest connected component through time",
                      "",
                      "",
                      "% largest connected component",
                      date_format=False)
graphic.ax.plot(perc_comp)
plt.xticks(range(0,len(arrcomp), 6),["03-2018","09-2018","03-2019","09-2019","03-2020","09-2020"])
graphic.save_graph("figures/","perc_conncomp.pdf")


graphic = GraphicBase("Degree of the most linked technology through time",
                      "",
                      "",
                      "degree of most linked technology",
                      date_format=False)
graphic.ax.plot(range(len(arrtop)),arrtop)
plt.xticks(range(0,len(arrtop), 6),["03-2018","09-2018","03-2019","09-2019","03-2020","09-2020"])
for x,y in zip(range(len(arrtop)), arrtop):
    label = key_tech[x].replace("_"," ")
    plt.annotate(label,
                 (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 fontsize=20,
                 ha="center")
graphic.save_graph("figures/","number_tech.pdf")

graphic = GraphicBase("Degree of the most linked company through time",
                      "",
                      "",
                      "degree of most linked company",
                      date_format=False)
graphic.ax.plot(arrtopc)
plt.xticks(range(0,len(arrtopc), 6),["03-2018","09-2018","03-2019","09-2019","03-2020","09-2020"])
for x,y in zip(range(len(arrtopc)), arrtopc):
    label = key_comp[x].replace("_"," ")
    plt.annotate(label,
                 (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 rotation = 45,
                 ha="center")
graphic.save_graph("figures/","number_company.pdf")


graphic = GraphicBase("Maximum number of job opening linking a company and technology through time",
                      "",
                      "",
                      "Number of job openings",
                      date_format=False)
graphic.ax.plot(iw_max)
graphic.save_graph("figures/","iw_max.pdf")


graphic = GraphicBase("Maximum number of patents linking a company and technology through time",
                      "",
                      "",
                      "Number of patents",
                      date_format=False)
graphic.ax.plot(pw_max)
graphic.save_graph("figures/","pw_max.pdf")


graphic = GraphicBase("Mean number of job openings linking a company and technology through time",
                      "",
                      "",
                      "Number of job openings",
                      date_format=False)
graphic.ax.plot(iw_mean)
graphic.save_graph("figures/","iw_mean.pdf")


graphic = GraphicBase("Mean number of patents linking a company and technology through time",
                      "",
                      "",
                      "Number of patents",
                      date_format=False)
graphic.ax.plot(pw_mean)
graphic.save_graph("figures/","pw_mean.pdf")



graphic = GraphicBase("New edges through time",
                      "",
                      "",
                      "Number of new edges",
                      date_format=False)
graphic.ax.plot(new_edge)
plt.xticks(range(0,len(del_edge), 6),["04-2018","10-2018","04-2019","10-2019","04-2020","10-2020"])
graphic.save_graph("figures/","new_edge.pdf")


graphic = GraphicBase("Edge disappearance through time",
                      "",
                      "",
                      "Number of edge disapearing",
                      date_format=False)
graphic.ax.plot(del_edge)
plt.xticks(range(0,len(del_edge), 6),["04-2018","10-2018","04-2019","10-2019","04-2020","10-2020"])
graphic.save_graph("figures/","del_edge.pdf")


graphic = GraphicBase("Number of edges that survived from one month to another",
                      "",
                      "",
                      "Number of edges",
                      date_format=False)
graphic.ax.plot(same_edge)
graphic.save_graph("figures/","same_edge.pdf")

graphic = GraphicBase("percentage of edge that appeared or dissapeared",
                      "",
                      "",
                      "\% of edges",
                      date_format=False)
graphic.ax.plot(perc_edge)
plt.xticks(range(0,len(del_edge), 6),["04-2018","10-2018","04-2019","10-2019","04-2020","10-2020"])
graphic.save_graph("figures/","perc_edge.pdf")

# for key, arr in techdeg.items():
#     graphic = GraphicBase("Degree evolution of "+key.replace("_"," "),
#                           "",
#                           "",
#                           "Degree",
#                           date_format=False)
#     graphic.ax.plot(arr)
#     plt.xticks(range(0,len(del_edge), 6),["03-2018","09-2018","03-2019","09-2019","03-2020","09-2020"])
#     graphic.save_graph("figures/","degevo"+key.replace("_"," ")+".pdf")
