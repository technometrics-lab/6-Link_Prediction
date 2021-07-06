# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:55:58 2021

@author: antiago
"""


import json
import os
import argparse
import networkx as nx
from tqdm  import tqdm
import glob
from networkx.readwrite import json_graph
import pickle
from multiprocessing import Process


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


def prepare_data(args,cutoff=1000):
    company_id_list = list()
    # url = "http://dbpedia.org/data/{}.jsod"
    company_path = args.company
    indeed_path = args.indeed
    patentsview_path = args.patentsview

    indeed_annotaion_dict = dict()
    for result_path in glob.glob(
            indeed_path + "/annotation/**/part*",
            recursive=True):
        N = 0
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                N += 1
                if N > cutoff:
                    break
                json_line = json.loads(line)
                indeed_id = json_line['uid']
                if indeed_id not in indeed_annotaion_dict:
                    indeed_annotaion_dict[indeed_id] = []
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    for anno in annotations:
                        term = anno['uri'].split("/")[-1].lower()
                        if term in tech_list:
                            indeed_annotaion_dict[indeed_id].append(term)
                else:
                    try:
                        term = json_line['uri'].split("/")[-1].lower()
                        if term in tech_list:
                            indeed_annotaion_dict[indeed_id].append(term)
                    except:
                        term = json_line['technology'].split("/")[-1].lower()
                        if term in tech_list:
                            indeed_annotaion_dict[indeed_id].append(term)



    indeed_company_dict = dict()
    for result_path in glob.glob(
            indeed_path+"/linked/dataset.json/**/part*",
            recursive=True):
        N = 0
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                N += 1
                if N>cutoff:
                    break
                json_line = json.loads(line)
                indeed_id = json_line['id']
                if indeed_id not in indeed_company_dict:
                    indeed_company_dict[indeed_id] = []
                company_id = json_line['uid']
                indeed_company_dict[indeed_id].append(company_id)
                if company_id not in company_id_list:
                    company_id_list.append(company_id)



    # patent_annotaion_dict = dict()
    # for result_path in glob.glob(
    #         patentsview_path + "/annotation/**/part*",
    #         recursive=True):
    #     N=0
    #     try:
    #         with open(result_path, 'r', encoding='utf-8') as file:
    #             for line in tqdm(file):
    #                 N+=1
    #                 if N>cutoff:
    #                     break
    #                 json_line = json.loads(line)
    #                 indeed_id = json_line['uid']
    #                 if indeed_id not in patent_annotaion_dict:
    #                     patent_annotaion_dict[indeed_id] = []
    #                 if 'annotations' in json_line:
    #                     annotations = json_line['annotations']
    #                     for anno in annotations:
    #                         term = anno['uri'].split("/")[-1].lower()
    #                         if term in tech_list:
    #                             patent_annotaion_dict[indeed_id].append(term)
    #                 else:
    #                     try:
    #                         term = json_line['uri'].split("/")[-1].lower()
    #                         if term in tech_list:
    #                             patent_annotaion_dict[indeed_id].append(term)
    #                     except:
    #                         term = json_line['technologies'].split("/")[-1].lower()
    #                         if term in tech_list:
    #                             patent_annotaion_dict[indeed_id].append(term)
    #     except:
    #         continue
    #
    # patent_company_dict = dict()
    # for result_path in glob.glob(
    #         patentsview_path +"/linked/dataset.json/**/part*",
    #         recursive=True):
    #     N=0
    #     try:
    #         with open(result_path, 'r', encoding='utf-8') as file:
    #             for line in tqdm(file):
    #                 N += 1
    #                 if N>cutoff:
    #                     break
    #                 json_line = json.loads(line)
    #                 indeed_id = json_line['patent_id']
    #                 if indeed_id not in patent_company_dict:
    #                     patent_company_dict[indeed_id] = []
    #                 company_id = json_line['uid']
    #                 patent_company_dict[indeed_id].append(company_id)
    #                 if company_id not in company_id_list:
    #                     company_id_list.append(company_id)
        except:
            continue

    tech_comp_indeed=[]
    for key in indeed_annotaion_dict.keys() & indeed_company_dict.keys():
        for company in indeed_company_dict[key]:
            if not indeed_annotaion_dict[key]==[]:
                tech_comp_indeed.append((company,indeed_annotaion_dict[key]))

    # tech_comp_patent=[]
    # for key in patent_annotaion_dict.keys() & patent_company_dict.keys():
    #     for company in patent_company_dict[key]:
    #         if not patent_annotaion_dict[key]==[]:
    #             tech_comp_patent.append((company,patent_annotaion_dict[key]))


    return company_id_list, tech_comp_indeed, tech_comp_patent

def make_graph(company_id_list,tech_comp_indeed, tech_comp_patent):

    G = nx.Graph()
    bi_dict={}
    for item in tqdm(tech_comp_indeed):
        n=tech_comp_indeed.count(item)
        bi_dict[item[0]]=0
        bi_dict[item[1]]=1
        G.add_edge(item[0],item[1],iw=n,pw=0)

    # for item in tqdm(tech_comp_patent):
    #     n=tech_comp_patent.count(item)
    #     bi_dict[item[0]]=0
    #     bi_dict[item[1]]=1
    #     if item[0] in list(G):
    #         if item[0] in G.neighbors(item[1]):
    #             G.add_edge(item[0], item[1], pw=n)
    #         else:
    #             G.add_edge(item[0], item[1], pw=n)
    #     else:
    #         G.add_edge(item[0], item[1], pw=n)

    nx.set_node_attributes(G,bi_dict,"bipartite")

    return G


def graph_maker(path,cutoff):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--company',
                        default="Z:/sel_data/"+path+"/company.json/",
                        type=str,
                        help='path to folder of spacy-indeed result')
    parser.add_argument('-i', '--indeed',
                        default="Z:/sel_data/"+path+"/indeed/",
                        type=str,
                        help='path to folder of spacy-patent result')
    parser.add_argument('-m', '--mag',
                        default="Z:/sel_data/"+path+"/mag/",
                        type=str,
                        help='path to folder of spacy-register result')
    parser.add_argument('-pv', '--patentsview',
                        default="Z:/sel_data/"+path+"/patentsview/",
                        type=str,
                        help='path to folder of indeed result')
    parser.add_argument('-od', '--output_data',
                        default='DBpedia',
                        type=str,
                        help='output path ',
                        )
    parser.add_argument('-o', '--output_path',
                        default='result_pale',
                        type=str,
                        help='output path ',
                        )
    args = parser.parse_args()

    company_id_list,tech_comp_indeed, tech_comp_patent  = prepare_data(args,cutoff)

    G = make_graph(company_id_list, tech_comp_indeed, tech_comp_patent)
    G.name=path[0:6]

    res = json_graph.node_link_data(G)
    dir = "graphsindeed/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    dataset="graph"+path[0:6]


    with open(dir + dataset + ".json", 'w') as outfile:
        json.dump(res, outfile)


#class of multiproccessing
class Graph(Process):

    def __init__(self,path):
        #init
        Process.__init__(self)
        self.path=path

    def run(self):
        print("start")
        cutoff=500
        graph_maker(self.path,cutoff)
        print("stop")



if __name__=='__main__':
    path_list=["20180301T000000","20180401T000000","20180501T000000",
               "20180601T000000","20180701T000000","20180801T000000","20180901T000000",
               "20181001T000000","20181101T000000","20181201T000000"]

    graph_list=[]
    for path in path_list[0:1]:
        graph=Graph(path)
        graph_list.append(graph)
        graph.start()
    for graph in graph_list:
        graph.join()
