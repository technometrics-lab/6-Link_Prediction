# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:55:58 2021

@author: Santiago
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

#global list of technologies we consider
tech_list = ["cloud_computing","cryptography","quantum_computing",
           "trusted_computing","internet_of_things","intrusion_detection_system"
           ,"digital_signature","electronic_signature","machine_learning",
           "feature_(machine_learning)","deep_learning","linear_predictive_analysis","automated_machine_learning",
           "file_sharing","computer_vision","5g","penetration_test"]


#Input: args which is a parser of the path to TMM data, cutoff just to have a 
#limit of line analysed for quick computations
#Output: List of companies, dict of technologies: companies
def prepare_data(args, cutoff = 1000):
    
    #setting up the paths
    tech_comp_dict=dict()
    company_id_list = list()
    company_path = args.company
    indeed_path = args.indeed
    patentsview_path = args.patentsview
    
    #opening all the files in company.json to get all companies
    for result_path in glob.glob(
            company_path + "/part*"):
        N=0
        with open(result_path, 'r', encoding='utf-8') as file:
            #read line per line the file to extract companies
            for line in tqdm(file):
                #cutoff limit of iteration
                N+=1
                if N>cutoff:
                    break
                
                json_line = json.loads(line) 
                #retrieving the company
                company_id = json_line['uid']
                company_id_list.append(company_id)
                
                
    #opening all the files in indeed to get a dict indeed_id:company
    indeed_company_dict = dict()
    for result_path in glob.glob(
            indeed_path + "/linked/dataset.json/**/part*",
            recursive = True):
        
        N = 0
        with open(result_path, 'r', encoding = 'utf-8') as file:
            #Read line per line to extract indeed id: company
            for line in tqdm(file):
                #cutoff limits the number of iterations
                N += 1
                if N > cutoff:
                    break
                #get indeed id 
                json_line = json.loads(line)
                indeed_id = json_line['id']
                #if not already present in dict add it to keys
                if indeed_id not in indeed_company_dict:
                    indeed_company_dict[indeed_id] = []
                #get company id
                company_id = json_line['uid']
                indeed_company_dict[indeed_id].append(company_id)
                #if it was not captured by the company search add it to the list
                if company_id not in company_id_list:
                    company_id_list.append(company_id)
    
    #opening all the files in indeed to get a dict indeed_id:technology
    indeed_annotaion_dict = dict()
    for result_path in glob.glob(
            indeed_path + "/annotation/**/part*",
            recursive = True):
        
        N = 0
        with open(result_path, 'r', encoding = 'utf-8') as file:
            for line in tqdm(file):
                #cutoff limit of iteration
                N += 1
                if N > cutoff:
                    break
                
                #retrieve indeed id
                json_line = json.loads(line)
                indeed_id = json_line['uid']
                
                #check if the id is already in dict keys otherwise add it
                if indeed_id not in indeed_annotaion_dict:
                    indeed_annotaion_dict[indeed_id] = []
                    
                #if annotation then this indeed id is linked to multiple tech 
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    #separating each individual tech
                    for anno in annotations:
                        #separating the tech from dbpedia formating
                        term = anno['uri'].split("/")[-1].lower()
                        #checking if tech is in our list to consider it 
                        if term in tech_list:
                            indeed_annotaion_dict[indeed_id].append(term)
                # else it is linked to only one technology
                else:
                    #separating the tech from dbpedia formating
                    term = json_line['uri'].split("/")[-1].lower()
                    #checking if tech is in our list to consider it
                    if term in tech_list:
                        indeed_annotaion_dict[indeed_id].append(term)


    
            
    #opening all the files in patentsview to get patentid: tech dict
    patent_annotaion_dict = dict()
    for result_path in glob.glob(
            patentsview_path + "/annotation/**/part*",
            recursive=True):
        
        N = 0
        with open(result_path, 'r', encoding = 'utf-8') as file:
            
            #read line per line to get one instanc of patentid: [tech1,tech2] or :tech
            for line in tqdm(file):
                #cutoff limit of iterations
                N += 1
                if N > cutoff:
                    break
                
                #get patent id 
                json_line = json.loads(line)
                patent_id = json_line['uid']
                
                # if not already in dict add it to keys
                if patent_id not in patent_annotaion_dict:
                    patent_annotaion_dict[patent_id] = []
                #if annotation then this patent id is linked to multiple tech
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    #separate the different tech                    
                    for anno in annotations:
                        #separating the tech from dpbedia formating
                        term = anno['uri'].split("/")[-1].lower()
                        #add it only if it is in techlist
                        if term in tech_list:
                            patent_annotaion_dict[patent_id].append(term)
                #else it is linked to only one tech
                else:
                    #separating tech from dbpedia formating
                    term = json_line['uri'].split("/")[-1].lower()
                    #add it only if it is in tech_list
                    if term in tech_list:
                        patent_annotaion_dict[patent_id].append(term)
    
    #opening all the files in Patentsview to get a dict patent_id:company
    patent_company_dict = dict()
    for result_path in glob.glob(
            patentsview_path + "/linked/dataset.json/**/part*",
            recursive=True):
        
        N = 0
        with open(result_path, 'r', encoding = 'utf-8') as file:
            #read line per line the file to extract companies
            for line in tqdm(file):
                #cutoff limits the number of iterations
                N += 1
                if N > cutoff:
                    break
                #get patent id 
                json_line = json.loads(line)
                indeed_id = json_line['patent_id']
                #if not already present in dict add it to keys
                if indeed_id not in patent_company_dict:
                    patent_company_dict[indeed_id] = []
                #get company id
                company_id = json_line['uid']
                patent_company_dict[indeed_id].append(company_id)
                #if it was not captured by the company search add it to the lis
                if company_id not in company_id_list:
                    company_id_list.append(company_id)
    
    #Create the company:[tech1,tech2,...] dict from the indeed and patent dicts
    tech_comp_dict = dict()
    #Start with indeed that share the same id 
    for key in indeed_annotaion_dict.keys() & indeed_company_dict.keys():
        #scroll trough companies for that indeed id
        for company in indeed_company_dict[key]:
            #if not already in keys add it
            if company not in tech_comp_dict:
                tech_comp_dict[company] = []
            #if the tech is not just empty tech due to TMM data then add it
            if not indeed_annotaion_dict[key] == []:
                tech_comp_dict[company].append(indeed_annotaion_dict[key])
    
    #we do the same for patents
    for key in patent_annotaion_dict.keys() & patent_company_dict.keys():
        for company in patent_company_dict[key]:
            if company not in tech_comp_dict:
                tech_comp_dict[company] = []
            if not patent_annotaion_dict[key] == []:
                tech_comp_dict[company].append(patent_annotaion_dict[key])
                

    return company_id_list, tech_comp_dict

#Input: company list and company:tech dict from prepare_data(...)
#Output: Bipartite NetworkX graph of company:technology
def make_graph(company_id_list, tech_comp_dict):
    #initialize graph and graphs attributes as well as mapping company:node_id
    G = nx.Graph()
    node_id = 0
    company_type, term_type  = list(range(2))
    company_map, term_map = {}, {}
    G.graph["partition"] = []
    
    #add a node for each company and update mapping of company to node
    for company in tqdm(company_id_list):
        G.add_node(node_id, feature=[company_type], label=[company_type],\
                   content=[company] ,bipartite=0)
        company_map[company] = node_id
        node_id += 1
        
    #set the partition of company    
    G.graph["partition"].append(list(range(0, node_id)))
    id_comp = node_id
    
    #add each technology from tech_list
    for term in tqdm(tech_list):
        G.add_node(node_id, feature=[term_type], label=[term_type], content=[term], bipartite=1)
        term_map[term] = node_id
        node_id += 1
    #set the partition of technologies    
    G.graph["partition"].append(list(range(id_comp+1, node_id)))
    
    #add an edge between each comp and tech in comp:[tech1,tech2] dict
    for key,item in tech_comp_dict.items():
        for term in item:
            G.add_edge(company_map[key], term_map[term[0]])

    return G


#Input: path to TMM time related file, cutoff number of max iterations
#output: None saves graphs in graphs/graph##.json
def graph_maker(path, cutoff):
    
    #parser to give to prepare data with path to the different data files
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
    
    #compute company lists and relation dict between company and technologies
    company_id_list, tech_comp_dict  = prepare_data(args, cutoff)

    G = make_graph(company_id_list, tech_comp_dict)
    
    #save the graph
    res = json_graph.node_link_data(G)
    dir = "graphs/"
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    dataset = "graph"+path[4:6]

    with open(dir + dataset + ".json", 'w') as outfile:
        json.dump(res, outfile)


#graph class for computing several graph at the same time with multiprocessing
class Graph(Process):
    
    def __init__(self,path):
        Process.__init__(self)
        #path that caracterizes one graph
        self.path = path
    
    def run(self):
        
        print("start")
        cutoff = 1000000000
        graph_maker(self.path, cutoff)
        print("stop")
        


if __name__=='__main__':
    path_list = ["20200101T000000", "20200201T000000", "20200301T000000", "20200401T000000",
               "20200501T000000", "20200601T000000", "20200701T000000", "20200801T000000",
               "20200901T000000", "20201001T000000", "20201101T000000", "20201201T000000"]
    graph_list = []
    for path in path_list:
        graph = Graph(path)
        graph_list.append(graph)
        graph.start()
    for graph in graph_list:
        graph.join()
