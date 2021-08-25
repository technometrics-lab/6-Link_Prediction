# loads graph to do quick manip on them
import json
import os
import glob
import random
import math
from statistics import mean
import networkx as nx
from tqdm  import tqdm
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.readwrite import json_graph
import numpy as np
from graphic_base import GraphicBase


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
