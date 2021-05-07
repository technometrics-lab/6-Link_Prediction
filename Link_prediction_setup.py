from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import expm
from sklearn import svm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from statsmodels.tsa.arima.model import ARIMA
import time
from scipy.sparse.linalg import inv
from scipy.sparse import identity
import warnings
from sklearn.svm import LinearSVC
import scipy
from tqdm  import tqdm
from multiprocessing import Process, Array


warnings.filterwarnings("ignore")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid = False):

    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])

        
    # Store negative edge predictions, actual values
    preds_neg = []

    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score, roc_curve_tuple

#Input:ROC curve from scikit roc_curve() function and root bool to know which of the two method to use
#Output: Index of the optimal threshold in roccurve and value of optimal threshold
def gmeans(roc_curve,root = False):
    
    if root:
        #gmeans method found in literature
        g = np.sqrt(roc_curve[0] * (1-roc_curve[1]))
    else:
        #basic method
        g = roc_curve[1]-roc_curve[0]

    ind = np.argmax(g)
    threshold = roc_curve[2][ind]
    return ind, threshold

#Input:Netowrkx training graph, damping factor alpha
#Output: Matrix of scores
def sinh_scores(g_train, alpha = 1):
    adj_train = nx.to_scipy_sparse_matrix(g_train)
    sh_scores = {}
    sinh_mat = (expm(adj_train)-expm(-adj_train))/2
    sinh_mat = sinh_mat/sinh_mat.max()
    sinh_mat = sinh_mat.todense()
    sh_scores["mat"] = sinh_mat
    return sh_scores

# Input: NetworkX training graph
# Output: Score matrix
def preferential_attachment_scores(g_train):
    adj_train = nx.to_scipy_sparse_matrix(g_train)
    pa_scores = {}
    # Calculate scores
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train):# (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p # make sure it's symmetric
    pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix
    pa_scores["mat"] = pa_matrix
    return pa_scores

#Input:Networkx training graph,max_power
#Output: Matrix scores
def katz_scores(g_train, max_power = 5, beta = 0.001):
    adj_train = nx.to_scipy_sparse_matrix(g_train)
    ka_scores = {}
    ka_score_matrix = (inv(identity(adj_train.shape[1])-beta*adj_train)-identity(adj_train.shape[1])).todense()
    ka_scores["mat"] = ka_score_matrix
    return ka_scores

#Input:edge list to train on,edge list to test on, scores matrix of other metrics
#Output: mmmmh not sure yet
def SVM_score(test_split1, test_split2, ka_scores, pa_scores, sh_scores):
    
    train_pos, train_neg, train_all = test_split1
    test_pos, test_neg, test_all = test_split2
        
    #create feature vectore for train and test
    att_train_pos = create_attributes(train_pos, ka_scores, pa_scores, sh_scores)
    att_train_neg = create_attributes(train_neg, ka_scores, pa_scores, sh_scores)
    att_test_pos = create_attributes(test_pos, ka_scores, pa_scores, sh_scores)
    att_test_neg = create_attributes(test_neg, ka_scores, pa_scores, sh_scores)
    
    #Train SVM
    preds_all = np.vstack([att_train_pos, att_train_neg])
    labels_all = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    clf = LinearSVC()
    clf.fit(preds_all, labels_all)
    
    #Test SVM
    preds_test_all = np.vstack([att_test_pos, att_test_neg])
    labels_test = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    return clf.decision_function(preds_test_all), labels_test

#Input:sparse matrix
#Output:flat array
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

#Input:Networkx graph with partition attribute otherwise it wont work,adjacency matrix of the graph
#Output:Present edge list, non present edge list, all possible edge list 
def bipartite_data_edge(G,adj):
    
    #get upper triangular of adj since it contains all possible edges(even more than all possibles in bipartite)
    adj_triu = sp.triu(adj)
    edges_tuple = sparse_to_tuple(adj_triu)
    edges = edges_tuple[0]

    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    set_edge = set(edge_tuples)
    #create negative edges list
    #don't include edges that are between node in same groups
    false_edge = set()
    all_pos_edge = []
    for x in G.graph["partition"][0]:
        for y in G.graph["partition"][1]:
            false = (min(x, y), max(x, y))
            all_pos_edge.append(false)
            if false in set_edge:
                continue
            else:
                false_edge.add(false)
                
    edge_neg = np.array([list(edge_tuple) for edge_tuple in false_edge])
    
    return np.array(edge_tuples), edge_neg, all_pos_edge

#Input:edge list to create feature vector for, scores matrices
#Output: feature flat array for SVM
def create_attributes(edge_list, ka_scores, pa_scores, sh_scores):
    attr = np.zeros((edge_list.shape[0], 3))
    n = 0
    for edge in edge_list:

        attr[n][0] = ka_scores[edge[0], edge[1]]
        attr[n][2] = sh_scores[edge[0], edge[1]]
        attr[n][1] = pa_scores[edge[0], edge[1]]
        n = n + 1

    return attr

#Input: array of graphs
#output:dict of performance results
def calculate_time_score(arr):
    ka_mat = []
    pa_mat = []
    sh_mat = []
    uns_res = {}
    
    for n in range(len(arr)-1):
        g0 = arr[n]
        
        pa_scores = preferential_attachment_scores(g0)["mat"]
        ka_scores = katz_scores(g0)["mat"]
        sh_scores = sinh_scores(g0)["mat"]
        
        ka_mat.append(mat_to_arr(ka_scores))
        pa_mat.append(mat_to_arr(pa_scores))
        sh_mat.append(mat_to_arr(sh_scores))
    
    true_ka = np.array(ka_mat).reshape((len(arr)-1, ka_mat[0].shape[1]))
    true_pa = np.array(pa_mat).reshape((len(arr)-1, pa_mat[0].shape[1]))
    true_sh = np.array(sh_mat).reshape((len(arr)-1, sh_mat[0].shape[1]))
    
    train = bipartite_data_edge(arr[-2], nx.to_scipy_sparse_matrix(arr[-2]))
    test_pos, test_neg, all_edge = bipartite_data_edge(arr[-1], nx.to_scipy_sparse_matrix(arr[-1]))
    
    pred_ka = time_series_predict(true_ka).reshape(ka_scores.shape)
    pred_pa = time_series_predict(true_pa).reshape(pa_scores.shape)
    pred_sh = time_series_predict(true_sh).reshape(sh_scores.shape)
    pred_svm, labels_svm = SVM_score(train, [test_pos, test_neg, all_edge], pred_ka, pred_pa, pred_sh)
    
    uns_res["ka"] = get_roc_score(test_pos,test_neg, pred_ka)
    uns_res["pa"] = get_roc_score(test_pos,test_neg, pred_pa)
    uns_res["sh"] = get_roc_score(test_pos,test_neg, pred_sh)
    uns_res["svm"] = roc_auc_score(labels_svm,pred_svm), average_precision_score(labels_svm,pred_svm), roc_curve(labels_svm,pred_svm)
    
    return uns_res


#Input: scores matrix
#Output:score matrix flattened like i want
def mat_to_arr(mat):
    A = np.matrix.flatten(mat)
    if A.shape[0] > 1:
        A = A.reshape((1, A.shape[0]))
    return A

#Input:array of flattened metrics score
#Output: predicted one step ahead metrics score
def time_series_predict(arr):
    predicted = []
    for n in tqdm(range(arr.shape[1])):
        predicted.append(ARIMA(arr[:, n].T, order=(1,0,0)).fit().forecast())
    return np.asarray(predicted)


def result_formater(res):
    plt.figure()
    plt.plot([0,1],[0,1],"g--")
    label={"ka":"Katz Index AUC = ","pa":"Preferential Attachment Index AUC = ",
           "sh":"Hyperbolic Sine Index AUC = ","svm":"SVM AUC = "}
    for key in res.keys():
        test_fpr, test_tpr, threshold = res[key][2]
        plt.plot(test_fpr, test_tpr, label = label[key] + str(round(res[key][0],3)))
    
    plt.legend(loc = "lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
    
    
    
arr=[]
for n in range(1,12):
    arr.append(nx.random_partition_graph((10,20),0,n/24))
    
res=calculate_time_score(arr)
result_formater(res)