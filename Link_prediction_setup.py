import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import json
import time
import glob
import random
from sklearn.linear_model import LinearRegression
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm
from networkx.readwrite import json_graph
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score
from statsmodels.tsa.arima.model import ARIMA
from scipy.sparse.linalg import inv
from scipy.sparse import identity
import warnings
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from tqdm  import tqdm
from graphic_base import GraphicBase


warnings.filterwarnings("ignore")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid = False):

    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions
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

    acc = threshold_prediction(preds_all, ["","1", roc_curve_tuple], labels_all)
    return roc_score, ap_score, roc_curve_tuple, acc


#Input:ROC curve from scikit roc_curve() function and root bool to know which of the two method to use
#Output: Index (labels_all, preds_all)of the optimal threshold in roccurve and value of optimal threshold
def gmeans(roc_curve,root = False):

    if root:
        #gmeans method found in literature
        g = np.sqrt(roc_curve[0] * (1 - roc_curve[1]))
    else:
        #basic method
        g = roc_curve[1] - roc_curve[0]

    #find index of optimal threshold, then the threshold
    ind = np.argmax(g)
    threshold = roc_curve[2][ind]
    return ind, threshold

#Input:Netowrkx training graph, damping factor alpha
#Output: Matrix of scores
def sinh_scores(g_train, nodelist0, alpha = 0.01):
    #compute adjacency matrix
    adj_train = nx.to_numpy_matrix(g_train, nodelist=nodelist0)
    sinh_mat = (expm(alpha * adj_train) - expm(-alpha * adj_train))/2
    sinh_mat = sinh_mat/sinh_mat.max()
    return sinh_mat

# Input: NetworkX training graph
# Output: Score matrix
def preferential_attachment_scores(g_train, nodelist0):
    adj_train = nx.to_numpy_matrix(g_train, nodelist = nodelist0)
    #map of node label to integer index
    pos, neg, all = bipartite_data_edge(g_train, adj_train, nodelist0)
    mapping = nodelabel_to_index(nodelist0)
    ebunch1 = get_ebunch(all, nodelist0)
    pa_matrix = np.zeros(adj_train.shape)
    #compute scores for each edge
    for u, v, p in nx.preferential_attachment(g_train, ebunch1):# (u, v) = node indices, p = Jaccard coefficient
        i = mapping[u]
        j = mapping[v]
        pa_matrix[i][j] = p
        pa_matrix[j][i] = p
        # make sure it's symmetric
    if not pa_matrix.max() == 0:
        pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix

    return pa_matrix

#Input:Networkx training graph,max_power
#Output: Matrix scores
def katz_scores(g_train, nodelist0, beta =  0.006):
    adj_train = nx.to_numpy_matrix(g_train, nodelist=nodelist0)
    ka_score_matrix = np.linalg.inv(identity(adj_train.shape[1]) - beta * adj_train) - identity(adj_train.shape[1])
    return ka_score_matrix

#Input:edge list to train on,edge list to test on, scores matrix of other metrics
#Output: mmmmh not sure yet
def SVM_score(test_split1, test_split2, ka_scores, pa_scores, sh_scores, pw, iw, c = 0.06):

    train_pos, train_neg, train_all = test_split1
    test_pos, test_neg, test_all = test_split2

    #create feature vectore for train and test
    att_train_pos = create_attributes(train_pos, ka_scores[0], pa_scores[0], sh_scores[0], pw, iw)
    att_train_neg = create_attributes(train_neg, ka_scores[0], pa_scores[0], sh_scores[0], pw, iw)
    att_test_pos = create_attributes(test_pos, ka_scores[1], pa_scores[1], sh_scores[1], pw, iw)
    att_test_neg = create_attributes(test_neg, ka_scores[1], pa_scores[1], sh_scores[1], pw, iw)

    #Train SVM
    preds_all = np.vstack([att_train_pos, att_train_neg])
    labels_all = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    clf = SVC(C=c, gamma=0.00001)
    clf.fit(preds_all, labels_all)

    #Test SVM
    preds_test_all = np.vstack([att_test_pos, att_test_neg])
    labels_test = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    return clf.decision_function(preds_test_all), labels_test, accuracy_score(clf.predict(preds_test_all), labels_test)

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
def bipartite_data_edge(G, adj, nodelist0):

    adj = coo_matrix(adj)
    #get upper triangular of adj since it contains all possible edges(even more than all possibles in bipartite)
    adj_triu = sp.triu(adj)
    edges_tuple = sparse_to_tuple(adj_triu)
    edges = edges_tuple[0]
    mapping = nodelabel_to_index(nodelist0)
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]

    set_edge = set(edge_tuples)
    #create negative edges list
    #don't include edges that are between node in same groups
    false_edge = set()
    all_pos_edge = []
    for x in [x for x, y in G.nodes(data=True) if y["bipartite"] == 0]:
        for y in [x for x, y in G.nodes(data=True) if y["bipartite"] == 1]:
            i = mapping[x]
            j = mapping[y]
            false = (min(i, j), max(j, i))
            all_pos_edge.append(false)
            if false in set_edge:
                continue
            else:
                false_edge.add(false)

    edge_neg = [edge_tuple for edge_tuple in false_edge]

    return edge_tuples, random.sample(edge_neg,k=len(edge_tuples)), all_pos_edge

#Input:edge list to create feature vector for, scores matrices
#Output: feature flat array for SVM
def create_attributes(edge_list, ka_scores, pa_scores, sh_scores, pw, iw):
    attr = np.zeros((len(edge_list), 4))
    n = 0
    for edge in edge_list:
        attr[n][3] = ka_scores[edge[0], edge[1]]
        attr[n][1] = sh_scores[edge[0], edge[1]]
        attr[n][0] = pa_scores[edge[0], edge[1]]
        attr[n][2] = iw[edge[0], edge[1]]
        n = n + 1
    return attr

#Input: array of graphs
#output:dict of performance results
def calculate_time_score(arr, nodelist0, month_gap = 1):
    if len(arr) == 2:
        mat = {}
        uns_res = {}
        pa_scores = preferential_attachment_scores(arr[0], nodelist0)
        ka_scores = katz_scores(arr[0], nodelist0)
        sh_scores = sinh_scores(arr[0], nodelist0)
        test_pos, test_neg, all_edge = bipartite_data_edge(arr[-1], nx.to_numpy_matrix(arr[-1], nodelist0), nodelist0)
        uns_res["ka"] = get_roc_score(test_pos, test_neg, ka_scores)
        uns_res["pa"] = get_roc_score(test_pos,test_neg, pa_scores)
        uns_res["sh"] = get_roc_score(test_pos,test_neg, sh_scores)
        mat["ka"] = ka_scores
        mat["pa"] = pa_scores
        mat["sh"] = sh_scores
        return uns_res, mat

    else:
        mat = {}
        ka_mat = []
        pa_mat = []
        sh_mat = []
        uns_res = {}
        g1 = nx.Graph(arr[0])
        for u in [n for n in list(g) if g.nodes[n]["bipartite"] == 0]:
            for v in [n for n in list(g) if g.nodes[n]["bipartite"] == 1]:
                g1.add_edge(u, v)

        for n in range(len(arr) - 1):
            g0 = arr[n]

            pa_scores = preferential_attachment_scores(g0, nodelist0)
            ka_scores = katz_scores(g0, nodelist0)
            sh_scores = sinh_scores(g0, nodelist0)

            ka_mat.append(mat_to_arr(ka_scores))
            pa_mat.append(mat_to_arr(pa_scores))
            sh_mat.append(mat_to_arr(sh_scores))

        true_ka = np.array(ka_mat).reshape((len(arr)-1, ka_mat[0].shape[1]))
        true_pa = np.array(pa_mat).reshape((len(arr)-1, pa_mat[0].shape[1]))
        true_sh = np.array(sh_mat).reshape((len(arr)-1, sh_mat[0].shape[1]))
        adj = nx.to_numpy_matrix(g1, nodelist0)
        t = mat_to_arr(adj)

        train = bipartite_data_edge(arr[-2], nx.to_numpy_matrix(arr[-2], nodelist0), nodelist0)
        test_pos, test_neg, all_edge = bipartite_data_edge(arr[-1], nx.to_numpy_matrix(arr[-1], nodelist0), nodelist0)
        pred_ka = time_series_predict(true_ka, t, month_gap).reshape(ka_scores.shape)
        pred_pa = time_series_predict(true_pa, t, month_gap).reshape(pa_scores.shape)
        pred_sh = time_series_predict(true_sh, t, month_gap).reshape(sh_scores.shape)

        pw, iw = extract_edge_attribute(arr[-2], nx.to_numpy_matrix(arr[-2], nodelist=nodelist0), nodelist0)
        pred_svm, labels_svm, acc = SVM_score(train, [test_pos, test_neg, all_edge], [ka_scores, pred_ka] , [pa_scores, pred_pa], [sh_scores, pred_sh], pw, iw)

        uns_res["ka"] = get_roc_score(test_pos, test_neg, pred_ka)
        uns_res["pa"] = get_roc_score(test_pos,test_neg, pred_pa)
        uns_res["sh"] = get_roc_score(test_pos,test_neg, pred_sh)
        rocsvm = ["","", roc_curve(labels_svm, pred_svm)]
        uns_res["svm"] = roc_auc_score(labels_svm, pred_svm), average_precision_score(labels_svm, pred_svm), rocsvm, threshold_prediction(pred_svm, rocsvm, labels_svm)
        mat["ka"] = pred_ka
        mat["svm"] = pred_svm
        mat["pa"] = pred_pa
        mat["sh"] = pred_sh

        return uns_res, mat


#Input: scores matrix
#Output:score matrix flattened like i want
def mat_to_arr(mat):
    A = np.matrix.flatten(mat)
    if A.shape[0] > 1:
        A = A.reshape((1, A.shape[0]))
    return A

#Input:array of flattened metrics score
#Output: predicted one step ahead metrics score
def time_series_predict(arr, should, month_gap):
    predicted = np.zeros(arr.shape[1])
    for n in range(arr.shape[1]):
        if should[0,n] == 1:
            lr = LinearRegression().fit(np.asarray(range(len(arr[:, n]))).reshape(-1, 1), arr[:, n])
            val = lr.predict(np.asarray(len(arr[:,n])+month_gap-1).reshape(1, -1))
            # predicted.append(ARIMA(arr[:, n].T, order=(0,0,1)).fit().forecast()[0])
            predicted[n] = val[0]
    return np.asarray(predicted)


def result_formater(res, path_folder, file_name):
    plt.figure()
    plt.plot([0,1],[0,1],"g--")
    label={"ka":"Katz Index AUC = ", "pa":"Preferential Attachment Index AUC = ",
           "sh":"Hyperbolic Sine Index AUC = ", "svm":"SVM AUC = "}
    graphic = GraphicBase("ROC Curve",
                          "",
                          "False Positive Rate",
                          "True Positive Rate",
                          date_format=False)
    for key in res.keys():
        test_fpr, test_tpr, threshold = res[key][2]
        graphic.ax.plot(test_fpr, test_tpr, label = label[key] + str(round(res[key][0], 3)), lw = 5)
    graphic.ax.plot([0, 1], [0, 1], linestyle="--", lw = 5, label = "Random Classifier", alpha=.8)
    plt.legend(loc = "lower right",prop={'size': 40})
    graphic.save_graph(path_folder, file_name)


def cross_val_split(arr, nodelist0):
    if len(arr) < 3:
        raise ValueError("wesh")
    key=["ka", "pa", "sh", "svm"]
    label={"ka":"Katz Index", "pa":"Preferential Attachment Index",
           "sh":"Hyperbolic Sine Index", "svm":"SVM"}
    res1 = {}
    std = {}
    mean_res = {}
    graphic = GraphicBase("AUC evolution through time",
                          "",
                          "",
                          "AUC",
                          date_format=False)
    for k in key:
        mean_res[k] = 0
        res1[k] = []
    for n in range(3, len(arr) + 1):
        res, __ = calculate_time_score(arr[0:n], nodelist0)
        # result_formater(res, "figures/ROC/", "ROC_"+arr[n-1].name+".pdf")
        for k in key:
            res1[k].append(res[k][0])
            mean_res[k] += res[k][0]/(len(arr) - 2)
    for k in key:
        std[k] = np.std(res1[k])
        graphic.ax.plot(res1[k], label = label[k], lw = 5)
    plt.legend(loc = "lower right",  prop={'size': 40})
    graphic.save_graph("figures/ROC/","AUC_evolution.pdf")

    return mean_res, std , res1


def opti_hyperparam(arr, nodelist0):
    res = []
    best = 0
    alphabest = 0
    g1 = nx.Graph(arr[0])
    for u in [n for n in list(g) if g.nodes[n]["bipartite"] == 0]:
        for v in [n for n in list(g) if g.nodes[n]["bipartite"] == 1]:
            g1.add_edge(u, v)
    adj = nx.to_numpy_matrix(g1, nodelist0)
    t = mat_to_arr(adj)
    for a in np.arange(0.01, 0.2, 0.01):
        print(a)
        res, __ = calculate_time_score(arr, nodelist0, a)["svm"][0]
        print(res," ",a)
        if res>best:
            best = res
            alphabest=a
    return alphabest, best

def nodelabel_to_index(nodelist0):
    mapping = {}
    for label in nodelist0:
        mapping[label] = nodelist0.index(label)
    return mapping

def extract_edge_attribute(G, adj, nodelist0):

    pw = np.zeros(adj.shape)
    iw = np.zeros(adj.shape)
    d_pw = nx.get_edge_attributes(G, "pw")
    d_iw = nx.get_edge_attributes(G, "iw")
    mapping = nodelabel_to_index(nodelist0)

    for keys, item in d_pw.items():
        x = mapping[keys[0]]
        y = mapping[keys[1]]
        pw[x, y] = item
        pw[y, x] = item
    for keys, item in d_iw.items():
        x = mapping[keys[0]]
        y = mapping[keys[1]]
        iw[x, y] = item
        iw[y, x] = item

    return pw, iw

def get_ebunch(edge_list, nodelist0):
    label_list = []
    for e in edge_list:
        label_list.append((nodelist0[e[0]],nodelist0[e[1]]))
    return label_list

def printer(res):
    keys = ["ka", "pa", "sh", "svm"]
    for key in keys:
        print(key," AUC: ",res[key][0])
        print(key," APR: ",res[key][1])

def threshold_prediction(pred, res, edge_label):
    ind, treshold = gmeans(res[2])
    pred_lab = []
    for i in pred:
        if i>= treshold:
            pred_lab.append(1)
        else:
            pred_lab.append(0)
    acc = 0
    for n in range(len(pred)):
        if pred_lab[n] == edge_label[n]:
            acc += 1

    return acc / len(pred)

def get_edge_label(g, edge_list, nodelist0):
    edge_label = []
    for edge in edge_list:
        if nodelist0[edge[0]] in g.neighbors(nodelist0[edge[1]]):
            edge_label.append(1)
        else:
            edge_label.append(0)
    return edge_label

def new_edge(g1, g2, nodelist0):
    adj = nx.to_numpy_matrix(g2, nodelist = nodelist0)
    new_edge = []
    pos_edge, __, ____ = bipartite_data_edge(g2, adj, nodelist0)

    for edge in pos_edge:
        if  nodelist0[edge[0]] not in g1.neighbors(nodelist0[edge[1]]):
            new_edge.append(edge)
    return new_edge

def diss_edge(g1, g2, nodelist0):
    adj = nx.to_numpy_matrix(g2, nodelist = nodelist0)
    diss_edge = []
    __ , neg_edge, ___ = bipartite_data_edge(g2, adj, nodelist0)

    for edge in neg_edge:
        if  nodelist0[edge[0]] in g1.neighbors(nodelist0[edge[1]]):
            diss_edge.append(edge)
        return diss_edge

#cross-validation on same split length
# obsolete since cross_val_xmonth
def cross_val_rolling(arr, nodelist0, gap):
    if len(arr) < 3:
        raise ValueError("wesh")
    key=["ka", "pa", "sh", "svm"]
    label={"ka":"Katz Index", "pa":"Preferential Attachment Index",
           "sh":"Hyperbolic Sine Index", "svm":"SVM"}
    res1 = {}
    std = {}
    mean_res = {}
    graphic = GraphicBase("AUC evolution through time",
                          "",
                          "",
                          "AUC",
                          date_format=False)
    for k in key:
        mean_res[k] = 0
        res1[k] = []
    for n in range(gap+1, len(arr) + 1):
        res, __ = calculate_time_score(arr[n-gap-1:n], nodelist0)
        result_formater(res, "figure/ROC_ROLLING/", "ROC_ROLL_"+arr[n-1].name+".pdf")
        for k in key:
            res1[k].append(res[k][0])
            mean_res[k] += res[k][0]/(len(arr) - 2)
    for k in key:
        std[k] = np.std(res1[k])
        graphic.ax.plot(res1[k], label = label[k], lw = 5)
    plt.legend(loc = "lower right",  prop={'size': 40})
    graphic.save_graph("figures/ROC/","Rolling_AUC_evolution.pdf")
    return mean_res, std , res1

# crossvalidation on xx month ahead prediction with either same or cumulative
# split length
def cross_val_xmonth(arr, nodelist0, month_gap, cumul = True, fixed_period = 0):
    if len(arr) < 3:
        raise ValueError("wesh")
    key=["ka", "pa", "sh", "svm"]
    label={"ka":"Katz Index", "pa":"Preferential Attachment Index",
           "sh":"Hyperbolic Sine Index", "svm":"SVM"}
    res1 = {}
    res2 = {}
    std = {}
    mean_res2 = {}
    mean_res = {}

    if cumul:
        file_name = "Cumul_"+str(month_gap)+"_"
    else:
        if fixed_period<2:
            raise ValueError("on peut pas predire avec un seul graph ou moins")
        file_name = "Rolling_"+str(month_gap)+"_"+str(fixed_period)+"_"

    for k in key:
        mean_res[k] = 0
        res1[k] = []
        res2[k] = []
        mean_res2[k] = 0

    for n in range(3, len(arr)-month_gap+2):

        if n+month_gap-1>=len(arr):
            break

        if cumul:
            cop = arr[0:n-1]
            cop.append(arr[n+month_gap-2])
            res, __ = calculate_time_score(cop, nodelist0, month_gap)
        else:
            if n <=fixed_period:
                continue

            cop = arr[n-fixed_period-1:n-1]
            cop.append(arr[n+month_gap-2])

            res, __ = calculate_time_score(cop, nodelist0, month_gap)
        print(cop[-1].name," ",round(res["svm"][3],3))
        result_formater(res, "figures/ROCindeed/",file_name + "ROC"+cop[-1].name+".pdf")
        for k in key:
            res2[k].append(round(res[k][3],3))
            res1[k].append(res[k][0])
            mean_res[k] += res[k][0]
            mean_res2[k] += res2[k][-1]
    graphic = GraphicBase("AUC evolution through time",
                          "",
                          "",
                          "AUC",
                          date_format=False)
    for k in key:
        mean_res[k] = mean_res[k] / len(res1[k])
        std[k] = np.std(res1[k])
        graphic.ax.plot(res1[k], label = label[k] + " Mean AUC: " + str(round(mean_res[k],3)), lw = 5)
    plt.legend( loc = "lower right",  prop={'size': 40})
    graphic.save_graph("figures/ROCindeed/",file_name+"_AUC_evolution.pdf")
    graphic = GraphicBase("Accuracy evolution through time",
                          "",
                          "",
                          "Accuracy",
                          date_format=False)
    for k in key:
        mean_res2[k] = mean_res2[k] / len(res2[k])
        graphic.ax.plot(res2[k], label = label[k] + " Mean Accuracy: " + str(round(mean_res2[k],3)), lw = 5)
    plt.legend( loc = "lower right",  prop={'size': 40})
    graphic.save_graph("figures/ROCindeed/",file_name+"_Accuracy_evolution.pdf")
    return mean_res, std , res1

dir = "indeed_graph/graph*"
arr=[]
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        arr.append(g)

nodelist0=list(arr[0])
std_dict={}
key=["ka", "pa", "sh", "svm"]
for k in key:
    std_dict[k]=[]
for n in range(1,8):
    _,std,___=cross_val_xmonth(arr, nodelist0,n, False, 6)
    print("\n".join([str(x) for x in std]))
    for k in key:
        std_dict[k].append(std[k])

with open("std_mean_AUC.json", "w") as fp:
    json.dump(std_dict, fp, sort_keys = True, indent = 4)
