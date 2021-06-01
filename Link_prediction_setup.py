import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import json
import time
import glob
import random
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm
from networkx.readwrite import json_graph
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from statsmodels.tsa.arima.model import ARIMA
from scipy.sparse.linalg import inv
from scipy.sparse import identity
import warnings
from sklearn.svm import LinearSVC
from tqdm  import tqdm



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
    #sometimes it fails due to some numbers being arrays but i havent found The
    #issue
    try:
        # Calculate scores
        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

        roc_score = roc_auc_score(labels_all, preds_all)
        roc_curve_tuple = roc_curve
        ap_score = average_precision_score(labels_all, preds_all)
        return roc_score, ap_score, roc_curve_tuple
    except:
        #handles the exception
        preds_neg1=[x[0] for x in preds_neg]
        #sometimes it switches the neg and pos so if it happens it switch back
        if sum(preds_pos)==0:
            preds_all = np.hstack([preds_neg1, preds_pos])
            labels_all = np.hstack([np.ones(len(preds_neg1)), np.zeros(len(preds_pos))])
        else:
            preds_all = np.hstack([preds_pos, preds_neg1])
            labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg1))])

        roc_score = roc_auc_score(labels_all, preds_all)
        roc_curve_tuple = roc_curve
        ap_score = average_precision_score(labels_all, preds_all)
        return roc_score, ap_score, roc_curve_tuple



#Input:ROC curve from scikit roc_curve() function and root bool to know which of the two method to use
#Output: Index (labels_all, preds_all)of the optimal threshold in roccurve and value of optimal threshold
def gmeans(roc_curve,root = False):

    if root:
        #gmeans method found in literature
        g = np.sqrt(roc_curve[0] * (1-roc_curve[1]))
    else:
        #basic method
        g = roc_curve[1]-roc_curve[0]

    #find index of optimal threshold, then the threshold
    ind = np.argmax(g)
    threshold = roc_curve[2][ind]
    return ind, threshold

#Input:Netowrkx training graph, damping factor alpha
#Output: Matrix of scores
def sinh_scores(g_train, nodelist0, alpha = 0.005):
    #compute adjacency matrix
    adj_train = nx.to_numpy_matrix(g_train,nodelist=nodelist0)
    sinh_mat = (expm(alpha*adj_train)-expm(-alpha*adj_train))/2
    sinh_mat = sinh_mat/sinh_mat.max()
    sinh_mat = sinh_mat
    return sinh_mat

# Input: NetworkX training graph
# Output: Score matrix
def preferential_attachment_scores(g_train,nodelist0):
    adj_train = nx.to_numpy_matrix(g_train,nodelist = nodelist0)
    # Calculate scores
    mapping = nodelabel_to_index(nodelist0  )
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train):# (u, v) = node indices, p = Jaccard coefficient
        i = mapping[u]
        j = mapping[v]
        pa_matrix[i][j] = p
        pa_matrix[j][i] = p
        # make sure it's symmetric
    pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix

    return pa_matrix

#Input:Networkx training graph,max_power
#Output: Matrix scores
def katz_scores(g_train, nodelist0, max_power = 5, beta =  0.045):
    adj_train = nx.to_numpy_matrix(g_train, nodelist=nodelist0)
    ka_scores = {}
    ka_score_matrix = (np.linalg.inv(identity(adj_train.shape[1])-beta*adj_train)-identity(adj_train.shape[1]))
    return ka_score_matrix

#Input:edge list to train on,edge list to test on, scores matrix of other metrics
#Output: mmmmh not sure yet
def SVM_score(test_split1, test_split2, ka_scores, pa_scores, sh_scores, c = 0.9):

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
    clf = LinearSVC(C=c)
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
def bipartite_data_edge(G, adj, nodelist0):
    adj=coo_matrix(adj)
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
def calculate_time_score(arr, nodelist0):
    ka_mat = []
    pa_mat = []
    sh_mat = []
    uns_res = {}
    g1=nx.Graph(arr[0])
    for u in [n for n in list(g) if g.nodes[n]["bipartite"]==0]:
        for v in [n for n in list(g) if g.nodes[n]["bipartite"]==1]:
            g1.add_edge(u,v)
    for n in range(len(arr)-1):
        print(n)
        g0 = arr[n]
        time1 = time.time()
        pa_scores = preferential_attachment_scores(g0,nodelist0)
        print("PA time: ", time.time() - time1)
        time1=time.time()
        ka_scores = katz_scores(g0, nodelist0)
        print("KA time: ", time.time() - time1)
        time1=time.time()
        sh_scores = sinh_scores(g0, nodelist0)
        print("sh time: ", time.time() - time1)

        ka_mat.append(mat_to_arr(ka_scores))
        pa_mat.append(mat_to_arr(pa_scores))
        sh_mat.append(mat_to_arr(sh_scores))


    true_ka = np.array(ka_mat).reshape((len(arr)-1, ka_mat[0].shape[1]))
    true_pa = np.array(pa_mat).reshape((len(arr)-1, pa_mat[0].shape[1]))
    true_sh = np.array(sh_mat).reshape((len(arr)-1, sh_mat[0].shape[1]))

    adj=nx.to_numpy_matrix(g1, nodelist0)
    t=mat_to_arr(adj)

    train = bipartite_data_edge(arr[-2], nx.to_numpy_matrix(arr[-2]),nodelist0)
    test_pos, test_neg, all_edge = bipartite_data_edge(arr[-1], nx.to_numpy_matrix(arr[-1]),nodelist0)
    pred_ka = time_series_predict(true_ka,t).reshape(ka_scores.shape)
    pred_pa = time_series_predict(true_pa,t).reshape(pa_scores.shape)
    pred_sh = time_series_predict(true_sh,t).reshape(sh_scores.shape)

    pred_svm, labels_svm = SVM_score(train, [test_pos, test_neg, all_edge], pred_ka, pred_pa, pred_sh)

    uns_res["ka"] = get_roc_score(test_pos, test_neg, pred_ka)
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
def time_series_predict(arr,should):
    predicted = []
    for n in tqdm(range(arr.shape[1])):
        if should[0,n] == 1:
            print(predicted[-1])
            predicted.append(ARIMA(arr[:, n].T, order=(1,0,0)).fit().forecast()[0])
        else:
            predicted.append(0)
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
    plt.savefig("res1.pdf")

def cross_val(arr, nodelist0):
    if len(arr)<3:
        raise ValueError("wesh")
    key=["ka","pa","sh","svm"]
    mean_res={}
    for k in key:
        mean_res[k]=0
    for n in range(3,len(arr)+1):
        res=calculate_time_score(arr[0:n], nodelist0)
        for k in key:
            mean_res[k] += res[k][0]/(len(arr)-2)
    return mean_res

def opti_hyperparam(arr, nodelist0):
    res = []
    best=0
    alphabest=0
    for a in tqdm(np.arange(0.1,1,0.1)):
        for n in range(len(arr)-1):
            g0 = arr[n]
            ka_scores = katz_scores(g0, nodelist0)
            sh_scores = sinh_scores(g0, nodelist0)
            pa_scores = preferential_attachment_scores(g0,nodelist0)
            test_pos, test_neg, test_all = bipartite_data_edge(arr[n+1], nx.to_numpy_matrix(arr[n+1]), nodelist0)
            train = bipartite_data_edge(arr[n], nx.to_numpy_matrix(arr[n]),nodelist0)
            pred_svm, labels_svm = SVM_score(train, [test_pos, test_neg, test_all], ka_scores, pa_scores, sh_scores,a)
            res.append(roc_auc_score(labels_svm,pred_svm))
        if sum(res)/len(res)>best:
            best = sum(res)/len(res)
            alphabest=a

    return alphabest, best

def nodelabel_to_index(nodelist0):
    mapping = {}
    for label in nodelist0:
        mapping[label] = nodelist0.index(label)
    return mapping




dir = "final_graph/graph*"
arr=[]
for path in glob.glob(dir,recursive=True):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        if g.name=="201803":
            set1=random.choices([x for x, y in g.nodes(data=True) if y["bipartite"] == 0], k=30)
            set2 = random.choices([x for x, y in g.nodes(data=True) if y["bipartite"] == 1], k=10)
            set1.extend(set2)
            sett=set1
        print(len(sett))
        g1=g.subgraph(sett)
        if nx.number_of_edges(g1)==0:
            continue

        print(g1.name)
        arr.append(g1)

nodelist0=list(arr[0])
print(nx.info(arr[0]))
a=calculate_time_score(arr[0:3], nodelist0)
print(a["sh"][0])
