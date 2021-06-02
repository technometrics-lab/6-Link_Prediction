import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import json
import time
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
    try:
    # Calculate scores
        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

        roc_score = roc_auc_score(labels_all, preds_all)
        roc_curve_tuple = roc_curve(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        return roc_score, ap_score, roc_curve_tuple
    except:
        preds_neg1=[x[0] for x in preds_neg]
        if sum(preds_pos)==0:
            preds_all = np.hstack([preds_neg1, preds_pos])
            labels_all = np.hstack([np.ones(len(preds_neg1)), np.zeros(len(preds_pos))])
        else:
            preds_all = np.hstack([preds_pos, preds_neg1])
            labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg1))])

        roc_score = roc_auc_score(labels_all, preds_all)
        roc_curve_tuple = roc_curve(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        return roc_score, ap_score, roc_curve_tuple

    # return roc_score, roc_curve_tuple, ap_score


#Input:ROC curve from scikit roc_curve() function and root bool to know which of the two method to use
#Output: Index (labels_all, preds_all)of the optimal threshold in roccurve and value of optimal threshold
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
def sinh_scores(g_train, alpha = 0.005):
    adj_train = nx.to_scipy_sparse_matrix(g_train)
    sinh_mat = (expm(alpha*adj_train)-expm(-alpha*adj_train))/2
    sinh_mat = sinh_mat/sinh_mat.max()
    sinh_mat = sinh_mat.todense()
    return sinh_mat

# Input: NetworkX training graph
# Output: Score matrix
def preferential_attachment_scores(g_train):
    adj_train = nx.to_scipy_sparse_matrix(g_train)
    # Calculate scores
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train):# (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p
        # make sure it's symmetric
    pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix

    return pa_matrix

#Input:Networkx training graph,max_power
#Output: Matrix scores
def katz_scores(g_train, max_power = 5, beta = 0.045):
    adj_train = nx.to_scipy_sparse_matrix(g_train)
    ka_score_matrix = (inv(identity(adj_train.shape[1])-beta*adj_train)-identity(adj_train.shape[1])).todense()
    return ka_score_matrix

#Input:edge list to train on,edge list to test on, scores matrix of other metrics
#Output: mmmmh not sure yet
def SVM_score(test_split1, test_split2, ka_scores, pa_scores, sh_scores,c = 0.9):

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
    for x in [x for x, y in G.nodes(data=True) if y["bipartite"] == 0]:
        for y in [x for x, y in G.nodes(data=True) if y["bipartite"] == 1]:
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
    ka_res = []
    pa_res = []
    sh_res = []
    svm_res = []
    res={}
    for n in range(len(arr)-1):
        print(n)
        g0 = arr[n]
        time1 = time.time()
        pa_scores = preferential_attachment_scores(g0)
        print("PA time: ", time.time() - time1)
        time1 = time.time()
        ka_scores = katz_scores(g0)
        print("KA time: ", time.time() - time1)
        time1 = time.time()
        sh_scores = sinh_scores(g0)
        print("sh time: ", time.time() - time1)

        train = bipartite_data_edge(arr[n], nx.to_scipy_sparse_matrix(arr[n]))
        test_pos, test_neg, test_all = bipartite_data_edge(arr[n+1], nx.to_scipy_sparse_matrix(arr[n+1]))

        pred_svm, labels_svm = SVM_score(train, [test_pos, test_neg, test_all], ka_scores, pa_scores, sh_scores)

        ka_res.append(get_roc_score(test_pos,test_neg, ka_scores))
        pa_res.append(get_roc_score(test_pos,test_neg, pa_scores))
        sh_res.append(get_roc_score(test_pos,test_neg, sh_scores))
        prep = roc_auc_score(labels_svm,pred_svm), average_precision_score(labels_svm,pred_svm), roc_curve(labels_svm,pred_svm)
        svm_res.append(prep)

    res["ka"]=ka_res
    res["pa"]=pa_res
    res["sh"]=sh_res
    res["svm"]=svm_res

    return res


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
            lr=LinearRegression().fit(np.asarray(range(len(arr[:,n]))).reshape(-1,1), arr[:,n])
            val = lr.predict(np.asarray(len(arr[:,n])).reshape(1,-1))
            # predicted.append(ARIMA(arr[:, n].T, order=(1,0,0)).fit().forecast()[0])
            predicted.append(val[0])
        else:
            predicted.append(0)
    return np.asarray(predicted)


def result_formater(res):
    plt.figure()
    plt.plot([0,1],[0,1],"g--")
    AUC={}
    APR={}
    label={"ka":"Katz AUC = ","pa":"Preferential Attachment AUC = ",
           "sh":"Hyperbolic Sine AUC = ","svm":"SVM AUC = "}

    for key in label.keys():
        max1=0
        maxkey=0
        AUC[key]=0
        APR[key]=0
        for n in range(len(res[key])):
            if res[key][n][0]>max1:
                max1=res[key][n][0]
                maxkey=n
            AUC[key] += res[key][n][0]/len(res[key])
            APR[key] += res[key][n][1]/len(res[key])

        test_fpr, test_tpr, threshold = res[key][maxkey][2]
        plt.plot(test_fpr, test_tpr, label = label[key] + str(round(max1,3)))

    plt.legend(loc = "lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("figures/reslinear.pdf")
    plt.close()



def opti_hyperparam(arr):
    res = []
    best=0
    alphabest=0
    for a in tqdm(np.arange(0.1,1,0.1)):
        for n in range(len(arr)-1):
            g0 = arr[n]
            ka_scores = katz_scores(g0)
            sh_scores = sinh_scores(g0)
            pa_scores = preferential_attachment_scores(g0)
            test_pos, test_neg, test_all = bipartite_data_edge(arr[n+1], nx.to_scipy_sparse_matrix(arr[n+1]))
            train = bipartite_data_edge(arr[n], nx.to_scipy_sparse_matrix(arr[n]))
            pred_svm, labels_svm = SVM_score(train, [test_pos, test_neg, test_all], ka_scores, pa_scores, sh_scores,a)
            res.append(roc_auc_score(labels_svm,pred_svm))
        if sum(res)/len(res)>best:
            best = sum(res)/len(res)
            alphabest=a

    return alphabest, best


path = ["01","02","03","04","05","06","07","08","09","10","11"]
arr = []
for p in path:
    result_path = "test_graphs/graph" + p + ".json"
    with open(result_path, 'r', encoding = 'utf-8') as file:
        data = json.load(file)
        g=json_graph.node_link_graph(data)
        g.name=p
        arr.append(g)


res=calculate_time_score(arr)
result_formater(res)
