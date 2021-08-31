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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score, precision_recall_curve, PrecisionRecallDisplay
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
def get_roc_score(edges_pos, edges_neg, score_matrix, thresholds, apply_sigmoid = False):

    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions
    preds_pos = []
    print("test_set: ", len(edges_pos))
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

    # stack positives and negatives
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    # calculate scores and roc curve
    roc_score = roc_auc_score(labels_all, preds_all)
    roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    pr_curve_tuple = precision_recall_curve(labels_all, preds_all)
    acc = []
    for t in thresholds:
        acc.append(threshold_prediction(preds_all,t,labels_all))

    return roc_score, ap_score, roc_curve_tuple, pr_curve_tuple, acc
# Input: edge_lists, score matrix and bool to know if SVM
# Output: three best thresholds from three methods
def get_best_thresh(edges_pos, edges_neg, score_matrix, svm = False):
    if svm:
        # here edge_pose corresponds to all predicted scores and edge_neg to actual labels
        # score matrix is empty
        roc_curve_tuple = roc_curve(edges_neg, edges_pos)
        pr_curve_tuple = precision_recall_curve(edges_neg, edges_pos)
    else:
        preds_pos = []
        preds_neg = []
        for edge in edges_pos:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        for edge in edges_neg:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
        roc_curve_tuple = roc_curve(labels_all, preds_all)
        pr_curve_tuple = precision_recall_curve(labels_all, preds_all)

    t1, ix1 = gmeans(roc_curve_tuple)
    t2, ix2 = gmeans(roc_curve_tuple, True)
    t3, ix3 = pr_threshold(pr_curve_tuple)
    return t1, t2, t3

def pr_threshold(pr_curve):
    fscore = (2 * pr_curve[0] * pr_curve[1]) / (pr_curve[0] + pr_curve[1])
    ix = np.argmax(fscore)

    thr = pr_curve[2][ix]
    fmax = fscore[ix]
    return thr, fmax

# Input:ROC curve from scikit roc_curve() function and root bool to know which of the two method to use
# Output: Index (labels_all, preds_all)of the optimal threshold in roccurve and value of optimal threshold
def gmeans(roc_curve,root = False):

    if root:
        # gmeans method found in literature
        g = np.sqrt(roc_curve[1] * (1 - roc_curve[0]))
    else:
        # basic method
        g = roc_curve[1] - roc_curve[0]

    # find index of optimal threshold, then the threshold
    ind = np.argmax(g)
    threshold = roc_curve[2][ind]
    return threshold, g[ind]

# Input:Netowrkx training graph, damping factor alpha
# Output: Matrix of scores
def sinh_scores(g_train, nodelist0, alpha = 0.018):
    # compute adjacency matrix
    adj_train = nx.to_numpy_matrix(g_train, nodelist=nodelist0)
    sinh_mat = (expm(alpha * adj_train) - expm(-alpha * adj_train))/2
    sinh_mat = sinh_mat/sinh_mat.max()
    return sinh_mat

# Input: NetworkX training graph
# Output: Score matrix
def preferential_attachment_scores(g_train, nodelist0):
    adj_train = nx.to_numpy_matrix(g_train, nodelist = nodelist0)
    # map of node label to integer index
    pos, neg, all = bipartite_data_edge(g_train, adj_train, nodelist0)
    mapping = nodelabel_to_index(nodelist0)
    ebunch1 = get_ebunch(all, nodelist0)
    pa_matrix = np.zeros(adj_train.shape)
    # compute scores for each edge
    for u, v, p in nx.preferential_attachment(g_train, ebunch1):# (u, v) = node indices, p = Jaccard coefficient
        i = mapping[u]
        j = mapping[v]
        pa_matrix[i][j] = p
        pa_matrix[j][i] = p
        # make sure it's symmetric
    if not pa_matrix.max() == 0:
        pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix

    return pa_matrix

# Input:Networkx training graph,max_power
# Output: Matrix scores
def katz_scores(g_train, nodelist0, beta =  0.013):
    adj_train = nx.to_numpy_matrix(g_train, nodelist=nodelist0)
    ka_score_matrix = np.linalg.inv(identity(adj_train.shape[1]) - beta * adj_train) - identity(adj_train.shape[1])
    ka_score_matrix = ka_score_matrix / ka_score_matrix.max()
    return ka_score_matrix

# Input:edge list to train on,edge list to test on, scores matrix of other metrics
# Output: test scores, test labels, train scores, train labels
def SVM_scores(train_split, test_split, ka_scores, pa_scores, sh_scores, pw, iw, c = 0.14):

    train_pos, train_neg, train_all = train_split
    test_pos, test_neg, test_all = test_split

    # create feature vectore for train and test
    att_train_pos = create_attributes(train_pos, ka_scores[0], pa_scores[0], sh_scores[0], pw, iw)
    att_train_neg = create_attributes(train_neg, ka_scores[0], pa_scores[0], sh_scores[0], pw, iw)
    att_test_pos = create_attributes(test_pos, ka_scores[1], pa_scores[1], sh_scores[1], pw, iw)
    att_test_neg = create_attributes(test_neg, ka_scores[1], pa_scores[1], sh_scores[1], pw, iw)

    # Train SVM
    preds_all = np.vstack([att_train_pos, att_train_neg])
    labels_all = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    clf = SVC(C=c, gamma=0.00001, probability=True)
    clf.fit(preds_all, labels_all)

    # Test SVM
    preds_test_all = np.vstack([att_test_pos, att_test_neg])
    labels_test = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    return clf.predict_proba(preds_test_all)[:,1], labels_test, clf.predict_proba(preds_all)[:,1], labels_all

# Input:sparse matrix
# Output:flat array
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Input:Networkx graph with partition attribute otherwise it wont work,adjacency matrix of the graph
# Output:Present edge list, non present edge list, all possible edge list
def bipartite_data_edge(G, adj, nodelist0):

    adj = coo_matrix(adj)
    # get upper triangular of adj since it contains all possible edges(even more than all possibles in bipartite)
    adj_triu = sp.triu(adj)
    edges_tuple = sparse_to_tuple(adj_triu)
    edges = edges_tuple[0]
    mapping = nodelabel_to_index(nodelist0)
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]

    set_edge = set(edge_tuples)
    # create negative edges list
    # don't include edges that are between node in same groups
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

    return edge_tuples, edge_neg, all_pos_edge

# Input:edge list to create feature vector for, scores matrices
# Output: feature flat array for SVM
def create_attributes(edge_list, ka_scores, pa_scores, sh_scores, pw, iw):
    attr = np.zeros((len(edge_list), 4))
    n = 0
    for edge in edge_list:
        attr[n][3] = ka_scores[edge[0], edge[1]]
        attr[n][1] = sh_scores[edge[0], edge[1]]
        attr[n][0] = pa_scores[edge[0], edge[1]]
        attr[n][2] = pw[edge[0], edge[1]]
        n = n + 1
    return attr

# Input: array of graphs
# output:dict of performance results
def calculate_time_score(arr, nodelist0, month_gap = 1):
    # if simple one step link prediction we don't use forecasting methods
    if len(arr) == 2:
        mat = {}
        uns_res = {}
        # get score from first graph
        pa_scores = preferential_attachment_scores(arr[0], nodelist0)
        ka_scores = katz_scores(arr[0], nodelist0)
        sh_scores = sinh_scores(arr[0], nodelist0)
        train_pos, train_neg, all_edge = bipartite_data_edge(arr[0], nx.to_numpy_matrix(arr[0], nodelist0), nodelist0)
        test_pos, test_neg, all_edge = bipartite_data_edge(arr[-1], nx.to_numpy_matrix(arr[-1], nodelist0), nodelist0)
        diff_pos = edge_diff(arr[0],arr[1],nodelist0)
        # get performance from second graph
        t1 = get_best_thresh(train_pos,train_neg, ka_scores)
        t2 = get_best_thresh(train_pos,train_neg, pa_scores)
        t3 = get_best_thresh(train_pos,train_neg, sh_scores)
        uns_res["ka"] = get_roc_score(test_pos, test_neg, ka_scores, t1)
        uns_res["pa"] = get_roc_score(test_pos,test_neg, pa_scores, t2)
        uns_res["sh"] = get_roc_score(test_pos,test_neg, sh_scores, t3)
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
        t_ka = []
        t_pa = []
        t_sh = []
        # create complete bipartite graph to obtain complete adjacency matrix
        g1 = nx.Graph(arr[0])
        for u in [n for n in list(g) if g.nodes[n]["bipartite"] == 0]:
            for v in [n for n in list(g) if g.nodes[n]["bipartite"] == 1]:
                g1.add_edge(u, v)

        # compute scores matrix for each graph in training period
        for n in range(len(arr) - 1):
            g0 = arr[n]

            pa_scores = preferential_attachment_scores(g0, nodelist0)
            ka_scores = katz_scores(g0, nodelist0)
            sh_scores = sinh_scores(g0, nodelist0)
            ka_mat.append(mat_to_arr(ka_scores))
            pa_mat.append(mat_to_arr(pa_scores))
            sh_mat.append(mat_to_arr(sh_scores))


        # setup nicely so that an edge score trough time is one row vector in this matrix
        true_ka = np.array(ka_mat).reshape((len(arr)-1, ka_mat[0].shape[1]))
        true_pa = np.array(pa_mat).reshape((len(arr)-1, pa_mat[0].shape[1]))
        true_sh = np.array(sh_mat).reshape((len(arr)-1, sh_mat[0].shape[1]))
        # get complete graph adjacency matrix and flat array
        adj = nx.to_numpy_matrix(g1, nodelist0)
        t = mat_to_arr(adj)

        # predict score evolution
        train = bipartite_data_edge(arr[-2], nx.to_numpy_matrix(arr[-2], nodelist0), nodelist0)
        test_pos, test_neg, all_edge = bipartite_data_edge(arr[-1], nx.to_numpy_matrix(arr[-1], nodelist0), nodelist0)
        pred_ka = time_series_predict(true_ka, t, month_gap).reshape(ka_scores.shape)
        pred_pa = time_series_predict(true_pa, t, month_gap).reshape(pa_scores.shape)
        pred_sh = time_series_predict(true_sh, t, month_gap).reshape(sh_scores.shape)

        # get patent and job openings numbers and train predict svm
        pw, iw = extract_edge_attribute(arr[-2], nx.to_numpy_matrix(arr[-2], nodelist0), nodelist0)
        pred_svm, labels_svm , pred_train_svm, labels_train_svm = SVM_scores(train,
            [test_pos, test_neg, all_edge], [ka_scores, pred_ka] , [pa_scores, pred_pa], [sh_scores, pred_sh], pw, iw)

        # get thresholds from train data
        tka = get_best_thresh(train[0], train[1], ka_scores)
        tpa = get_best_thresh(train[0], train[1], pa_scores)
        tsh = get_best_thresh(train[0], train[1], sh_scores)
        tsvm = get_best_thresh(pred_svm, labels_svm, "", True)
        # obtain performance plots and metrics (except for svm)
        ka_res = get_roc_score(test_pos, test_neg, pred_ka, tka)
        pa_res = get_roc_score(test_pos, test_neg, pred_pa, tpa)
        sh_res = get_roc_score(test_pos, test_neg, pred_sh, tsh)

        # compute svm accuracy
        accsvm = []
        for t in tsvm:
            accsvm.append(threshold_prediction(pred_svm, t, labels_svm))
        # store all the results (like that to be of the same format than svm)
        uns_res["ka"] = ka_res[0], ka_res[1], ka_res[2], ka_res[3], ka_res[4]
        uns_res["pa"] = pa_res[0], pa_res[1], pa_res[2], pa_res[3], pa_res[4]
        uns_res["sh"] = sh_res[0], sh_res[1], sh_res[2], sh_res[3], sh_res[4]
        uns_res["svm"] = roc_auc_score(labels_svm, pred_svm), average_precision_score(labels_svm, pred_svm), roc_curve(labels_svm, pred_svm), precision_recall_curve(labels_svm, pred_svm), accsvm
        # get final score matrices just in case
        mat["ka"] = pred_ka
        mat["svm"] = pred_svm
        mat["pa"] = pred_pa
        mat["sh"] = pred_sh

        return uns_res, mat


# Input: scores matrix
# Output:score matrix flattened like i want
def mat_to_arr(mat):
    A = np.matrix.flatten(mat)
    if A.shape[0] > 1:
        A = A.reshape((1, A.shape[0]))
    return A

# Input:array of flattened metrics score, array to know which entries to perform prediction on
# month gap to predict
# Output: predicted one step ahead metrics score
def time_series_predict(arr, should, month_gap):

    predicted = np.zeros(arr.shape[1])
    for n in range(arr.shape[1]):
        # if we should compute or not
        if should[0,n] == 1:
            # fit
            lr = LinearRegression().fit(np.asarray(range(len(arr[:, n]))).reshape(-1, 1), arr[:, n])
            # predict and format output
            val = lr.predict(np.asarray(len(arr[:,n])+month_gap-1).reshape(1, -1))
            predicted[n] = val[0]
    return np.asarray(predicted)

# Input: result from calculate_time_score path in which to save and file name
# Output: saves roc curve in specified folder_file path
def result_formater(res, method, test_graph):
    label={"ka":"Katz Index", "pa":"Preferential Attachment Index",
           "sh":"Hyperbolic Sine Index", "svm":"SVM"}
    mark_dict={"ka":"v", "pa":"<",
           "sh":"s", "svm":"o"}
    # set up graphic title and axis label
    graphic = GraphicBase("",
                          "",
                          "False Positive Rate",
                          "True Positive Rate",
                          date_format=False)
    # plot roc curve for each method
    for key in res.keys():
        test_fpr, test_tpr, threshold = res[key][2]
        graphic.ax.plot(test_fpr, test_tpr, label = label[key] + " AUC = " + str(round(res[key][0], 3)),
                        lw = 5)
    graphic.ax.plot([0, 1], [0, 1], linestyle="--", lw = 5, label = "Random Classifier", alpha=.8)
    plt.legend(loc = "lower right",prop={'size': 40})
    # save figures
    graphic.save_graph("figures/ROCindeed/", method+"bROC"+test_graph+".pdf")

    graphic = GraphicBase("",
                          "",
                          "Recall",
                          "Precision",
                          date_format=False)
    # plot precision recall curve
    for key in res.keys():
        test_prec, test_rec, threshold = res[key][3]
        graphic.ax.plot(test_rec, test_prec, label = label[key] + " AP = " + str(round(res[key][1], 3)),
                         lw = 5)
    graphic.ax.plot([0, 1], [0.005, 0.005], linestyle="--", lw = 5, label = "Random Classifier", alpha=.8)
    plt.legend(loc = "upper right",prop={'size': 40})
    # save figures
    graphic.save_graph("figures/PRindeed/", method+"pAPR"+test_graph+".pdf")
    # save accuracies obtained from the three methods in json file
    d = {}
    for k in res.keys():
        d[k+"roc_linear"] = res[k][-1][0]
        d[k+"roc_gmeans"] = res[k][-1][1]
        d[k+"pr_fscore"] = res[k][-1][2]

    with open("figures/accindeed/"+method+"ACC_pos_"+test_graph+".json","w") as outfile:
        json.dump(d, outfile)


# Input: graph array, nodelist for consistant node ordering
# output best hyperparameter for method defined inside
def opti_hyperparam(arr, nodelist0):
    res = []
    best = 0
    alphabest = 0
    # simple grid search no need to be finer as not that much influence
    for a in np.arange(0.001, 0.02, 0.001):
        # get result with parameter
        res = calculate_time_score(arr, nodelist0, a)[0]["sh"][0]
        # if best then update best param
        if res>best:
            best = res
            alphabest=a

    return alphabest, best

# Input: nodelist for node ordering
# Output: mapping node:id in nodelist for matrix indexing
def nodelabel_to_index(nodelist0):
    mapping = {}
    for label in nodelist0:
        mapping[label] = nodelist0.index(label)
    return mapping

# Input: graph G, adjacency matrix of G, nodelist0
# Output: matrix edge:#job openings (and also patent)
def extract_edge_attribute(G, adj, nodelist0):
    # initial declaration
    pw = np.zeros(adj.shape)
    iw = np.zeros(adj.shape)
    # get edge:weight dict
    d_pw = nx.get_edge_attributes(G, "pw")
    d_iw = nx.get_edge_attributes(G, "iw")
    # get nodelist mapping
    mapping = nodelabel_to_index(nodelist0)

    for keys, item in d_pw.items():
        # get good matrix index for given node
        x = mapping[keys[0]]
        y = mapping[keys[1]]
        # simetric matrix
        pw[x, y] = item
        pw[y, x] = item
    for keys, item in d_iw.items():
        x = mapping[keys[0]]
        y = mapping[keys[1]]
        iw[x, y] = item
        iw[y, x] = item

    return pw, iw

# Input: edge_list in numbers, nodelist0
# output: edge_list with node label instead
def get_ebunch(edge_list, nodelist0):
    label_list = []
    for e in edge_list:
        label_list.append((nodelist0[e[0]],nodelist0[e[1]]))
    return label_list

def threshold_prediction(pred, threshold, edge_label):

    pred_lab = []
    print(threshold)
    print("median of scores ",np.median(pred))

    for i in pred:
        if i>= threshold:
            pred_lab.append(1)
        else:
            pred_lab.append(0)
    acc = 0
    for n in range(len(pred)):
        if pred_lab[n] == edge_label[n]:
            acc += 1
    print("accuracy: ",acc / len(pred))
    return acc / len(pred)

def get_edge_label(g, edge_list, nodelist0):
    edge_label = []
    for edge in edge_list:
        if nodelist0[edge[0]] in g.neighbors(nodelist0[edge[1]]):
            edge_label.append(1)
        else:
            edge_label.append(0)
    return edge_label

# input: array of graphs, ordering of nodes in nodelist0, the forecast range,
# boolean indicator if cumulative or blocked cross validation, fixed period for
# blocked cross validation
# writes all the diagnostics figures for cross validation
def cross_val_xmonth(arr, nodelist0, month_gap, cumul = True, fixed_period = 0):
    # can't do cross val if <3
    if len(arr) < 3:
        raise ValueError("wesh")
    # initialize all variables
    key=["ka", "pa", "sh", "svm"]
    label={"ka":"Katz Index", "pa":"Preferential Attachment Index",
           "sh":"Hyperbolic Sine Index", "svm":"SVM"}
    res1 = {}
    res2 = {}
    res31 = {}
    res32 = {}
    res33 = {}
    std1 = {}
    std2 = {}
    std3 = {}
    mean_res1 = {}
    mean_res2 = {}
    mean_res31 = {}
    mean_res32 = {}
    mean_res33 = {}
    res3 = []
    mean_res3 = []

    # set up file names
    if cumul:
        file_name = "Cumul_"+str(month_gap)+"_"
    else:
        if fixed_period<2:
            raise ValueError("on peut pas predire avec un seul graph ou moins")
        file_name = "Rolling_"+str(month_gap)+"_"+str(fixed_period)+"_"
    # init dictionaries
    for k in key:
        mean_res1[k] = 0
        res1[k] = []
        res2[k] = []
        mean_res2[k] = 0
        res31[k] = []
        res32[k] = []
        res33[k] = []
        mean_res31[k] = 0
        mean_res32[k] = 0
        mean_res33[k] = 0

    # cross validation loop
    for n in range(3, len(arr)-month_gap+2):
        # safe guard
        if n+month_gap-1>=len(arr):
            break

        # train models and compute diagnostics
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

        # plot results
        result_formater(res, file_name, cop[-1].name)

        # compute mean results
        for k in key:
            res1[k].append(res[k][0])
            mean_res1[k] += res[k][0]
            res2[k].append(res[k][1])
            mean_res2[k] += res[k][1]
            res31[k].append(res[k][-1][0])
            res32[k].append(res[k][-1][1])
            res33[k].append(res[k][-1][2])
            mean_res31[k] += res[k][-1][0]
            mean_res32[k] += res[k][-1][1]
            mean_res33[k] += res[k][-1][2]
    res3.append(res31)
    res3.append(res32)
    res3.append(res33)
    mean_res3.append(mean_res31)
    mean_res3.append(mean_res32)
    mean_res3.append(mean_res33)

    # mean results figures
    graphic = GraphicBase("",
                          "",
                          "",
                          "AUC",
                          date_format=False)
    for k in key:
        mean_res1[k] = mean_res1[k] / len(res1[k])
        std1[k] = np.std(res1[k])
        graphic.ax.plot(res1[k], label = label[k] + " Mean AUC: " + str(round(mean_res1[k],3)), lw = 5)
    plt.legend( loc = "lower right",  prop={'size': 40})
    graphic.save_graph("figures/ROCindeed/",file_name+"_AUC_evolution.pdf")

    graphic = GraphicBase("",
                          "",
                          "",
                          "AP",
                          date_format=False)
    for k in key:
        mean_res2[k] = mean_res2[k] / len(res2[k])
        std2[k] = np.std(res2[k])
        graphic.ax.plot(res2[k], label = label[k] + " Mean AP: " + str(round(mean_res2[k],3)), lw = 5)
    plt.legend( loc = "lower right",  prop={'size': 40})
    graphic.save_graph("figures/PRindeed/",file_name+"_APR_evolution.pdf")
    for i in [0,1,2]:
        graphic = GraphicBase("",
                              "",
                              "",
                              "Accuracy",
                              date_format=False)
        for k in key:
            mean_res3[i][k] = mean_res3[i][k] / len(res3[i][k])
            graphic.ax.plot(res3[i][k], label = label[k] + " Mean Accuracy: " + str(round(mean_res3[i][k],3)), lw = 5)
        plt.legend( loc = "lower right",  prop={'size': 40})
        graphic.save_graph("figures/accindeed/",file_name+"_"+str(i)+"_Acc_evolution.pdf")

# computes the symetric difference between the two edges sets of graph g1 g2
def edge_diff(g1,g2,nodelist0):
    adj1 = nx.to_numpy_matrix(g1,nodelist0)
    adj2 = nx.to_numpy_matrix(g2, nodelist0)
    a,b,c = bipartite_data_edge(g1, adj1, nodelist0)
    d, e , f = bipartite_data_edge(g2, adj2, nodelist0)
    diff = []
    for edge in d:
        if edge not in a:
            diff.append(edge)

    return diff

if __name__=='__main__':
    dir = "indeed_graph/graph*"
    arr=[]
    for path in glob.glob(dir,recursive=True):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            g=json_graph.node_link_graph(data)
            arr.append(g)
    nodelist0=list(arr[0])
    # here we can do whatever we need calculate time score, cross val, opti hyperparameter
