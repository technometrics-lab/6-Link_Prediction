Link Prediction Setup
=====================

Main File of model fitting, analysis and diagnostics.

Training Functions
------------------

**preferential_attachment_scores** (g_train: Networkx.classes.Graph, nodelist0:
array) -> array

  Computes the PA scores matrix for given graph and ordered according to nodelist0.

  **Parameters** :
    * **g_train** (Netowrkx.classes.Graph) - graph on which to compute the scores.
    * **nodelist0** (array) - vertex list that gives the right ordering for the
      score matrix.

**sinh_scores** (g_train: Networkx.classes.Graph, nodelist0: array, alpha: float)
-> array

  Computes the sinh scores matrix for given graph and ordered according to nodelist0.

  **Parameters** :
    * **g_train** (Netowrkx.classes.Graph) - graph on which to compute the scores.
    * **nodelist0** (array) - vertex list that gives the right ordering for the
      score matrix.
    * **alpha** (float) - parameter to compute the scores.

**katz_scores** (g_train: Networkx.classes.Graph, nodelist0: array, beta: float)
-> array

  Computes the katz scores matrix for given graph and ordered according to nodelist0.

  **Parameters** :
    * **g_train** (Netowrkx.classes.Graph) - graph on which to compute the scores.
    * **nodelist0** (array) - vertex list that gives the right ordering for the
      score matrix.
    * **beta** (float) - parameter to compute the scores.

**SVM_scores** (train_split: array, test_split: array, ka_scores: array, pa_scores:
array, sh_scores: array, pw: array, iw: array, c: float) -> array

  Trains an SVM on all the edges in train_split using KA,PA,SH scores then computes
  the score matrix for the test graph represented by test_split.

  **Parameters** :
    * **train_split** (array) - array divided into positive and negative edges for
      training graphs.
    * **test_split** (array) - array divided into positive and negatives edges for
      the test graph.
    * **ka_scores** (array) - katz score matrices computed from the train graphs
      and test graph.
    * **pa_scores** (array) - PA score matrices computed from the train graphs
      and test graph.
    * **sh_scores** (array) - SH score matrices computed from the train graphs
      and test graph.
    * **pw** (array) - feature vector of number of patents linking a company and
      technology. Can be None.
    * **iw** (array) - feature vector of number of job openings linking a company and
      technology.
    * **c** (float) - regularization parameter of the SVM.

**get_best_thresh** (edge_pos: array, edge_neg: array, score_matrix, svm:
optional[Bool] = False) -> array

  Computes the three best thresholds from three different methods.

  **Parameters** :
    * **edge_pos** (array) - array that contains each positive edge coordinates.
    * **edge_neg** (array) - array that contains each negative edge coordinates.
    * **score_matrix** (array) - matrix that contains the score for each edge
      coordinate.
    * **svm** (Bool) - Boolean that indicates if the score matrix is obtained through
      the SVM.

Forecast Functions
------------------
Functions that compute the score matrices for the whole timeframe required and
forecast their evolution.

**time_series_predict** (arr: array, should: array, month_gap: int) -> array

  Uses the score matrices in arr to forecast the score matrix month_gap months in
  the future.

  **Parameters** :
    * **arr** (array) - array of the training score matrices ordered by time.
    * **should** (array) - binary array that indicates if we should fit the time
      series to this edge or not.
    * **month_gap** (int) - number that indicates how far in the future we should
      forecast.

**calculate_time_score** (arr: array, nodelist0: array, month_gap: optional[int] = 1) -> dict

  Function that computes the score matrix for all graphs in arr and then forecast
  their evolution. It then runs the diagnostics and performance results. returns
  the performance result in a dict format.

  **Parameters** :
    * **arr** (array) - array containing all the train graphs required and the
      test graph at the last position.
    * **nodelist0** (array) - common vertex ordering for all the graphs based on
      the first graph of the time series.
    * **mont_gap** (int) - number of month separating the last graph in the
      training set and the test graph.

Get Diagnostics Metrics Function
--------------------------------
Functions to obtain diagnostics score and array for plots.

**get_roc_score** (edge_pos: array, edge_neg: array, score_matrix:
array, thresholds: array, apply_sigmoid: optional[Bool] = False) -> array

  This function computes the AUC, ROC curve, AP, Precision Recall curve and
  accuracy for given score matrix and thresholds. Does not take SVM score matrix
  as it is easier to compute diagnostics on it.

  **Parameters** :
    * **edge_pos** (array) - array that contains each positive edge coordinates.
    * **edge_neg** (array) - array that contains each negative edge coordinates.
    * **score_matrix** (array) - matrix that contains the score for each edge
      coordinate.
    * **thresholds** (array) - array that contains the three best thresholds
      obtained from train data.
    * **apply_sigmoid** (Bool) - boolean that indicates if we should apply the
      sigmoid to the score matrix.
