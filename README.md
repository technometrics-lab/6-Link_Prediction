Link Prediction for Cybersecurity Companies and Technologies
===============================================================

## Goal

This project aims to construct a dynamic bipartite network that model the
relationship between companies and technologies in th ecybersecurity landscape
using job openings. Then predict the evolution of these networks in the future
using link predictions algorithms.

------
## Abstract

The cybersecurity market is a dynamic environment in which novel entities -- technologies and companies -- arise and disappear swiftly. In such a fast-paced context, assessing the relations (i.e., links) between those entities is crucial for investment decisions that aims to foster cybersecurity. In this paper, we present a framework for capturing such relations within the Swiss cybersecurity landscape. By using open data, we first model our dataset as a bipartite graph in which nodes are represented by technologies and companies involved in cybersecurity. Then, we use job-openings data to link these two entities. By extracting time series of such graphs, and by using link-prediction methods, we forecast the (dis)appearance of links (and thus relationships) between technologies and companies. We apply several unsupervised learning similarity-based algorithms, a supervised learning method, and finally we select the best method that models such links. Our results show good performance and promising validation of our predicting power. We suggest that our framework is useful for investment decisions in the domain of cybersecurity, as assessing and forecasting links formation and disappearance between companies and technologies enables to shed some light on the rather opaque cybersecurity landscape. Our framework brings decisions-makers a structured tool for more informed investment decisions.

For more information please refer to the documentation and the [article](Article.pdf).
