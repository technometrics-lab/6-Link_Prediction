Graph Maker
===========
Functions that reads the json file to output networkx graphs given the path to
the files.

The graphs outputed have the following attributes:
 * Each node has a *bipartite* attribute which is a binary indicator of whether
   they are a technology or a company.
 * Each edge has a *iw* attribute which stands for indeed weight that indicates
   the number of job openings that link the two entities.
