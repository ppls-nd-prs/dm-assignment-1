import numpy as np

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.s = None
        self.parent = None
        self.l = None
        self.r = None
        
def impurity(arr):
    sum = np.sum(arr)
    p_1 = sum/arr.size
    return p_1 * (1 - p_1)

def candidate_splits(x):
    sort = np.sort(np.unique(x))
    splitpoints = (sort[0:len(sort)-1] + sort[1:len(sort)])/2
    return splitpoints

def impurity_reduction(s,x,y):
    i_y = impurity(y)
    l = y[x < s]
    r = y[x > s]
    pi_l = len(l)/len(y)
    pi_r = len(r)/len(y)
    i_l = impurity(l)
    i_r = impurity(r)
    return i_y - pi_l*i_l - pi_r*i_r

def bestsplit(x,y):
    # Moet uitgebreid worden naar meerdere features
    x,y = zip(*sorted(zip(x, y)))
    x = np.array(x)
    y = np.array(y)
    S = candidate_splits(x)
    highest_redux = 0
    for s in S:
        d_i = impurity_reduction(s,x,y)
        if d_i > highest_redux:
            highest_redux = d_i
            best_split = s
    return best_split

def generate_children(s,x,y):
    lx = x[x < s]
    ly = y[x < s]
    rx = x[x > s]
    ry = y[x > s]

    return Node(lx,ly),Node(rx,ry)

def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int):
    '''
    Grows a classification tree.
    Params:
        x: numpy.ndarray: 2d matrix of data points
        y: numpy.ndarray: 1d vector of class labels
        nmin: int: minimal number of observations a node must contain
        minleaf: int: minimum number of observations required for a leaf node
        nfeat: int: number of features that must considered for each split
    '''
    start_node = Node(x,y)
    tree = (start_node,[])
    nodelist = [start_node]
    while len(nodelist) > 0:
        current_node = nodelist.pop(0)
        x,y = (current_node.x, current_node.y)
        if impurity(y) > 0:
            s = bestsplit(x,y)
            current_node.s = s
            l,r = generate_children(s,x,y)
            current_node.l = l
            current_node.r = r
            nodelist.append(l)
            print("l.x:", l.x)
            print("r.x:", r.x)
            nodelist.append(r)
            tree[1].append(l)
            tree[1].append(r)
    return tree

def tree_pred(x: np.ndarray,tr: tuple):
    pass

