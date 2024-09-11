import numpy as np

class Node:
    '''
    Class coding a Node for a tree constructed using tree.
    Properties:
        - self.x : the datapoints
        - self.y : the class labels
        - self.f : the feature to be split on
        - self.s : the split value
        - self.l : the left child
        - self.r : the right child
    '''
    def __init__(self, x : np.ndarray, y : np.ndarray):
        self.x = x
        self.y = y
        self.s = None
        self.f = None
        self.l = None
        self.r = None
        
def impurity(arr : np.ndarray):
    '''
    Calculates a vector's binary impurity with the gini-index
    Params:
        - arr : 1d numpy.ndarray : an array consisting of 0's and 1's
    '''
    sum = np.sum(arr)
    p_1 = sum/arr.size
    return p_1 * (1 - p_1)

def candidate_splits(x : np.ndarray, minleaf : int):
    '''
    Calculates candidate split values for a real-valued array, taking
    a minimal split-size into account.
    Params:
        x : 1d numpy.ndarray : real-valued array to be split
        minleaf : int : the minimal splitsize
    '''
    sort = np.sort(np.unique(x))
    splitpoints = (sort[0+minleaf:len(sort)-(1+minleaf)] + sort[1+minleaf:len(sort)-minleaf])/2
    return splitpoints

def impurity_reduction(s : int, f : int, x : np.ndarray, y : np.ndarray):
    '''
    Calculates the impurity reduction resulting from a split in datapoints
    Params:
        - s : int : the split value
        - f : int : the feature to be split on
        - x : 2d numpy.ndarray : the (multi-featured) data points
        - y : 1d numpy.ndarray : the class labels
    '''
    i_y = impurity(y)
    l = y[x[:,f] < s]
    r = y[x[:,f] > s]
    pi_l = len(l)/len(y)
    pi_r = len(r)/len(y)
    i_l = impurity(l)
    i_r = impurity(r)
    return i_y - pi_l*i_l - pi_r*i_r

def bestsplit(x : np.ndarray, y : np.ndarray, nfeat : int, minleaf : int):
    '''
    Calculates the split that leads to the highest impurity reduction.
    Params:
        - x : 2d numpy.ndarray : the (multifeatured) datapoints
        - y : 1d numpy.ndarray : the class labels
        - nfeat : int : the amount of feature to randomly select a split candidates
        - minleaf : int : the minimum number of observation allowed in a split
    '''
    features = np.random.choice(len(x[0]),nfeat)
    print("features selected:", features)
    highest_redux = 0
    best_split = None
    for f in features:
        #x,y = zip(*sorted(zip(x, y)))
        S = candidate_splits(x[:,f],minleaf)
        for s in S:
            d_i = impurity_reduction(s,f,x,y)
            if d_i > highest_redux:
                highest_redux = d_i
                best_split = s
                best_feature = f
    if best_split:
        return best_split,best_feature
    else:
        return None

def generate_children(s : int, f : int, x : np.ndarray, y : np.ndarray):
    '''
    Generate child nodes based on a split in a feature on a dataset.
    Params:
        - s : int : the split value
        - f : int : the feature to split on
        - x : 2d numpy.ndarray : the (multifeatured) datapoints
        - y : 1d numpy.ndarray : the class labels
    '''
    lx = x[x[:,f] < s]
    ly = y[x[:,f] < s]
    rx = x[x[:,f] > s]
    ry = y[x[:,f] > s]

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
        if len(y) >= nmin:
            if impurity(y) > 0:
                split = bestsplit(x,y,nfeat,minleaf)
                if split:
                    s,f = split
                    current_node.s = s
                    current_node.f = f
                    l,r = generate_children(s,f,x,y)
                    current_node.l = l
                    current_node.r = r
                    nodelist.append(l)
                    nodelist.append(r)
                    tree[1].append(l)
                    tree[1].append(r)
    return tree

def tree_pred(x: np.ndarray,tr: tuple):
    pass

