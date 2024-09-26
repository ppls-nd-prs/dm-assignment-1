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
    naive_splitpoints = (sort[0:len(sort)-1] + sort[1:len(sort)])/2
    splitpoints = np.array([s for s in naive_splitpoints if len(x[x < s]) >= minleaf and len(x[x > s]) >= minleaf])
    return np.array(splitpoints)

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
    features = np.random.choice(len(x[0]),nfeat,replace=False)
    highest_redux = 0
    best_split = None
    for f in features:
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

def sample(x : np.ndarray, y : np.ndarray, size : int):
    '''
    Produces a uniform random sample from data.
    Params:
        - x : 2d numpy.ndarray : the (multi-featured) data points
        - y : 1d numpy.ndarray : the class labels
        - size : int : the size the sample should take
    '''
    sample_x = []
    sample_y = []
    for _ in range(size):
        i = np.random.choice(range(len(x)))
        sample_x.append(x[i])
        sample_y.append(y[i])
    return (np.array(sample_x),np.array(sample_y))
           
def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int):
    '''
    Grows a classification tree.
    Params:
        - x: numpy.ndarray: 2d matrix of data points
        - y: numpy.ndarray: 1d vector of class labels
        - nmin: int: minimal number of observations a node must contain
        - minleaf: int: minimum number of observations required for a leaf node
        - nfeat: int: number of features that must considered for each split
    '''
    start_node = Node(x,y)
    tree = start_node
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
    return tree

def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m : int):
    '''
    Build multiple trees for different bootstrap sample from the original dataset.
    Params:
        - x: numpy.ndarray: 2d matrix of data points
        - y: numpy.ndarray: 1d vector of class labels
        - nmin: int: minimal number of observations a node must contain
        - minleaf: int: minimum number of observations required for a leaf node
        - nfeat: int: number of features that must considered for each split
        - m: int: the amount of trees to be trained on bootstrap samples.
    '''
    trees = []
    for i in range(m):
        sample_x,sample_y = sample(x,y,len(x))
        tree = tree_grow(sample_x,sample_y,nmin,minleaf,nfeat)
        trees.append(tree)
    return trees

def traverse_tree(e : np.ndarray, node : Node):
    '''
    Traverses the tree based on a data point's feature and nodes'
    split values.
    Params:
        - e : 1d numpy.ndarray : the data point to traverse the tree
        - node : Node : the node to start the traversing in 
    '''
    if not(node.s):
        return node
    elif e[node.f] <= node.s: # is it true that only here this is a problem?
        return traverse_tree(e, node.l)
    elif e[node.f] > node.s:
        return traverse_tree(e, node.r)
    else:
        print("node.x")
        raise Exception("Data point feature value is equal to node split value. This could be caused by a bug in candidate_splits.")

def tree_pred(x : np.ndarray, tr : tuple):
    '''
    Return a numpy.ndarray containing prediction labels for datapoints
    based on the majority label of the leaf node the data points get
    classified into.
    Params:
        - x : 2d numpy.ndarray : the (multi-featured) data points
        - tr : tuple (Node,[Node]) : the classification tree
    '''
    y_hat = []
    for e in x:
        leaf = traverse_tree(e, tr)
        y_hat.append(round(sum(leaf.y)/len(leaf.y)))
    return np.array(y_hat)

def tree_pred_b(x : np.ndarray, tree_list : list):
    '''
    Gives majority vote label predictions for data points based
    on a list of classification trees.
    Params:
        - tree_list : list : a list of classification trees
        - x : np.ndarray : the (multi-featured) data points
    '''
    pred_list = []
    for tree in tree_list:
        tree_hat_y = tree_pred(x,tree)
        pred_list.append(tree_hat_y)
    y_hat = np.round(np.sum(pred_list,axis=0)/len(pred_list))
    return np.array(y_hat)

        
    


