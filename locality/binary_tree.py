import numpy as np
from locality.random.sample import shuffle

class TreeError(Exception):
    pass

class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val

class Tree:
    def __init__(self, equal=lambda a,b: a==b, compare=lambda a,b: a < b):
        self.root = None
        self.equal = equal
        self.compare = compare

    def getRoot(self):
        return self.root

    def add(self, val):
        if(self.root is None):
            self.root = Node(val)
        else:
            self._add(val, self.root)

    def _add(self, val, node):
        if self.compare(val, node.v):
            if(node.l is not None):
                self._add(val, node.l)
            else:
                node.l = Node(val)
        else:
            if(node.r is not None):
                self._add(val, node.r)
            else:
                node.r = Node(val)

    def find(self, val):
        if self.root is not None:
            return self._find(val, self.root)
        return None
    
    def _find(self, val, node):
        if node is None:
            raise TreeError("Value not currently in tree.")
        if self.equal(node.v, val):
            return node
        elif self.compare(node.v, val):
            return self._find(val, node.r)
        return self._find(val, node.l)
    
    def range(self, lo, hi):
        out = []
        if(self.root is not None):
            self._range(lo, hi, self.root, out)
        return out
    
    def _range(self, lo, hi, node, out=[]):
        if node is not None:
            f1 = self.compare(node.v, lo)
            f2 = self.compare(node.v, hi)
            if not f1:
                self._range(lo, hi, node.l, out)
            if f2:
                self._range(lo, hi, node.r, out)
            if (not f1) and f2:
                out.append(node)

    def deleteTree(self):
        # garbage collector will do this for us. 
        self.root = None

    def printTree(self):
        if(self.root is not None):
            self._printTree(self.root)

    def _printTree(self, node):
        if(node is not None):
            self._printTree(node.l)
            print(str(node.v) + ' ')
            self._printTree(node.r)

def sorted_tree(sorted_list, tree=Tree()):
    n = len(sorted_list)
    if n == 0:
        return
    mid = n // 2
    tree.add(sorted_list[mid])
    sorted_tree(sorted_list[:mid], tree)
    sorted_tree(sorted_list[mid + 1:], tree)
    return tree

def random_tree(object_list, length):
    tree = Tree()
    order = np.arange(length)
    shuffle(order)
    for i in order:
        tree.add(object_list[i])
    return tree
    