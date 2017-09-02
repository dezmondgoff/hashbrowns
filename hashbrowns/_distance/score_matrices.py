import os

def load_matrix(filename):

    out = dict()

    with open(filename) as f:
        L = f.readline();
        while L.startswith('#'):
            L = f.readline()

        keys = [str(x) for x in L.split()]

        L = f.readline()
        while L != '':
            values = L.split()
            k1 = values[0]
            for i, v in enumerate(values[1:]):
                out[k1 + keys[i]] = int(v)
            L = f.readline()

    return out

def load_blosum62():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return load_matrix(os.path.join(dir_path, 'score_matrices/blosum62.txt'))

def load_pam250():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return load_matrix(os.path.join(dir_path, 'score_matrices/pam250.txt'))
