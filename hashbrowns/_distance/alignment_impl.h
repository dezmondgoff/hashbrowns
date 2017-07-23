typedef bool (*scorePtr)(PyObject *, const char * const, const char * const, 
             int * const);

/* SCORING FUNCTIONS*/

static NPY_INLINE bool binary_search(const int n, const char *A, const char k, 
                                     int * const out)
{
    int L = 0;
    int R = n - 1; 
    int m;
        
    while (true) {
        if (L > R)
            return false;
        m = (L + R) / 2;
        if (A[m] < k)
            L = m + 1;
        else if (A[m] > k)
            R = m - 1;
        else {
            *out = m;
            return true;
        }
    }
}

static NPY_INLINE bool
levenshtein_score(PyObject *dummy, const char * const k1, 
                  const char * const k2, int * const out) 
{
    int tmp;
    
    if (!out)
        return false;
    
    if (*k1 != *k2)
        tmp = -1;
    else 
        tmp = 0;
    
    *out = tmp;
    
    return true;
}

static NPY_INLINE bool
alignment_score(PyObject *dok, const char * const k1, const char * const k2, 
            int * const out) 
{   
    PyObject *ptr, *key, *s1, *s2;
    Py_ssize_t one = 1;
    Py_ssize_t two = 2;
    int tmp;
    
    s1 = PyBytes_FromStringAndSize(k1, one);
    s2 = PyBytes_FromStringAndSize(k2, one);
    
    key = PyTuple_Pack(two, s1, s2);
    
    ptr = PyDict_GetItem(dok, key);
    
    if (ptr) {
        tmp = (int) PyLong_AsLong(ptr);
    } else {
        
        Py_DECREF(key);
        
        key = PyTuple_Pack(two, s2, s1);   
        
        ptr = PyDict_GetItem(dok, key);
        
        if (ptr)
            tmp = (int) PyLong_AsLong(ptr);
        else {
            Py_DECREF(s1);
            Py_DECREF(s2);
            Py_DECREF(key);
            return false;
        }
    }
    
    Py_DECREF(s1);
    Py_DECREF(s2);
    Py_DECREF(key);
    
    *out = tmp;
    
    return true;
}

static NPY_INLINE bool
blosum62_score(PyObject *dummy, const char * const k1, const char * const k2, 
               int * const out) 
{
    static const char amino_code[24] = {'*','A','B','C','D','E','F','G','H','I','K',
                                   'L','M','N','P','Q','R','S','T','V','W','X',
                                   'Y','Z'};
    static const int8_t blosum62[24][24] = {
        { 1, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4},
        {-4,  4, -2,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3,  0, -2, -1},
        {-4, -2,  4, -3,  4,  1, -3, -1,  0, -3,  0, -4, -3,  3, -2,  0, -1,  0, -1, -3, -4, -1, -3,  1},
        {-4,  0, -3,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2, -2, -3},
        {-4, -2,  4, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -1, -3,  1},
        {-4, -1,  1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -1, -2,  4},
        {-4, -2, -3, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1, -1,  3, -3},
        {-4,  0, -1, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -1, -3, -2},
        {-4, -2,  0, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2, -1,  2,  0},
        {-4, -1, -3, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1, -1, -3},
        {-4, -1,  0, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -1, -2,  1},
        {-4, -1, -4, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1, -1, -3},
        {-4, -1, -3, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1, -1, -1},
        {-4, -2,  3, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -1, -2,  0},
        {-4, -1, -2, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -2, -3, -1},
        {-4, -1,  0, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1, -1,  3},
        {-4, -1, -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -1, -2,  0},
        {-4,  1,  0, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3,  0, -2,  0},
        {-4,  0, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2,  0, -2, -1},
        {-4,  0, -3, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1, -1, -2},
        {-4, -3, -4, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, -2,  2, -3},
        {-4,  0, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1,  0,  0, -1, -2, -1, -1, -1},
        {-4, -2, -3, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2, -1,  7, -2},
        {-4, -1,  1, -3,  1, 4, - 3, -2,  0, -3,  1, -3, -1,  0, -1,  3,  0,  0, -1, -2, -3, -1, -2,  4}};  
    
    int i, j;
    
    if (binary_search(24, amino_code, *k1, &i)) {
        if (*k1 != *k2) {
            if (binary_search(24, amino_code, *k2, &j)) {
                *out = blosum62[i][j];
                return true;
            }
        } else {
            *out = blosum62[i][i];
            return true;     
        }
    }
    
    return false;
}

static NPY_INLINE bool 
get_max_score(const char *u, const char *v, const npy_intp nu, 
              const npy_intp nv, const scorePtr func, PyObject *submat, 
              int *out)
{
    int i, s1, s2, t;
    const char *c; 
    
    if (func == levenshtein_score) {
        *out = 0;
        return true;
    }
        
    s1 = 0;
    s2 = 0;
    
    c = u;
    
    for (i = 0; i < nu; i++) {
        if(func(submat, c, c, &t))
            s1 += t;
        else
            return false;
        c++;
    }
    
    c = v;
    
    for (i = 0; i < nv; i++) {
        if(func(submat, c, c, &t))
            s2 += t;
        else
            return false;
        c++;
    }
    
    if (s1 > s2)
        *out = s1;
    else
        *out = s2;
    
    return true;
}

/*O(n) SPACE ALIGNMENT*/

static NPY_INLINE bool 
affine(const char *A, const char *B, const npy_intp nA, const npy_intp nB, 
       const scorePtr func, PyObject *submat, const int gap_open, 
       const int gap_ext, int ** const graph, int * const score, 
       int * const length, int ** const middle, char * const backtrack) 
{  
    int i, j, index, v1, v2, v3, t;
    
    graph[0][0] = INT_MIN;
    graph[0][1] = -gap_open;
    graph[1][0] = INT_MIN;
    graph[2][0] = 0;
    graph[2][1] = -gap_open;
    graph[3][0] = -gap_open;
    graph[4][0] = INT_MIN;
    graph[4][1] = INT_MIN;
    
    graph[6][0] = 0;
    graph[6][1] = 1;
    graph[7][0] = 1;
    graph[8][0] = 0;
    graph[8][1] = 1;
    graph[9][0] = 1;
    graph[10][0] = 0;
    graph[10][1] = 1;
    
    for (j = 2; j <= nA; j++) {
        graph[0][j] = graph[0][j - 1] - gap_ext;
        graph[2][j] = graph[0][j];
        graph[4][j] = INT_MIN;
        graph[6][j] = graph[6][j - 1] + 1;
        graph[8][j] = graph[6][j];
        graph[10][j] = graph[6][j];
    }
    
    index = 0;
    
    for (i = 1; i <= nB; i++) {
        
        index = 1 - index;
        
        if (i > 1) {
            graph[4 + index][0] = graph[5 - index][0] - gap_ext;
            graph[2 + index][0] = graph[4 + index][0];
            graph[10 + index][0] = graph[11 - index][0] + 1;
            graph[8 + index][0] = graph[10 + index][0];
        }

        for (j = 1; j <= nA; j++) {
            // lower
            v1 = graph[2 + index][j-1] - gap_open;
            v2 = graph[index][j-1] - gap_ext;
            if (v1 > v2) {
                graph[index][j] = v1;
                graph[6 + index][j] = graph[8 + index][j-1] + 1;
            } else {
                graph[index][j] = v2;
                graph[6 + index][j] = graph[6 + index][j-1] + 1;
            }
            // upper
            v1 = graph[3 - index][j] - gap_open;
            v2 = graph[5 - index][j] - gap_ext;
            if (v1 > v2) {
                graph[4 + index][j] = v1;
                graph[10 + index][j] = graph[9 - index][j] + 1; 
            } else {
                graph[4 + index][j] = v2;
                graph[10 + index][j] = graph[11 - index][j] + 1; 
            }
            // middle 
            if (func(submat, A + j - 1, B + i - 1, &t))   
                v1 = graph[3 - index][j-1] + t;
            else
                return false;
            v2 = graph[index][j];
            v3 = graph[4 + index][j];
            if (v1 > v2 && v1 > v3) {
                graph[2 + index][j] = v1;
                graph[8 + index][j] = graph[9 - index][j-1] + 1; 
            } else if (v2 > v3) {
                graph[2 + index][j] = v2;
                graph[8 + index][j] = graph[6 + index][j]; 
            } else {
                graph[2 + index][j] = v3;
                graph[8 + index][j] = graph[10 + index][j]; 
            }
        } 
    }
    
    if (score != NULL)
        *score = graph[2 + index][nA];
    
    if (length != NULL)
        *length = graph[6 + index][nA];
    
    if (middle != NULL)
        *middle = graph[2 + index];
    
    if (backtrack != NULL) {    
        // calculate middle backtrack
        backtrack[0] = 'R';
        for (j = 1; j <= nA; j++){
            if (func(submat, A + j - 1, B + i - 1, &t)) {   
                if (graph[2 + index][j] == graph[3 - index][j-1] + t)
                    backtrack[j] = 'D';
                else if (graph[2 + index][j] == graph[4 + index][j])
                    backtrack[j] = 'R';
                else
                    backtrack[j] = 'D';
            } else
                return false;
        }
    }
    
    return true;
}

static NPY_INLINE double
normalized_alignment_distance(const char *u, const char *v, const npy_intp nu, 
                              const npy_intp nv, const scorePtr func, 
                              PyObject *submat, const int gap_open,
                              const int gap_ext, 
                              int ** const graph)
{
    int score;
    int length;
    int MAX;
    int * const p = &score;
    int * const q = &length;

    if(!affine(u, v, nu, nv, func, submat, gap_open, gap_ext, graph, p, q, 
               NULL, NULL))
        return -1.;
    
    if(!get_max_score(u, v, nu, nv, func, submat, &MAX))
        return -1.;
    
    return ((double) (MAX - score)) / length;
} 

static NPY_INLINE double
alignment_distance(const char *u, const char *v, const npy_intp nu, 
                   const npy_intp nv, const scorePtr func, PyObject *submat, 
                   const int gap_open, const int gap_ext, 
                   int ** const graph)
{
    int score;
    int MAX;
    int * const p = &score;
    
    if(!affine(u, v, nu, nv, func, submat, gap_open, gap_ext, graph, p, NULL, 
               NULL, NULL))
        return -1.;
    
    if(!get_max_score(u, v, nu, nv, func, submat, &MAX))
        return -1.;
    
    return (double) (MAX - score);
} 

#define DEFINE_ALIGN_PDIST(name)                                              \
static void pdist_ ## name(const PyBytesObject *X, const scorePtr func,       \
                           PyObject *submat, const int gap_open,              \
                           const int gap_ext, double *dm, const npy_intp m,   \
                           int * const buf, const npy_intp n)                 \
    {                                                                         \
        npy_intp i, j;                                                        \
        const PyBytesObject *u, *v;                                           \
        int ** const graph = malloc(12 * sizeof(int *));                      \
                                                                              \
        graph[0] = buf;                                                       \
        for (i = 1; i < 12; i++)                                              \
            graph[i] = graph[i-1] + n;                                        \
                                                                              \
        u = X + 1;                                                            \
                                                                              \
        for (i = 1; i < m; i++) {                                             \
            v = X;                                                            \
            for (j = 0; j < i; j++) {                                         \
                *dm = name ## _distance(PyBytes_AS_STRING(u),                 \
                                        PyBytes_AS_STRING(v),                 \
                                        PyBytes_GET_SIZE(u),                  \
                                        PyBytes_GET_SIZE(v), func, submat,    \
                                        gap_open, gap_ext, graph);            \
                v++;                                                          \
                dm++;                                                         \
            }                                                                 \
            u++;                                                              \
        }                                                                     \
                                                                              \
        free(graph);                                                          \
    }                                                                         \
            
DEFINE_ALIGN_PDIST(alignment)
DEFINE_ALIGN_PDIST(normalized_alignment)
            
#define DEFINE_ALIGN_CDIST(name)                                              \
    static void cdist_ ## name(const PyBytesObject *XA,                       \
                               const PyBytesObject *XB,                       \
                               const scorePtr func, PyObject *submat,         \
                               const int gap_open, const int gap_ext,         \
                               double *dm, const npy_intp mA,                 \
                               const npy_intp mB, int * const buf,            \
                               const npy_intp n)                              \
    {                                                                         \
        npy_intp i, j;                                                        \
        const PyBytesObject *u, *v;                                           \
        int ** const graph = malloc(12 * sizeof(int *));                      \
                                                                              \
        graph[0] = buf;                                                       \
        for (i = 1; i < 12; i++)                                              \
            graph[i] = graph[i-1] + n;                                        \
                                                                              \
        u = XA;                                                               \
                                                                              \
        for (i = 0; i < mA; i++) {                                            \
            v = XB;                                                           \
            for (j = 0; j < mB; j++) {                                        \
                *dm = name ## _distance(PyBytes_AS_STRING(u),                 \
                                        PyBytes_AS_STRING(v),                 \
                                        PyBytes_GET_SIZE(u),                  \
                                        PyBytes_GET_SIZE(v), func, submat,    \
                                        gap_open, gap_ext, graph);            \
                v++;                                                          \
                dm++;                                                         \
            }                                                                 \
            u++;                                                              \
        }                                                                     \
                                                                              \
        free(graph);                                                          \
    }                                                                         \

DEFINE_ALIGN_CDIST(alignment)
DEFINE_ALIGN_CDIST(normalized_alignment)
            
#define DEFINE_ALIGN_SSDIST(name)                                             \
    static void ssdist_ ## name(const PyBytesObject *XA,                      \
                               const PyBytesObject *XB,                       \
                               const npy_intp *indicesA,                      \
                               const npy_intp *indicesB,                      \
                               const npy_intp *indptr, const scorePtr func,   \
                                PyObject *submat, const int gap_open,         \
                               const int gap_ext, double *ssdm,               \
                               const npy_intp mA, int * const buf,            \
                               const npy_intp n)                              \
    {                                                                         \
        npy_intp i, j;                                                        \
        const PyBytesObject *u, *v;                                           \
        int ** const graph = malloc(12 * sizeof(int *));                      \
                                                                              \
        graph[0] = buf;                                                       \
        for (i = 1; i < 12; i++)                                              \
            graph[i] = graph[i-1] + n;                                        \
                                                                              \
        for (i = 0; i < mA; i++) {                                            \
            u = XA + indicesA[i];                                             \
            for (j = *indptr; j < *(indptr + 1); j++) {                       \
                v = XB + indicesB[j];                                         \
                *ssdm = name ## _distance(PyBytes_AS_STRING(u),               \
                                          PyBytes_AS_STRING(v),               \
                                          PyBytes_GET_SIZE(u),                \
                                          PyBytes_GET_SIZE(v), func, submat,  \
                                          gap_open, gap_ext, graph);          \
                ssdm++;                                                       \
            }                                                                 \
            indptr++;                                                         \
        }                                                                     \
                                                                              \
        free(graph);                                                          \
    }                                                                         \
                
DEFINE_ALIGN_SSDIST(alignment)
DEFINE_ALIGN_SSDIST(normalized_alignment)

static NPY_INLINE int *align_buffer(const npy_intp n) {
    int *buf;
    buf = calloc(n, 12 * sizeof(int));
    if (!buf) {
        PyErr_Format(PyExc_MemoryError, "could not allocate %zd * %zd bytes",
                     n, 12 * sizeof(int));
    }
    return buf;
}