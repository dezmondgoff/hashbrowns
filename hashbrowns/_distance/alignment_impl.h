#ifndef ALIGNMENT_IMPL_INCLUDED

#define ALIGNMENT_IMPL_INCLUDED

#define _BLANK do{;} while(0)

#define _JOIN2(a, b) a ## b
#define JOIN2(a, b) _JOIN2(a, b)

#define _JOIN3(a, b, c) a ## b ## c
#define JOIN3(a, b, c) _JOIN3(a, b, c)

#define _JOIN4(a, b, c, d) a ## b ## c ## d
#define JOIN4(a, b, c, d) _JOIN4(a, b, c, d)

#define _JOIN5(a, b, c, d, e) a ## b ## c ## d ## e
#define JOIN5(a, b, c, d, e) _JOIN5(a, b, c, d, e)

#define _JOIN6(a, b, c, d, e, f) a ## b ## c ## d ## e ## f
#define JOIN6(a, b, c, d, e, f) _JOIN6(a, b, c, d, e, f)

#define PRE_HYPHEN(name) JOIN2(_, name)
#define POST_HYPHEN(name) JOIN2(name, _)

#define ADD_TYPED_ARG(boolean, type, name) 									  \
	JOIN2(ADD_TYPED_ARG_, boolean)(type, name)
#define ADD_TYPED_ARG_1(type, name) type name,
#define ADD_TYPED_ARG_0(type, name)

#define ADD_ARG(boolean, name) JOIN2(ADD_ARG_, boolean)(name)
#define ADD_ARG_1(name) name,
#define ADD_ARG_0(name)

#define ADD_TOKEN(boolean, name) JOIN2(ADD_TOKEN_, boolean)(name)
#define ADD_TOKEN_1(name) name
#define ADD_TOKEN_0(name)
#define ADD_TOKEN_str(name) name
#define ADD_TOKEN_PyObject(name)

#define ADD_COMMA(boolean) JOIN2(ADD_COMMA_, boolean)
#define ADD_COMMA_1 ,
#define ADD_COMMA_0
#define ADD_COMMA_str ,
#define ADD_COMMA_PyObject

#define ARRAY_TYPE(array_type) JOIN2(ARRAY_TYPE_, array_type)
#define ARRAY_TYPE_str uint32_t *
#define ARRAY_TYPE_PyObject PyObject **

#define ARRAY_TYPE_ALIAS(array_type) JOIN2(ARRAY_TYPE_ALIAS_, array_type)
#define ARRAY_TYPE_ALIAS_str uint32_t
#define ARRAY_TYPE_ALIAS_PyObject char

#define GET_STRING(array_type, name) JOIN2(GET_STRING_, array_type)(name)
#define GET_STRING_str(name) name
#define GET_STRING_PyObject(name) (char *) PyUnicode_1BYTE_DATA(*name)

#define GET_LENGTH(array_type, name, len) 								  	  \
	JOIN2(GET_LENGTH_, array_type)(name, len)
#define GET_LENGTH_str(name, len) (npy_intp) u32_strnlen(name, (size_t) len)
#define GET_LENGTH_PyObject(name, len) (npy_intp) PyUnicode_GET_LENGTH(*name)

#define UPDATE_PTR(array_type, name, update) 								  \
	JOIN2(UPDATE_PTR_, array_type)(name, update)
#define UPDATE_PTR_str(name, update) name += update
#define UPDATE_PTR_PyObject(name, update) name++

#define PY_FUNC(name) JOIN2(PY_FUNC_, name)
#define PY_FUNC_long PyLong_AS_LONG
#define PY_FUNC_double PyFloat_AS_DOUBLE

#define UPPER(name) JOIN2(UPPER_, name)
#define UPPER_double DOUBLE
#define UPPER_long LONG

#define LOWER(name) JOIN2(LOWER_, name)
#define LOWER_PyObject pyobject
#define LOWER_str str

#define NO_MEMORY -1

#define ZERO(type, array, len) JOIN2(ZERO_, type)(array, len)
#define ZERO_double(array, len) for (int k = 0; k < len; k++) array[k] = 0
#define ZERO_long(array, len) memset(array, 0, len * sizeof(long))

typedef int (*scoreptr) (const void *, const void *, void *);
typedef int (*scoreptr_dict) (PyObject*, const void *, const void *, void *);

/* SCORING FUNCTIONS*/

static NPY_INLINE
bool binary_search(const int n, const char *a, const uint8_t k,
		npy_intp * out)
{
    npy_intp el = 0;
    npy_intp r = n - 1;
    npy_intp m;

    while (true) {
        if (el > r)
            return false;
        m = (el + r) / 2;
        if (a[m] < k)
            el = m + 1;
        else if (a[m] > k)
            r = m - 1;
        else {
            *out = m;
            return true;
        }
    }
}

#define LEVENSHTEIN_SCORE(array_type)	\
static NPY_INLINE int	\
JOIN2(array_type, _levenshtein_score) (const void *k1, \
		const void *k2, void *out)\
{\
    if (* (array_type *) k1 != * (array_type *) k2)\
        * (int8_t *) out = -1;\
    else\
        * (int8_t *) out = 0;\
\
    return 0;\
}

LEVENSHTEIN_SCORE(char)
LEVENSHTEIN_SCORE(uint32_t)

#define NW_SCORE(array_type, score_type)								  	  \
static NPY_INLINE int														  \
JOIN3(array_type, PRE_HYPHEN(score_type), _needleman_wunsch_score) (		  \
		PyObject *d, const void *k1, const void *k2, void *out)			  \
{																			  \
	PyObject *key, *value;													  \
																			  \
    key = PyUnicode_New(2, 127);											  \
    PyUnicode_WriteChar(key, 0, (Py_UCS4) * (array_type *) k1);				  \
    PyUnicode_WriteChar(key, 1, (Py_UCS4) * (array_type *) k2);				  \
    value = PyDict_GetItem(d, key);											  \
    																		  \
    if (value) {															  \
    	* (score_type *) out = PY_FUNC(score_type)(value);					  \
    } else {																  \
        Py_DECREF(key);														  \
        key = PyUnicode_New(2, 127);										  \
        PyUnicode_WriteChar(key, 1, (Py_UCS4) * (array_type *) k1);			  \
    	PyUnicode_WriteChar(key, 0, (Py_UCS4) * (array_type *) k2);			  \
		value = PyDict_GetItem(d, key);										  \
        if (value) {														  \
            * (score_type *) out = PY_FUNC(score_type)(value);				  \
        } else {															  \
            return -((* (array_type*) k1 << 8) + * (array_type*) k2);		  \
        }																	  \
    }																		  \
																			  \
    Py_DECREF(key);															  \
    return 0;																  \
}

NW_SCORE(char, long)
NW_SCORE(char, double)
NW_SCORE(uint32_t, long)
NW_SCORE(uint32_t, double)

#define LEXICON										  						  \
{'*','A','B','C','D','E','F','G','H','I','K', 'L','M','N','P','Q','R','S',	  \
	'T','V','W','X','Y','Z'}
#define LEXICON_SIZE 24

#define _MATRIX(name) JOIN2(MATRIX_, name)
#define MATRIX(name) _MATRIX(name)
#define MATRIX_blosum62 													  \
{\
	{ 1, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4},\
	{-4,  4, -2,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3,  0, -2, -1},\
	{-4, -2,  4, -3,  4,  1, -3, -1,  0, -3,  0, -4, -3,  3, -2,  0, -1,  0, -1, -3, -4, -1, -3,  1},\
	{-4,  0, -3,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2, -2, -3},\
	{-4, -2,  4, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -1, -3,  1},\
	{-4, -1,  1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -1, -2,  4},\
	{-4, -2, -3, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1, -1,  3, -3},\
	{-4,  0, -1, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -1, -3, -2},\
	{-4, -2,  0, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2, -1,  2,  0},\
	{-4, -1, -3, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1, -1, -3},\
	{-4, -1,  0, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -1, -2,  1},\
	{-4, -1, -4, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1, -1, -3},\
	{-4, -1, -3, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1, -1, -1},\
	{-4, -2,  3, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -1, -2,  0},\
	{-4, -1, -2, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -2, -3, -1},\
	{-4, -1,  0, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1, -1,  3},\
	{-4, -1, -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -1, -2,  0},\
	{-4,  1,  0, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3,  0, -2,  0},\
	{-4,  0, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2,  0, -2, -1},\
	{-4,  0, -3, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1, -1, -2},\
	{-4, -3, -4, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, -2,  2, -3},\
	{-4,  0, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1,  0,  0, -1, -2, -1, -1, -1},\
	{-4, -2, -3, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2, -1,  7, -2},\
	{-4, -1,  1, -3,  1, 4, - 3, -2,  0, -3,  1, -3, -1,  0, -1,  3,  0,  0, -1, -2, -3, -1, -2,  4}\
}
#define MATRIX_pam250													  	  \
{\
	{1, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8},\
	{-8, 2, 0, -2, 0, 0, -3, 1, -1, -1, -1, -2, -1, 0, 1, 0, -2, 1, 1, 0, -6, 0, -3, 0},\
	{-8, 0, 3, -4, 3, 3, -4, 0, 1, -2, 1, -3, -2, 2, -1, 1, -1, 0, 0, -2, -5, -1, -3, 2},\
	{-8, -2, -4, 12, -5, -5, -4, -3, -3, -2, -5, -6, -5, -4, -3, -5, -4, 0, -2, -2, -8, -3, 0, -5},\
	{-8, 0, 3, -5, 4, 3, -6, 1, 1, -2, 0, -4, -3, 2, -1, 2, -1, 0, 0, -2, -7, -1, -4, 3},\
	{-8, 0, 3, -5, 3, 4, -5, 0, 1, -2, 0, -3, -2, 1, -1, 2, -1, 0, 0, -2, -7, -1, -4, 3},\
	{-8, -3, -4, -4, -6, -5, 9, -5, -2, 1, -5, 2, 0, -3, -5, -5, -4, -3, -3, -1, 0, -2, 7, -5},\
	{-8, 1, 0, -3, 1, 0, -5, 5, -2, -3, -2, -4, -3, 0, 0, -1, -3, 1, 0, -1, -7, -1, -5, 0},\
	{-8, -1, 1, -3, 1, 1, -2, -2, 6, -2, 0, -2, -2, 2, 0, 3, 2, -1, -1, -2, -3, -1, 0, 2},\
	{-8, -1, -2, -2, -2, -2, 1, -3, -2, 5, -2, 2, 2, -2, -2, -2, -2, -1, 0, 4, -5, -1, -1, -2},\
	{-8, -1, 1, -5, 0, 0, -5, -2, 0, -2, 5, -3, 0, 1, -1, 1, 3, 0, 0, -2, -3, -1, -4, 0},\
	{-8, -2, -3, -6, -4, -3, 2, -4, -2, 2, -3, 6, 4, -3, -3, -2, -3, -3, -2, 2, -2, -1, -1, -3},\
	{-8, -1, -2, -5, -3, -2, 0, -3, -2, 2, 0, 4, 6, -2, -2, -1, 0, -2, -1, 2, -4, -1, -2, -2},\
	{-8, 0, 2, -4, 2, 1, -3, 0, 2, -2, 1, -3, -2, 2, 0, 1, 0, 1, 0, -2, -4, 0, -2, 1},\
	{-8, 1, -1, -3, -1, -1, -5, 0, 0, -2, -1, -3, -2, 0, 6, 0, 0, 1, 0, -1, -6, -1, -5, 0},\
	{-8, 0, 1, -5, 2, 2, -5, -1, 3, -2, 1, -2, -1, 1, 0, 4, 1, -1, -1, -2, -5, -1, -4, 3},\
	{-8, -2, -1, -4, -1, -1, -4, -3, 2, -2, 3, -3, 0, 0, 0, 1, 6, 0, -1, -2, 2, -1, -4, 0},\
	{-8, 1, 0, 0, 0, 0, -3, 1, -1, -1, 0, -3, -2, 1, 1, -1, 0, 2, 1, -1, -2, 0, -3, 0},\
	{-8, 1, 0, -2, 0, 0, -3, 0, -1, 0, 0, -2, -1, 0, 0, -1, -1, 1, 3, 0, -5, 0, -3, -1},\
	{-8, 0, -2, -2, -2, -2, -1, -1, -2, 4, -2, 2, 2, -2, -1, -2, -2, -1, 0, 4, -6, -1, -2, -2},\
	{-8, -6, -5, -8, -7, -7, 0, -7, -3, -5, -3, -2, -4, -4, -6, -5, 2, -2, -5, -6, 17, -4, 0, -6},\
	{-8, 0, -1, -3, -1, -1, -2, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -4, -1, -2, -1},\
	{-8, -3, -3, 0, -4, -4, 7, -5, 0, -1, -4, -1, -2, -2, -5, -4, -4, -3, -3, -2, 0, -2, 10, -4},\
	{-8, 0, 2, -5, 3, 3, -5, 0, 2, -2, 0, -3, -2, 1, 0, 3, 0, 0, -1, -2, -6, -1, -4, 3}\
}

#define PREDEFINED_MATRIX_SCORE(name, array_type)					  		  \
static NPY_INLINE int 														  \
JOIN3(array_type, PRE_HYPHEN(name), _score) (								  \
		const void *k1, const void *k2, void *out)							  \
{																			  \
	static const char lexicon[LEXICON_SIZE] = LEXICON;						  \
    static const int8_t matrix[LEXICON_SIZE][LEXICON_SIZE] = MATRIX(name);	  \
    																		  \
    npy_intp i, j;															  \
    																		  \
    if (binary_search(LEXICON_SIZE, lexicon, 								  \
    				  (char) * (array_type *) k1, &i)) {					  \
        if (* (array_type *) k1 != * (array_type *) k2) {					  \
            if (binary_search(LEXICON_SIZE, lexicon, 						  \
            				  (char) * (array_type *) k2, &j)) 				  \
			{																  \
            	* (int8_t *) out = matrix[i][j];							  \
                return 0;													  \
            }																  \
            else {														  	  \
            	return -(int) * (array_type *) k2;							  \
			}																  \
        } else {															  \
        	* (int8_t *) out = matrix[i][i];								  \
            return 0;														  \
        }																	  \
    }																		  \
    else {																  	  \
    	return -(int) * (array_type *) k1;									  \
	}																		  \
}

PREDEFINED_MATRIX_SCORE(blosum62, char)
PREDEFINED_MATRIX_SCORE(blosum62, uint32_t)
PREDEFINED_MATRIX_SCORE(pam250, char)
PREDEFINED_MATRIX_SCORE(pam250, uint32_t)

#define GET_MAX_SCORE(array_type, score_type, matrix_type, has_dict)		  \
static NPY_INLINE int														  \
JOIN4(array_type, PRE_HYPHEN(score_type), _get_max_score, 					  \
	  ADD_TOKEN(has_dict, _dict)) (				  	  						  \
				const array_type * u, const array_type * v, 				  \
				const npy_intp nu, const npy_intp nv, 						  \
				const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func, 	  \
				ADD_TYPED_ARG(has_dict, PyObject*, submat)					  \
				score_type * const out)		  	  							  \
{																			  \
	npy_intp i;																  \
	int status;																  \
	const array_type *c;													  \
	score_type s1, s2;														  \
	matrix_type t;															  \
																			  \
	if (func == char_levenshtein_score ||									  \
			func == uint32_t_levenshtein_score) {							  \
		*out = 0;															  \
		return 0;															  \
	}																		  \
																			  \
	s1 = 0;																  	  \
    c = u;																	  \
    for (i = 0; i < nu; i++) {												  \
        if((status = func(ADD_ARG(has_dict, submat) (void *) c, 		 	  \
        				  (void *) c, (void *) &t)							  \
        		) < 0)				  										  \
			return status;													  \
		s1 += t;													 		  \
        c++;																  \
    }																		  \
																			  \
	s2 = 0;																	  \
    c = v;																	  \
    for (i = 0; i < nv; i++) {												  \
        if((status = func(ADD_ARG(has_dict, submat) (void *) c, 			  \
        				  (void *) c, (void *) &t)							  \
        		) < 0)				  										  \
            return status;													  \
		s2 += t;															  \
        c++;																  \
    }																		  \
																			  \
	*out = s1 > s2 ? s1 : s2;        										  \
    return 0;																  \
}

GET_MAX_SCORE(char, long, int8_t, false)
GET_MAX_SCORE(uint32_t, long, int8_t, false)
GET_MAX_SCORE(char, long, long, true)
GET_MAX_SCORE(char, double, double, true)
GET_MAX_SCORE(uint32_t, long, long, true)
GET_MAX_SCORE(uint32_t, double, double, true)

#define LONG_MIN (long) INT_MIN
#define DOUBLE_MIN -INFINITY

/*O(n) SPACE ALIGNMENT*/

#define NW_ALIGN(array_type, score_type, matrix_type, has_dict)				  \
static NPY_INLINE int JOIN4(array_type, PRE_HYPHEN(score_type), _nw, 		  \
		ADD_TOKEN(has_dict, _dict)) (	  									  \
				const array_type *a, const array_type *b, 					  \
				const npy_intp na, const npy_intp nb, 						  \
				const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func,	  	  \
				ADD_TYPED_ARG(has_dict, PyObject*, submat) 					  \
				const matrix_type gap_open, const matrix_type gap_ext, 		  \
				score_type * const lo, score_type * const mid, 			  	  \
				score_type * const up, score_type * const score, 			  \
				score_type ** const middle, char * const backtrack)	  		  \
{																			  \
	npy_intp i, j, curr_index, prev_index, tmp_index;						  \
    int status;																  \
	score_type v1, v2, v3;													  \
	matrix_type s;															  \
																			  \
	ZERO(score_type, lo, 2 * (na + 1));									  	  \
  	ZERO(score_type, mid, 2 * (na + 1));									  \
  	ZERO(score_type, up, 2 * (na + 1));									  	  \
																			  \
	lo[0] = JOIN2(UPPER(score_type), _MIN);									  \
    lo[1] = -gap_open;														  \
    lo[na + 1] = lo[0];											  	  		  \
																			  \
    mid[0] = 0;																  \
    mid[1] = -gap_open;														  \
    mid[na + 1] = -gap_open;												  \
																			  \
    up[0] = lo[0];													  		  \
    up[1] = lo[0];													  		  \
    up[na + 1] = -gap_open;													  \
    																		  \
    for (j = 2; j <= na; j++) {												  \
        lo[j] = lo[j - 1] - gap_ext;										  \
		mid[j] = lo[j];														  \
		up[j] = lo[0];												  		  \
    }																		  \
																			  \
	curr_index = 0;															  \
	prev_index = na + 1;													  \
    for (i = 1; i <= nb; i++) {												  \
    	tmp_index = curr_index;												  \
		curr_index = prev_index;											  \
		prev_index = tmp_index;												  \
																			  \
        if (i > 1) {														  \
        	up[curr_index] = up[prev_index] - gap_ext;						  \
			mid[curr_index] = up[curr_index];								  \
        }																	  \
																			  \
        for (j = 1; j <= na; j++) {											  \
			v1 = mid[curr_index + j - 1] - gap_open;						  \
			v2 = lo[curr_index + j - 1] - gap_ext;							  \
            lo[curr_index + j] = v1 > v2 ? v1 : v2;						 	  \
            																  \
			v1 = mid[prev_index + j] - gap_open;							  \
            v2 = up[prev_index + j] - gap_ext;								  \
            up[curr_index + j] = v1 > v2 ? v1 : v2;				 	  	  	  \
            																  \
            if ((status = func(ADD_ARG(has_dict, submat) a + j - 1, 		  \
						 	   b + i - 1, &s)) < 0)	  						  \
				return status;												  \
			v1 = mid[prev_index + j - 1] + s;							  	  \
            v2 = lo[curr_index + j];										  \
            v3 = up[curr_index + j];										  \
            mid[curr_index + j] = ((v1 > v2) && (v1 > v3)) ? v1 : 			  \
           									(v2 > v3 ? v2 : v3);			  \
        }																	  \
	}																		  \
																			  \
    if (score != NULL)														  \
        *score = mid[curr_index + na];									  	  \
    if (middle != NULL)														  \
        *middle = mid + curr_index;											  \
    if (backtrack != NULL) {												  \
        backtrack[0] = 'R';													  \
        for (j = 1; j <= na; j++){										  	  \
            if ((status = func(ADD_ARG(has_dict, submat) a + j - 1, 		  \
            				   b + i - 1, &s)) < 0) 	  					  \
				return status;												  \
            if (mid[curr_index + j] == mid[prev_index + j - 1] + s)		  	  \
            	backtrack[j] = 'C';										  	  \
            else if (mid[curr_index + j] == up[curr_index + j])		  	  	  \
                backtrack[j] = 'R';										  	  \
            else														  	  \
                backtrack[j] = 'D';										  	  \
        }																	  \
    }																		  \
    return 0;																  \
}

NW_ALIGN(char, long, int8_t, false)
NW_ALIGN(uint32_t, long, int8_t, false)
NW_ALIGN(char, long, long, true)
NW_ALIGN(char, double, double, true)
NW_ALIGN(uint32_t, long, long, true)
NW_ALIGN(uint32_t, double, double, true)

#define NW_ALIGN_NORM(array_type, score_type, matrix_type, has_dict)		  \
static NPY_INLINE int JOIN4(array_type, PRE_HYPHEN(score_type), _nw_norm, 	  \
		ADD_TOKEN(has_dict, _dict)) (	  									  \
				const array_type *a, const array_type *b, 					  \
				const npy_intp na, const npy_intp nb, 					 	  \
				const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func,	  	  \
				ADD_TYPED_ARG(has_dict, PyObject*, submat) 					  \
				const double lambda, const matrix_type gap_open, 			  \
				const matrix_type gap_ext, double * const nlo, 				  \
				double * const nmid, double * const nup, 					  \
				score_type * const slo, score_type * const smid, 			  \
				score_type * const sup, long * const llo, 					  \
				long * const lmid, long * const lup, 						  \
				score_type * const score, long * length, 				 	  \
				score_type ** const middle, char * const backtrack)	  		  \
{																			  \
	npy_intp i, j, curr_index, prev_index, tmp_index;						  \
    int status;																  \
	double v1, v2, v3;													  	  \
	matrix_type s;															  \
																			  \
	/* clear buffers */														  \
	for (int k = 0; k < 2 * (na + 1); k++) {								  \
		nlo[k] = 0;															  \
		nmid[k] = 0;														  \
		nup[k] = 0;															  \
	}																		  \
																			  \
	ZERO(score_type, slo, 2 * (na + 1));									  \
	ZERO(score_type, smid, 2 * (na + 1));									  \
	ZERO(score_type, sup, 2 * (na + 1));									  \
																			  \
	memset(llo, 0, 2 * (na + 1) * sizeof(long));							  \
	memset(lmid, 0, 2 * (na + 1) * sizeof(long));							  \
	memset(lup, 0, 2 * (na + 1) * sizeof(long));							  \
																			  \
	/* initalize buffers*/													  \
	nlo[0] = -INFINITY;													  	  \
    nlo[1] = -gap_open;											  			  \
    nlo[na + 1] = -INFINITY;											  	  \
																			  \
    nmid[0] = 0;															  \
    nmid[1] = -gap_open;											  		  \
    nmid[na + 1] = -gap_open;										  	  	  \
																			  \
    nup[0] = -INFINITY;													  	  \
    nup[1] = -INFINITY;													  	  \
    nup[na + 1] = -gap_open;										 		  \
																			  \
	slo[0] = JOIN2(UPPER(score_type), _MIN);  								  \
    slo[1] = -gap_open;										  				  \
    slo[na + 1] = JOIN2(UPPER(score_type), _MIN);							  \
																			  \
    smid[0] = 0;															  \
    smid[1] = -gap_open;										  			  \
    smid[na + 1] = -gap_open;									  			  \
																			  \
    sup[0] = JOIN2(UPPER(score_type), _MIN);								  \
    sup[1] = JOIN2(UPPER(score_type), _MIN);								  \
    sup[na + 1] = -gap_open;									  			  \
																			  \
	llo[0] = 0;													  			  \
    llo[1] = 1;														  		  \
    llo[na + 1] = 0;											  	  		  \
																			  \
    lmid[0] = 0;															  \
    lmid[1] = 1;														  	  \
    lmid[na + 1] = 1;												      	  \
																			  \
    lup[0] = 0;													  			  \
    lup[1] = 0;													  			  \
    lup[na + 1] = 1;													  	  \
    																		  \
    for (j = 2; j <= na; j++) {												  \
        nlo[j] = nlo[j - 1] - gap_ext;								  		  \
		nmid[j] = nlo[j];													  \
		nup[j] = -INFINITY;													  \
																			  \
		slo[j] = slo[j - 1] - gap_ext;										  \
		smid[j] = slo[j];													  \
		sup[j] = JOIN2(UPPER(score_type), _MIN);							  \
																			  \
		llo[j] = llo[j - 1] + 1;											  \
		lmid[j] = llo[j];													  \
		lup[j] = 0;															  \
    }																		  \
																			  \
	curr_index = 0;															  \
	prev_index = na + 1;													  \
																		 	  \
    for (i = 1; i <= nb; i++) {												  \
    	tmp_index = curr_index;												  \
		curr_index = prev_index;											  \
		prev_index = tmp_index;												  \
																			  \
        if (i > 1) {														  \
        	nup[curr_index] = nup[prev_index] - gap_ext;			  		  \
			nmid[curr_index] = nup[curr_index];								  \
																			  \
        	sup[curr_index] = sup[prev_index] - gap_ext;					  \
			smid[curr_index] = sup[curr_index];								  \
																			  \
        	lup[curr_index] = lup[prev_index] + 1;						  	  \
			lmid[curr_index] = lup[curr_index];								  \
        }																	  \
																			  \
        for (j = 1; j <= na; j++) {											  \
			v1 = nmid[curr_index + j - 1] - gap_open;						  \
			v2 = nlo[curr_index + j - 1] - gap_ext;							  \
																			  \
			if (v1 > v2) {													  \
				nlo[curr_index + j] = v1;									  \
				slo[curr_index + j] = smid[curr_index + j - 1] - gap_open;	  \
				llo[curr_index + j] = lmid[curr_index + j - 1] + 1;			  \
			} else {														  \
				nlo[curr_index + j] = v2;									  \
				slo[curr_index + j] = slo[curr_index + j - 1] - gap_ext;	  \
				llo[curr_index + j] = llo[curr_index + j - 1] + 1;			  \
			}																  \
            																  \
			v1 = nmid[prev_index + j] - gap_open;					  		  \
            v2 = nup[prev_index + j] - gap_ext;					  			  \
																			  \
			if (v1 > v2) {													  \
				nup[curr_index + j] = v1;									  \
				sup[curr_index + j] = smid[prev_index + j] - gap_open;		  \
				lup[curr_index + j] = lmid[prev_index + j] + 1;				  \
			} else {														  \
				nup[curr_index + j] = v2;									  \
				sup[curr_index + j] = sup[prev_index + j] - gap_ext;		  \
				lup[curr_index + j] = lup[prev_index + j] + 1;				  \
			}																  \
            											  					  \
            if ((status = func(ADD_ARG(has_dict, submat) a + j - 1, 		  \
						 	   b + i - 1, &s)) < 0)	  						  \
				return status;												  \
																			  \
			v1 = nmid[prev_index + j - 1] + s - lambda;			  			  \
            v2 = nlo[curr_index + j];										  \
            v3 = nup[curr_index + j];										  \
																			  \
            if ((v1 > v2) && (v1 > v3)) {									  \
            	nmid[curr_index + j] = v1;									  \
            	smid[curr_index + j] = smid[prev_index + j - 1] + s;		  \
            	lmid[curr_index + j] = lmid[prev_index + j - 1] + 1;		  \
            } else if (v2 > v3) {											  \
            	nmid[curr_index + j] = v2;									  \
            	smid[curr_index + j] = slo[curr_index + j];					  \
            	lmid[curr_index + j] = llo[curr_index + j];					  \
            } else {														  \
            	nmid[curr_index + j] = v3;									  \
            	smid[curr_index + j] = sup[curr_index + j];					  \
            	lmid[curr_index + j] = lup[curr_index + j];					  \
            }			  													  \
        }																	  \
	}																		  \
																			  \
    if (score != NULL)														  \
        *score = smid[curr_index + na];										  \
	if (length != NULL)												  		  \
		*length = lmid[curr_index + na];								  	  \
    if (middle != NULL)														  \
        *middle = smid + curr_index;									  	  \
    if (backtrack != NULL) {												  \
        backtrack[0] = 'R';													  \
        for (j = 1; j <= na; j++){										  	  \
            if ((status = func(ADD_ARG(has_dict, submat) a + j - 1, 		  \
            				   b + i - 1, &s)) < 0) 	  					  \
				return status;												  \
            if (smid[curr_index + j] == smid[prev_index + j - 1] + s)		  \
            	backtrack[j] = 'C';										  	  \
            else if (smid[curr_index + j] == sup[curr_index + j])		  	  \
                backtrack[j] = 'R';										  	  \
            else														  	  \
                backtrack[j] = 'D';										  	  \
        }																	  \
    }																		  \
    return 0;																  \
}

NW_ALIGN_NORM(char, long, int8_t, false)
NW_ALIGN_NORM(uint32_t, long, int8_t, false)
NW_ALIGN_NORM(char, long, long, true)
NW_ALIGN_NORM(char, double, double, true)
NW_ALIGN_NORM(uint32_t, long, long, true)
NW_ALIGN_NORM(uint32_t, double, double, true)

#define NW_ALIGN_NORM_INIT(array_type, score_type, matrix_type, has_dict)	  \
static NPY_INLINE int JOIN4(array_type, PRE_HYPHEN(score_type), 			  \
		_nw_norm_init, ADD_TOKEN(has_dict, _dict)) (	  					  \
				const array_type *a, const array_type *b, 					  \
				const npy_intp na, const npy_intp nb, 						  \
				const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func,	  	  \
				ADD_TYPED_ARG(has_dict, PyObject*, submat) 				  	  \
				const matrix_type gap_open, 								  \
				const matrix_type gap_ext, 									  \
				score_type * const slo, score_type * const smid, 			  \
				score_type * const sup, long * const llo, 					  \
				long * const lmid, long * const lup,						  \
				score_type * score, long * len)	  					  		  \
{																			  \
	npy_intp i, j, curr_index, prev_index, tmp_index;						  \
    int status;																  \
	score_type w1, w2, w3;													  \
	long el1, el2, el3; 													  \
	double r1, r2, r3;														  \
	matrix_type s;															  \
																			  \
	ZERO(score_type, slo, 2 * (na + 1));									  \
	ZERO(score_type, smid, 2 * (na + 1));									  \
 	ZERO(score_type, sup, 2 * (na + 1));									  \
																			  \
	memset(llo, 0, 2 * (na + 1) * sizeof(long));							  \
	memset(lmid, 0, 2 * (na + 1) * sizeof(long));							  \
	memset(lup, 0, 2 * (na + 1) * sizeof(long));							  \
															  				  \
	slo[0] = JOIN2(UPPER(score_type), _MIN);								  \
    slo[1] = -gap_open;														  \
    slo[na + 1] = slo[0];											  	  	  \
																			  \
    smid[0] = 0;															  \
    smid[1] = -gap_open;													  \
    smid[na + 1] = -gap_open;												  \
																			  \
    sup[0] = slo[0];													  	  \
    sup[1] = slo[0];													  	  \
    sup[na + 1] = -gap_open;												  \
																			  \
	llo[0] = 0;													  			  \
    llo[1] = 1;														  		  \
    llo[na + 1] = 0;											  	  		  \
																			  \
    lmid[0] = 0;															  \
    lmid[1] = 1;														  	  \
    lmid[na + 1] = 1;												  		  \
																			  \
    lup[0] = 0;													  			  \
    lup[1] = 0;													  			  \
    lup[na + 1] = 1;														  \
    																		  \
    for (j = 2; j <= na; j++) {												  \
		slo[j] = slo[j - 1] - gap_ext;										  \
		smid[j] = slo[j];													  \
		sup[j] = slo[0];													  \
																			  \
		llo[j] = llo[j - 1] + 1;											  \
		lmid[j] = llo[j];													  \
		lup[j] = 0;															  \
    }																		  \
																			  \
	curr_index = 0;															  \
	prev_index = na + 1;													  \
    for (i = 1; i <= nb; i++) {												  \
    	tmp_index = curr_index;												  \
		curr_index = prev_index;											  \
		prev_index = tmp_index;												  \
																			  \
        if (i > 1) {														  \
        	sup[curr_index] = sup[prev_index] - gap_ext;					  \
			smid[curr_index] = sup[curr_index];								  \
																			  \
        	lup[curr_index] = lup[prev_index] + 1;						  	  \
			lmid[curr_index] = lup[curr_index];								  \
        }																	  \
																			  \
        for (j = 1; j <= na; j++) {											  \
    		w1 = smid[curr_index + j - 1] - gap_open;						  \
    		el1 = lmid[curr_index + j - 1] + 1;								  \
																			  \
			w2 = slo[curr_index + j - 1] - gap_ext;							  \
			el2 = llo[curr_index + j - 1] + 1;								  \
																			  \
			r1 = (double) w1 / el1;											  \
			r2 = (double) w2 / el2;											  \
																			  \
			if (r1 > r2) {													  \
				slo[curr_index + j] = w1;									  \
				llo[curr_index + j] = el1;									  \
			} else {														  \
				slo[curr_index + j] = w2;									  \
				llo[curr_index + j] = el2;									  \
			}																  \
            																  \
			w1 = smid[prev_index + j] - gap_open;							  \
			el1 = lmid[prev_index + j] + 1;									  \
																			  \
            w2 = sup[prev_index + j] - gap_ext;								  \
            el2 = lup[prev_index + j] + 1;									  \
																			  \
            r1 = (double) w1 / el1;											  \
            r2 = (double) w2 / el2;											  \
																			  \
			if (r1 > r2) {													  \
				sup[curr_index + j] = w1;									  \
				lup[curr_index + j] = el1;									  \
			} else {														  \
				sup[curr_index + j] = w2;									  \
				lup[curr_index + j] = el2;									  \
			}																  \
																			  \
            if ((status = func(ADD_ARG(has_dict, submat) a + j - 1, 		  \
						 	   b + i - 1, &s)) < 0)	  						  \
				return status;												  \
																			  \
			w1 = smid[prev_index + j - 1] + s;						  		  \
            el1 = lmid[prev_index + j - 1] + 1;								  \
																			  \
			w2 = slo[curr_index + j];										  \
			el2 = llo[curr_index + j];										  \
																			  \
            w3 = sup[curr_index + j];										  \
            el3 = lup[curr_index + j];										  \
																			  \
            r1 = (double) w1 / el1;											  \
            r2 = (double) w2 / el2;											  \
            r3 = (double) w3 / el3;											  \
																			  \
            if ((r1 > r2) && (r1 > r3)) {									  \
            	smid[curr_index + j] = w1;									  \
            	lmid[curr_index + j] = el1;									  \
            } else if (r2 > r3) {											  \
            	smid[curr_index + j] = w2;									  \
            	lmid[curr_index + j] = el2;									  \
            } else {														  \
            	smid[curr_index + j] = w3;									  \
            	lmid[curr_index + j] = el3;									  \
            }			  													  \
        }																	  \
	}																		  \
																			  \
	if (score != NULL) 														  \
		*score = smid[curr_index + na];										  \
	if (len != NULL) 														  \
		*len = lmid[curr_index + na]; 										  \
    return 0;																  \
}

NW_ALIGN_NORM_INIT(char, long, int8_t, false)
NW_ALIGN_NORM_INIT(uint32_t, long, int8_t, false)
NW_ALIGN_NORM_INIT(char, long, long, true)
NW_ALIGN_NORM_INIT(char, double, double, true)
NW_ALIGN_NORM_INIT(uint32_t, long, long, true)
NW_ALIGN_NORM_INIT(uint32_t, double, double, true)

#define ALIGN_DIST(array_type, score_type, matrix_type, has_dict)		  	  \
static NPY_INLINE int 														  \
JOIN4(array_type, PRE_HYPHEN(score_type), _align_dist, 						  \
	  ADD_TOKEN(has_dict, _dict)) (			  				  				  \
				const array_type *a, const array_type *b, 					  \
				const npy_intp na, const npy_intp nb, 						  \
				const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func, 	  \
				ADD_TYPED_ARG(has_dict, PyObject*, submat) 					  \
				const matrix_type gap_open, const matrix_type gap_ext, 		  \
				score_type * const lo, score_type * const mid, 			  	  \
				score_type * const up, double * const out)					  \
{																			  \
    int status;																  \
	score_type score, max_score;										  	  \
    																		  \
	if ((status = JOIN4(array_type, PRE_HYPHEN(score_type), 				  \
					   _get_max_score, ADD_TOKEN(has_dict, _dict)) (		  \
							   a, b, na, nb, func, 							  \
							   ADD_ARG(has_dict, submat) 					  \
							   &max_score)				  					  \
			) < 0)	  	  	  										  		  \
		return	status;													 	  \
	if ((status = JOIN4(array_type, PRE_HYPHEN(score_type), _nw, 			  \
						ADD_TOKEN(has_dict, _dict)) (						  \
							a, b, na, nb, func, ADD_ARG(has_dict, submat) 	  \
							gap_open, gap_ext, lo, mid, up, &score, NULL, 	  \
							NULL)											  \
			) < 0) 	  			  	  	  									  \
		return status;												  		  \
																			  \
	*out = (double) (max_score - score);  					  		  		  \
	return 0;																  \
}

ALIGN_DIST(char, long, int8_t, false)
ALIGN_DIST(uint32_t, long, int8_t, false)
ALIGN_DIST(char, long, long, true)
ALIGN_DIST(char, double, double, true)
ALIGN_DIST(uint32_t, long, long, true)
ALIGN_DIST(uint32_t, double, double, true)

#define ALIGN_DIST_NORM(array_type, score_type, matrix_type, has_dict)	  	  \
static NPY_INLINE int 														  \
JOIN4(array_type, PRE_HYPHEN(score_type), _align_dist_norm, 				  \
	  ADD_TOKEN(has_dict, _dict)) (			  								  \
				const array_type *a, const array_type *b, 					  \
				const npy_intp na, const npy_intp nb, 						  \
				const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func, 	  \
				ADD_TYPED_ARG(has_dict, PyObject*, submat) 					  \
				const matrix_type gap_open, const matrix_type gap_ext, 		  \
				double * const nlo, double * const nmid, 					  \
				double * const nup, score_type * const slo, 				  \
				score_type * const smid, score_type * const sup, 			  \
				long * const llo, long *const lmid, long * const lup,		  \
				double * const out, double tol)				  				  \
{																			  \
    int status;																  \
	score_type score, max_score;										  	  \
	long len;																  \
	double old_lambda, lambda;										  		  \
																			  \
	if ((status = JOIN4(array_type, PRE_HYPHEN(score_type), 				  \
						_get_max_score, ADD_TOKEN(has_dict, _dict)) (		  \
								a, b, na, nb, func, 						  \
								ADD_ARG(has_dict, submat) 					  \
								&max_score)				  					  \
			) < 0)	  	  	  										  		  \
		return	status;														  \
	if ((status = JOIN4(array_type, PRE_HYPHEN(score_type), 				  \
						_nw_norm_init, ADD_TOKEN(has_dict, _dict)) (	  	  \
							a, b, na, nb, func, ADD_ARG(has_dict, submat) 	  \
							gap_open, gap_ext, slo, smid, sup, 	  			  \
							llo, lmid, lup, &score, &len)					  \
			) < 0)															  \
		return status;														  \
																			  \
	lambda = ((double) (max_score - score)) / len;							  \
																			  \
	while (true) {															  \
		old_lambda = lambda;												  \
		if((status = JOIN4(array_type, PRE_HYPHEN(score_type), _nw_norm, 	  \
				ADD_TOKEN(has_dict, _dict)) 		  						  \
					(a, b, na, nb, func, ADD_ARG(has_dict, submat) 			  \
					old_lambda, gap_open, gap_ext, nlo, nmid, nup,	  		  \
					slo, smid, sup, llo, lmid, lup, &score, &len, 			  \
					NULL, NULL)) < 0) 	  			  	  	  				  \
			return status;												  	  \
																			  \
		lambda = ((double) (max_score - score)) / len;				  		  \
																			  \
		if (fabs(lambda - old_lambda) < tol) break;							  \
	}									  		  				  			  \
																			  \
	*out = lambda;  					  		  							  \
	return 0;																  \
}

ALIGN_DIST_NORM(char, long, int8_t, false)
ALIGN_DIST_NORM(uint32_t, long, int8_t, false)
ALIGN_DIST_NORM(char, long, long, true)
ALIGN_DIST_NORM(char, double, double, true)
ALIGN_DIST_NORM(uint32_t, long, long, true)
ALIGN_DIST_NORM(uint32_t, double, double, true)

#define BUF_INIT(prefix, type, var)											  \
type * prefix##buf = (type *) calloc(var + 1, 6 * sizeof(type));			  \
if (!prefix##buf)															  \
	return NO_MEMORY

#define BUF_PART(prefix, type, var)											  \
type * const prefix##lo = prefix##buf;							  			  \
type * const prefix##mid = prefix##lo + 2 * (var + 1);		  				  \
type * const prefix##up = prefix##mid + 2 * (var + 1)

#define DEFINE_ALIGN_PDIST(array_type, score_type, matrix_type, norm, 		  \
		has_dict)										  					  \
static int 																	  \
JOIN5(LOWER(array_type), PRE_HYPHEN(score_type), _align_pdist, 				  \
		ADD_TOKEN(norm, _norm), ADD_TOKEN(has_dict, _dict)) (	  	  		  \
				const ARRAY_TYPE(array_type)X, 								  \
				const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func, 	  \
				ADD_TYPED_ARG(has_dict, PyObject*, submat)  				  \
				const matrix_type gap_open, const matrix_type gap_ext, 		  \
				double *dm, const npy_intp m, const npy_intp n				  \
				ADD_COMMA(norm) ADD_TOKEN(norm, double tol))				  \
{                                                                         	  \
	npy_intp i, j;														  	  \
	int status;															  	  \
	const ARRAY_TYPE(array_type)u;										  	  \
	const ARRAY_TYPE(array_type)v;											  \
	BUF_INIT(s, score_type, n);												  \
	ADD_TOKEN(norm, BUF_INIT(n, double, n);)								  \
	ADD_TOKEN(norm, BUF_INIT(l, long, n);)									  \
	BUF_PART(s, score_type, n);												  \
	ADD_TOKEN(norm, BUF_PART(n, double, n);)								  \
	ADD_TOKEN(norm, BUF_PART(l, long, n);)									  \
																			  \
	u = X + 1 ADD_TOKEN(array_type, * n);                                 	  \
	for (i = 1; i < m; i++) {                                       	  	  \
		v = X;                                                        	  	  \
		for (j = 0; j < i; j++) {          								  	  \
			status = JOIN5(ARRAY_TYPE_ALIAS(array_type), 					  \
					PRE_HYPHEN(score_type), _align_dist, 					  \
					ADD_TOKEN(norm, _norm), 								  \
					ADD_TOKEN(has_dict, _dict)) ( 			  				  \
							GET_STRING(array_type, u), 				  		  \
							GET_STRING(array_type, v), 	  			  		  \
							GET_LENGTH(array_type, u, n), 				  	  \
							GET_LENGTH(array_type, v, n),  	  			  	  \
							func, ADD_ARG(has_dict, submat) gap_open, 	  	  \
							gap_ext, ADD_ARG(norm, nlo)  					  \
							ADD_ARG(norm, nmid) ADD_ARG(norm, nup)			  \
							slo, smid, sup, ADD_ARG(norm, llo)				  \
							ADD_ARG(norm, lmid)  ADD_ARG(norm, lup) dm		  \
							ADD_COMMA(norm) ADD_TOKEN(norm, tol));            \
			if (status < 0)												  	  \
				return status;										  	  	  \
			UPDATE_PTR(array_type, v, n);                                 	  \
			dm++;                                                         	  \
		}                                                                 	  \
		UPDATE_PTR(array_type, u, n);                                     	  \
	}																	  	  \
	return 0;														  	  	  \
}

DEFINE_ALIGN_PDIST(PyObject, long, int8_t, false, false)
DEFINE_ALIGN_PDIST(str, long, int8_t, false, false)
DEFINE_ALIGN_PDIST(PyObject, long, int8_t, true, false)
DEFINE_ALIGN_PDIST(str, long, int8_t, true, false)
DEFINE_ALIGN_PDIST(PyObject, long, long, false, true)
DEFINE_ALIGN_PDIST(str, long, long, false, true)
DEFINE_ALIGN_PDIST(PyObject, long, long, true, true)
DEFINE_ALIGN_PDIST(str, long, long, true, true)
DEFINE_ALIGN_PDIST(PyObject, double, double, false, true)
DEFINE_ALIGN_PDIST(str, double, double, false, true)
DEFINE_ALIGN_PDIST(PyObject, double, double, true, true)
DEFINE_ALIGN_PDIST(str, double, double, true, true)

#define DEFINE_ALIGN_CDIST(array_type, score_type, matrix_type, norm, 	  	  \
		has_dict)										  					  \
static int 																	  \
JOIN5(LOWER(array_type), PRE_HYPHEN(score_type), _align_cdist, 				  \
	  ADD_TOKEN(norm, _norm), ADD_TOKEN(has_dict, _dict)) (		  			  \
			  const ARRAY_TYPE(array_type)XA, 								  \
			  const ARRAY_TYPE(array_type)XB, 				  				  \
			  const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func, 	  	  \
			  ADD_TYPED_ARG(has_dict, PyObject *, submat) 					  \
			  const matrix_type gap_open, const matrix_type gap_ext, 		  \
			  double *dm, const npy_intp mA, const npy_intp mB, 			  \
			  const npy_intp nA ADD_COMMA(array_type)						  \
			  ADD_TOKEN(array_type, const npy_intp nB) 						  \
			  ADD_COMMA(norm) ADD_TOKEN(norm, double tol))					  \
{                                                                         	  \
    npy_intp i, j;                                                        	  \
    int status;															  	  \
    const ARRAY_TYPE(array_type)u;											  \
    const ARRAY_TYPE(array_type)v;										  	  \
	BUF_INIT(s, score_type, nA);											  \
	ADD_TOKEN(norm, BUF_INIT(n, double, nA);)								  \
	ADD_TOKEN(norm, BUF_INIT(l, long, nA);)									  \
	BUF_PART(s, score_type, nA);											  \
	ADD_TOKEN(norm, BUF_PART(n, double, nA);)								  \
	ADD_TOKEN(norm, BUF_PART(l, long, nA);)									  \
																		  	  \
	u = XA;                                                            	  	  \
    for (i = 0; i < mA; i++) {                                       	  	  \
        v = XB;                                                        	  	  \
        for (j = 0; j < mB; j++) {                                   	  	  \
            status = JOIN5(ARRAY_TYPE_ALIAS(array_type), 					  \
            		PRE_HYPHEN(score_type), _align_dist, 					  \
					ADD_TOKEN(norm, _norm), 								  \
					ADD_TOKEN(has_dict, _dict)) 			  	  			  \
						(GET_STRING(array_type, u), 						  \
                		 GET_STRING(array_type, v), 	  					  \
                		 GET_LENGTH(array_type, u, nA), 					  \
						 GET_LENGTH(array_type, v, nB),  	  				  \
						 func, ADD_ARG(has_dict, submat) gap_open, 	  		  \
						 gap_ext, ADD_ARG(norm, nlo)  						  \
						 ADD_ARG(norm, nmid) ADD_ARG(norm, nup)				  \
						 slo, smid, sup, ADD_ARG(norm, llo)					  \
						 ADD_ARG(norm, lmid)  ADD_ARG(norm, lup) dm			  \
						 ADD_COMMA(norm) ADD_TOKEN(norm, tol));            	  \
			if (status < 0)											  	  	  \
				return status;										  	  	  \
            UPDATE_PTR(array_type, v, nB);                                	  \
            dm++;                                                         	  \
        }                                                                 	  \
        UPDATE_PTR(array_type, u, nA);                                   	  \
    }                                                                     	  \
	return 0;														  	  	  \
}

DEFINE_ALIGN_CDIST(PyObject, long, int8_t, false, false)
DEFINE_ALIGN_CDIST(str, long, int8_t, false, false)
DEFINE_ALIGN_CDIST(PyObject, long, int8_t, true, false)
DEFINE_ALIGN_CDIST(str, long, int8_t, true, false)
DEFINE_ALIGN_CDIST(PyObject, long, long, false, true)
DEFINE_ALIGN_CDIST(str, long, long, false, true)
DEFINE_ALIGN_CDIST(PyObject, long, long, true, true)
DEFINE_ALIGN_CDIST(str, long, long, true, true)
DEFINE_ALIGN_CDIST(PyObject, double, double, false, true)
DEFINE_ALIGN_CDIST(str, double, double, false, true)
DEFINE_ALIGN_CDIST(PyObject, double, double, true, true)
DEFINE_ALIGN_CDIST(str, double, double, true, true)

#define DEFINE_ALIGN_SSDIST(array_type, score_type, matrix_type, norm, 		  \
		has_dict)										 					  \
static int 																	  \
JOIN5(LOWER(array_type), PRE_HYPHEN(score_type), _align_ssdist, 			  \
	  ADD_TOKEN(norm, _norm), ADD_TOKEN(has_dict, _dict)) (		  			  \
			  const ARRAY_TYPE(array_type)XA,								  \
			  const ARRAY_TYPE(array_type)XB, 				  				  \
			  const npy_intp *indicesA,									 	  \
			  const npy_intp *indicesB,										  \
			  const npy_intp *indptr,										  \
			  const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func, 	  	  \
			  ADD_TYPED_ARG(has_dict, PyObject *, submat) 					  \
			  const matrix_type gap_open, 		  							  \
			  const matrix_type gap_ext, 									  \
			  double *dm, 													  \
			  const npy_intp mA, 				  						 	  \
			  const npy_intp nA ADD_COMMA(array_type)						  \
			  ADD_TOKEN(array_type, const npy_intp nB) ADD_COMMA(norm)		  \
			  ADD_TOKEN(norm, double tol))							  		  \
    {                                                                         \
        npy_intp i, j;                                                        \
        int status;															  \
        const ARRAY_TYPE(array_type)u;										  \
		const ARRAY_TYPE(array_type)v;										  \
		BUF_INIT(s, score_type, nA);										  \
		ADD_TOKEN(norm, BUF_INIT(n, double, nA);)							  \
		ADD_TOKEN(norm, BUF_INIT(l, long, nA);)							 	  \
		BUF_PART(s, score_type, nA);										  \
		ADD_TOKEN(norm, BUF_PART(n, double, nA);)							  \
		ADD_TOKEN(norm, BUF_PART(l, long, nA);)  							  \
	  																	  	  \
		for (i = 0; i < mA; i++) {                                       	  \
			u = XA + indicesA[i] ADD_TOKEN(array_type, * nA);             	  \
			for (j = *indptr; j < *(indptr + 1); j++) {                  	  \
				v = XB + indicesB[j] ADD_TOKEN(array_type, * nB);             \
                status = JOIN5(ARRAY_TYPE_ALIAS(array_type), 				  \
                		PRE_HYPHEN(score_type), _align_dist, 				  \
						ADD_TOKEN(norm, _norm), 							  \
						ADD_TOKEN(has_dict, _dict)) (			  			  \
            					GET_STRING(array_type, u), 					  \
								GET_STRING(array_type, v), 	  				  \
								GET_LENGTH(array_type, u, nA), 				  \
								GET_LENGTH(array_type, v, nB),  	  		  \
								func, ADD_ARG(has_dict, submat) gap_open, 	  \
								gap_ext, ADD_ARG(norm, nlo)  				  \
								ADD_ARG(norm, nmid) ADD_ARG(norm, nup)		  \
								slo, smid, sup, ADD_ARG(norm, llo)			  \
								ADD_ARG(norm, lmid)  ADD_ARG(norm, lup) dm	  \
								ADD_COMMA(norm) ADD_TOKEN(norm, tol));        \
				if (status < 0)												  \
					return status;										  	  \
                dm++;                                                         \
            }                                                                 \
            indptr++;                                                     	  \
        }                                                                     \
		return 0;														  	  \
    }

DEFINE_ALIGN_SSDIST(PyObject, long, int8_t, false, false)
DEFINE_ALIGN_SSDIST(str, long, int8_t, false, false)
DEFINE_ALIGN_SSDIST(PyObject, long, int8_t, true, false)
DEFINE_ALIGN_SSDIST(str, long, int8_t, true, false)
DEFINE_ALIGN_SSDIST(PyObject, long, long, false, true)
DEFINE_ALIGN_SSDIST(str, long, long, false, true)
DEFINE_ALIGN_SSDIST(PyObject, long, long, true, true)
DEFINE_ALIGN_SSDIST(str, long, long, true, true)
DEFINE_ALIGN_SSDIST(PyObject, double, double, false, true)
DEFINE_ALIGN_SSDIST(str, double, double, false, true)
DEFINE_ALIGN_SSDIST(PyObject, double, double, true, true)
DEFINE_ALIGN_SSDIST(str, double, double, true, true)

#endif
