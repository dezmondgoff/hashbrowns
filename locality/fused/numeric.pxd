from cython cimport char, short, int, long, longlong, float, double, longdouble
from cython cimport floatcomplex, doublecomplex, longdoublecomplex

ctypedef fused numeric:
    char
    unsigned char
    short
    unsigned short
    int
    unsigned int
    long
    unsigned long
    float
    double
    long double
    float complex
    double complex
    long double complex
    
cdef int numeric_int_flag(numeric x)