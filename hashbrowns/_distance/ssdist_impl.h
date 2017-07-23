/**
 * Author: Dezmond Goff
 * Date:   September 22, 2007 (moved to new file on June 8, 2008)
 * Adapted for use with Numpy/Scipy
 *
 * Copyright (c) 2017, Dezmond Goff. All rights reserved.
 *
 */

/** ssdist */

#include "distance_impl.h"

static NPY_INLINE void
ssdist_mahalanobis(const double *XA, const double *XB, const npy_intp *indicesA,
                   const npy_intp *indicesB, const npy_intp *indptr, 
                   const double *covinv, double *dimbuf, double *ssdm, 
                   npy_intp mA, npy_intp n) {
    npy_intp i, j;
    const double *u, *v;

    double *dimbuf1 = dimbuf;
    double *dimbuf2 = dimbuf + n;

    for (i = 0; i < mA; i++) {
        u = XA + (n * indicesA[i]);
        for (j = *indptr; j < *(indptr + 1); j++) {
            v = XB + (n * indicesB[j]);
            *ssdm = mahalanobis_distance(u, v, covinv, dimbuf1, dimbuf2, n);
            ssdm++;
        }
        indptr++;
    }
}

static NPY_INLINE void
ssdist_seuclidean(const double *XA, const double *XB, const npy_intp *indicesA,
                  const npy_intp *indicesB, const npy_intp *indptr, 
                  const double *var, double *ssdm, npy_intp mA, npy_intp n) {
    npy_intp i, j;
    const double *u, *v;

    for (i = 0; i < mA; i++) {
        u = XA + n * indicesA[i];
        for (j = *indptr; j < *(indptr + 1); j++) {
            v = XB + n * indicesB[j];
            *ssdm = seuclidean_distance(var, u, v, n);
            ssdm++;
        }
        indptr++;
    }
}

static NPY_INLINE void
ssdist_minkowski(const double *XA, const double *XB, const npy_intp *indicesA,
                 const npy_intp *indicesB, const npy_intp *indptr, 
                 double *ssdm, npy_intp mA, npy_intp n, double p) {
    npy_intp i, j;
    const double *u, *v;

    for (i = 0; i < mA; i++) {
        u = XA + n * indicesA[i];
        for (j = *indptr; j < *(indptr + 1); j++) {
            v = XB + n * indicesB[j];
            *ssdm = minkowski_distance(u, v, n, p);
            ssdm++;
        }
        indptr++;
    }
}

static NPY_INLINE void
ssdist_weighted_minkowski(const double *XA, const double *XB, 
                          const npy_intp *indicesA, const npy_intp *indicesB, 
                          const npy_intp *indptr, double *ssdm, npy_intp mA, 
                          npy_intp n, double p, const double *w) {
    npy_intp i, j;
    const double *u, *v;

    for (i = 0; i < mA; i++) {
        u = XA + n * indicesA[i];
        for (j = *indptr; j < *(indptr + 1); j++) {
            v = XB + n * indicesB[j];
            *ssdm = weighted_minkowski_distance(u, v, n, p, w);
            ssdm++;
        }
        indptr++;
    }
}

#define DEFINE_SSDIST(name, type)                                             \
    static void ssdist_ ## name ## _ ## type(const type *XA,                  \
                                             const type *XB,                  \
                                             const npy_intp *indicesA,        \
                                             const npy_intp *indicesB,        \
                                             const npy_intp *indptr,          \
                                             double *ssdm,                    \
                                             npy_intp mA,                     \
                                             npy_intp n)                      \
    {                                                                         \
        npy_intp i, j;                                                        \
        const type *u;                                                        \
        const type *v;                                                        \
                                                                              \
        for (i = 0; i < mA; i++) {                                            \
            u = XA + n * indicesA[i];                                         \
            for (j = *indptr; j < *(indptr + 1); j++) {                       \
                v = XB + n * indicesB[j];                                     \
                *ssdm = name ## _distance_ ## type(u, v, n);                  \
                ssdm++;                                                       \
            }                                                                 \
            indptr++;                                                         \
        }                                                                     \
    }
            

DEFINE_SSDIST(bray_curtis, double)
DEFINE_SSDIST(canberra, double)
DEFINE_SSDIST(chebyshev, double)
DEFINE_SSDIST(city_block, double)
DEFINE_SSDIST(euclidean, double)
DEFINE_SSDIST(hamming, double)
DEFINE_SSDIST(jaccard, double)
DEFINE_SSDIST(sqeuclidean, double)

DEFINE_SSDIST(dice, char)
DEFINE_SSDIST(hamming, char)
DEFINE_SSDIST(jaccard, char)
DEFINE_SSDIST(kulsinski, char)
DEFINE_SSDIST(rogerstanimoto, char)
DEFINE_SSDIST(russellrao, char)
DEFINE_SSDIST(sokalmichener, char)
DEFINE_SSDIST(sokalsneath, char)
DEFINE_SSDIST(yule_bool, char)