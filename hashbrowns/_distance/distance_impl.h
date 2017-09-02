/**
 * Author: Damian Eads
 * Date:   September 22, 2007 (moved to new file on June 8, 2008)
 *
 * Copyright (c) 2007, 2008, Damian Eads. All rights reserved.
 * Adapted for incorporation into Scipy, April 9, 2008.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   - Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the
 *     following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer
 *     in the documentation and/or other materials provided with the
 *     distribution.
 *   - Neither the name of the author nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

static NPY_INLINE double
sqeuclidean_distance_double(const double *u, const double *v, npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = u[i] - v[i];
        s += d * d;
    }
    return s;
}

static NPY_INLINE double
euclidean_distance_double(const double *u, const double *v, npy_intp n)
{
    return sqrt(sqeuclidean_distance_double(u, v, n));
}

#if 0   /* XXX unused */
static NPY_INLINE double
ess_distance(const double *u, const double *v, npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]);
        s += d * d;
    }
    return s;
}
#endif

static NPY_INLINE double
chebyshev_distance_double(const double *u, const double *v, npy_intp n)
{
    double d, maxv = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]);
        if (d > maxv) {
            maxv = d;
        }
    }
    return maxv;
}

static NPY_INLINE double
canberra_distance_double(const double *u, const double *v, npy_intp n)
{
    double tot = 0.;
    npy_intp i;

    for (i = 0; i < n; i++) {
        double x = u[i], y = v[i];
        double snum = fabs(x - y);
        double sdenom = fabs(x) + fabs(y);
        if (sdenom > 0.) {
            tot += snum / sdenom;
        }
    }
    return tot;
}

static NPY_INLINE double
bray_curtis_distance_double(const double *u, const double *v, npy_intp n)
{
    double s1 = 0.0, s2 = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        s1 += fabs(u[i] - v[i]);
        s2 += fabs(u[i] + v[i]);
    }
    return s1 / s2;
}

/*
 * Timings with various BLAS implementations (gh-5657) have shown that using
 * OpenBLAS's cblas_ddot here can give some speedup, but only for high-d data
 * and an untuned ATLAS is slower than rolling our own.
 */
static NPY_INLINE double
dot_product(const double *u, const double *v, npy_intp n)
{
    double s = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        s += u[i] * v[i];
    }
    return s;
}

static NPY_INLINE double
mahalanobis_distance(const double *u, const double *v, const double *covinv,
                     double *dimbuf1, double *dimbuf2, npy_intp n)
{
    npy_intp i;

    for (i = 0; i < n; i++) {
        dimbuf1[i] = u[i] - v[i];
    }
    /*
     * Note: matrix-vector multiplication (GEMV). Again, OpenBLAS can speed
     * this up for high-d data.
     */
    for (i = 0; i < n; i++) {
        const double *covrow = covinv + (i * n);
        dimbuf2[i] = dot_product(dimbuf1, covrow, n);
    }
    return sqrt(dot_product(dimbuf1, dimbuf2, n));
}

static NPY_INLINE double
hamming_distance_double(const double *u, const double *v, npy_intp n)
{
    npy_intp i, s = 0;

    for (i = 0; i < n; i++) {
        s += (u[i] != v[i]);
    }
    return (double)s / n;
}

static NPY_INLINE double
hamming_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i, s = 0;

    for (i = 0; i < n; i++) {
        s += (u[i] != v[i]);
    }
    return (double)s / n;
}

static NPY_INLINE double
yule_bool_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, nff = 0, nft = 0, ntf = 0;

    for (i = 0; i < n; i++) {
        npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y;
        ntf += x & (!y);
        nft += (!x) & y;
    }
    nff = n - ntt - ntf - nft;
    return (2. * ntf * nft) / ((double)ntt * nff + (double)ntf * nft);
}

static NPY_INLINE double
dice_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    for (i = 0; i < n; i++) {
        npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y;
        ndiff += (x != y);
    }
    return ndiff / (2. * ntt + ndiff);
}

static NPY_INLINE double
rogerstanimoto_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    for (i = 0; i < n; i++) {
        npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y;
        ndiff += (x != y);
    }
    return (2. * ndiff) / ((double)n + ndiff);
}

static NPY_INLINE double
russellrao_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0;

    for (i = 0; i < n; i++) {
        ntt += (u[i] != 0) & (v[i] != 0);
    }
    return (double)(n - ntt) / n;
}

static NPY_INLINE double
kulsinski_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    for (i = 0; i < n; i++) {
        npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y;
        ndiff += (x != y);
    }
    return ((double)ndiff - ntt + n) / ((double)ndiff + n);
}

static NPY_INLINE double
sokalsneath_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    for (i = 0; i < n; i++) {
        npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y;
        ndiff += (x != y);
    }
    return (2. * ndiff) / (2. * ndiff + ntt);
}

static NPY_INLINE double
sokalmichener_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp i;
    npy_intp ntt = 0, ndiff = 0;

    for (i = 0; i < n; i++) {
        npy_bool x = (u[i] != 0), y = (v[i] != 0);
        ntt += x & y;
        ndiff += (x != y);
    }
    return (2. * ndiff) / ((double)ndiff + n);
}

static NPY_INLINE double
jaccard_distance_double(const double *u, const double *v, npy_intp n)
{
    npy_intp denom = 0, num = 0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        double x = u[i], y = v[i];
        num += (x != y) & ((x != 0.0) | (y != 0.0));
        denom += (x != 0.0) | (y != 0.0);
    }
    return (double)num / denom;
}

static NPY_INLINE double
jaccard_distance_char(const char *u, const char *v, npy_intp n)
{
    npy_intp num = 0, denom = 0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        npy_bool x = (u[i] != 0), y = (v[i] != 0);
        num += (x != y);
        denom += x | y;
    }
    return (double)num / denom;
}

static NPY_INLINE double
seuclidean_distance(const double *var, const double *u, const double *v,
                    npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = u[i] - v[i];
        s += (d * d) / var[i];
    }
    return sqrt(s);
}

static NPY_INLINE double
city_block_distance_double(const double *u, const double *v, npy_intp n)
{
    double s = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        s += fabs(u[i] - v[i]);
    }
    return s;
}

static NPY_INLINE double
minkowski_distance(const double *u, const double *v, npy_intp n, double p)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]);
        s += pow(d, p);
    }
    return pow(s, 1.0 / p);
}

static NPY_INLINE double
weighted_minkowski_distance(const double *u, const double *v, npy_intp n,
                            double p, const double *w)
{
    npy_intp i = 0;
    double s = 0.0, d;
    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]) * w[i];
        s += pow(d, p);
    }
    return pow(s, 1.0 / p);
}

#if 0   /* XXX unused */
static void
compute_mean_vector(double *res, const double *X, npy_intp m, npy_intp n)
{
    npy_intp i, j;
    const double *v;
    for (i = 0; i < n; i++) {
        res[i] = 0.0;
    }
    for (j = 0; j < m; j++) {

        v = X + (j * n);
        for (i = 0; i < n; i++) {
            res[i] += v[i];
        }
    }
    for (i = 0; i < n; i++) {
        res[i] /= (double)m;
    }
}
#endif
