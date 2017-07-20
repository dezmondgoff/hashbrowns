# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

import numpy as np

cdef double stable_non_gauss_scalar(double alpha, double beta, double c, 
                                    double mu) nogil:
    cdef double u, xi, w, t, out
    
    u = (random() - 0.5) * M_PI
    
    if alpha == 1.:
        xi = M_PI_2
        if beta != 0.:
            w = -log(random())
            out = 1/xi*(xi+beta*u)*tan(u)-beta*log(xi*w*cos(u)/(xi+beta*u))
        else:
            out = tan(u)
    else:
        inv_alpha = 1 / alpha
        w = -log(random())
        if beta == 1.:
            xi = M_PI_2
            zeta = tan(xi * alpha)
            t = (1 + zeta**2)**(1 / (2 * alpha))
        else:
            zeta = beta * tan(M_PI_2 * alpha)
            xi = inv_alpha * atan(zeta)
            t = (1 + zeta**2)**(1 / (2 * alpha))
        out = t * sin(alpha * (u + xi)) / (cos(u)**inv_alpha)
        out *= (cos(u - alpha * (u + xi)) / w)**(inv_alpha - 1)    
    if c != 1.:
        out *= c
        if alpha == 1.:
            out += 2/M_PI*beta*c*log(c)
    if mu != 0.:
        out += mu
    return out

cdef void stable_non_gauss_vector(int n, double alpha, double beta, double c, 
                                  double mu, double[::1] out) nogil:
    cdef int i
    cdef double u, xi, w, t
    
    if alpha == 1:
        xi = M_PI_2
        if beta != 0:
            for i in range(n):
                u = (random() - 0.5)*M_PI
                w = -log(random())
                out[i] = (xi+beta*u)*tan(u)-beta*log(xi*w*cos(u)/(xi+beta*u))
                out[i] *= 1/xi
        else:
            for i in range(n):
                out[i] = tan((random() - 0.5)*M_PI)
    else:
        inv_alpha = 1/alpha
        inv_alpha_minus_one = inv_alpha - 1
        if beta == 1:
            xi = M_PI_2
            zeta = tan(xi*alpha)
        else:
            zeta = beta*tan(M_PI_2*alpha)
            xi = inv_alpha*atan(zeta)
        t = (1 + zeta**2)**(1/(2*alpha))
        for i in range(n):
            u = (random() - 0.5)*M_PI
            w = -log(random())
            out[i] = t*sin(alpha*(u+xi))/(cos(u)**inv_alpha)
            out[i] *= (cos(u-alpha*(u+xi))/w)**(inv_alpha_minus_one)    
    if c != 1:
        for i in range(n):
            out[i] *= c
        if alpha == 1:
            t = 2/M_PI*beta*c*log(c)
            for i in range(n):
                out[i] += t
    if mu != 0:
        for i in range(n):
            out[i] += mu

def random_stable(double alpha, double beta=0, double c=1 , double mu=0, 
                  shape=None):
    if alpha <= 0 or alpha > 2:
        raise ValueError("Stability parameter must be on (0,2].")
    if beta > 1 or beta < -1:
        raise ValueError("Skewness parameter must be on [-1,1].")
    if c <= 0:
        raise ValueError("Scale parameter must be greater than zero.")
    if alpha == 2:
        return np.random.normal(mu, 2*c**2, shape=shape)
    if shape is None:
        return stable_non_gauss_scalar(alpha, beta, c, mu)
    elif shape == 1:
        return stable_non_gauss_scalar(alpha, beta, c, mu)
    n = np.prod(shape)
    out = np.empty(n)
    stable_non_gauss_vector(n, alpha, beta, c, mu, out)
    return out.reshape(shape)