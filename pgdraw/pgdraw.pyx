# cython: language_level=3
# distutils: language = c++
# distutils: extra_compile_args = -O3
# distutils: extra_link_args = -O3
# TODO: own gamma sampler
# TODO: polyagamma moments
# TODO: pass rng
# TODO: docstrings

import numpy as np
cimport numpy as np
from libc.math cimport exp, pi
np.import_array()

# Define constants
cdef double PI2 = np.pi * np.pi

def pgdraw_(np.ndarray[double, ndim=1] nv,
           np.ndarray[double, ndim=1] cv,
           int K=2):
    cdef int i, k
    cdef int n_samples = nv.shape[0]
    cdef np.ndarray[double, ndim=1] x = np.zeros(n_samples, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] c = np.abs(cv)
    cdef double a, b, nb, muK, varK, ec, ecp1, ecm1, mu, var

    for i in range(n_samples):
        muK = 0
        varK = 0

        # Truncated gamma series
        for k in range(1, K+1):
            # Compute a and b
            a = nv[i]
            b = (1/2/PI2) * 1 / ( (k - 0.5)*(k - 0.5) + c[i]*c[i]/(4*PI2) )
            x[i] += b * np.random.gamma(a)

            # Accumulate mu/var
            nb = nv[i]*b
            muK  += nb
            varK += nb*b

        # Now sample a gamma to approximate the remainder of the series
        # First determine the mean and variance of PG(n, c) variate
        # Handle small/large c appriately
        if c[i] <= 1e-3:
            mu = nv[i] / 4
            var = nv[i] / 24
        elif c[i] >= 300:
            mu = nv[i]/c[i]/2
            var = nv[i]/(c[i]*c[i]*c[i])/2
        else:
            ec = exp(c[i])
            ecp1 = ec + 1
            ecm1 = ec - 1
            mu = (ecm1) / (ecp1) * nv[i]/2/c[i]
            var = nv[i] * ((ec*ec - 2*c[i]*ec - 1)) / (2*(ecp1)*(ecp1)) / (c[i]**3)

        # Adjust mean and variance to account for truncated series
        mu = mu - muK
        var = var - varK

        b = var / mu
        a = mu*mu/var

        # Sample and add final gamma term
        x[i] += b * np.random.gamma(a)

    return x

# TODO: how much does this impact performance?
# cython does not use full cross-product of type combinations
# https://docs.cython.org/en/latest/src/userguide/fusedtypes.html#fused-types-and-arrays
ctypedef fused scalar_or_array:
    #int
    double
    np.ndarray[double, ndim=1]
ctypedef fused scalar_or_array2:
    #int
    double
    np.ndarray[double, ndim=1]

def pgdraw(scalar_or_array n, scalar_or_array2 c, int K=2):
    # TODO: do this better
    cdef np.ndarray[double, ndim=1] nv = np.atleast_1d(n).astype(float)
    cdef np.ndarray[double, ndim=1] cv = np.atleast_1d(c).astype(float)
    return_scalar = False
    if scalar_or_array is double and scalar_or_array2 is double:
        return_scalar = True
    elif scalar_or_array is double:
        nv = np.ones(cv.shape[0]) * n
    elif scalar_or_array2 is double:
        cv = np.ones(nv.shape[0]) * c

    x = pgdraw_(nv, cv, K)

    if return_scalar:
        return x[0]
    else:
        return x

