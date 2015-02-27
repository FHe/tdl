import numpy as Num
cimport numpy as Num
from libc.math cimport sin, cos, exp

FLOAT = Num.float
ctypedef Num.float_t FLOAT_t

cimport cython
@cython.boundscheck(False)
def calc_Fsurf(Num.ndarray[FLOAT_t, ndim=2] hkls, Num.ndarray[FLOAT_t, ndim=2] qs, Num.ndarray[FLOAT_t, ndim=2] atoms, Num.ndarray[FLOAT_t, ndim=2] fs):  
    assert hkls.dtype == FLOAT and qs.dtype == FLOAT and fs.dtype == FLOAT and atoms.dtype == FLOAT
    cdef int imax = hkls.shape[0]
    cdef int nmax = atoms.shape[0]
    cdef int domains = int(atoms[nmax-1,10])+1
    cdef int i, k, n
    cdef double pi = 3.141592653589793
    cdef double f, rq
    cdef FLOAT_t  re, im 
    cdef Num.ndarray[FLOAT_t, ndim=2] reF = Num.zeros((imax,domains),dtype=FLOAT)
    cdef Num.ndarray[FLOAT_t, ndim=2] imF = Num.zeros((imax,domains),dtype=FLOAT)
    for i in range(imax):
        k = 0
        for n in range(nmax):
            f = fs[i,n] * atoms[n,9] * exp(-2*pi**2*\
                (qs[i,0]*(qs[i,0]*atoms[n,3]+qs[i,1]*atoms[n,6]+qs[i,2]*atoms[n,7])+\
                 qs[i,1]*(qs[i,0]*atoms[n,6]+qs[i,1]*atoms[n,4]+qs[i,2]*atoms[n,8])+\
                 qs[i,2]*(qs[i,0]*atoms[n,7]+qs[i,1]*atoms[n,8]+qs[i,2]*atoms[n,5])))
            rq = 2*pi*(hkls[i,0]*atoms[n,0]+hkls[i,1]*atoms[n,1]+hkls[i,2]*atoms[n,2])
            re = f * cos(rq)
            im = f * sin(rq)
            if atoms[n,10] == k:
                reF[i,k] = re
                imF[i,k] = im
                k += 1
            else:
                reF[i,k-1] += re
                imF[i,k-1] += im
    return reF, imF 

cimport cython
@cython.boundscheck(False)
def calc_Fwater_layered(Num.ndarray[FLOAT_t, ndim=1] ls, FLOAT_t sig, FLOAT_t sig_bar, FLOAT_t d, FLOAT_t zwater, \
                        FLOAT_t Auc, Num.ndarray[FLOAT_t, ndim=1] fs ,Num.ndarray[FLOAT_t, ndim=1] qs):
    cdef int imax = ls.shape[0]
    cdef double pi = 3.141592653589793
    cdef double x, al, wert, f, rez, imz, relayer, imlayer
    cdef Num.ndarray[FLOAT_t, ndim=1] reFwater = Num.zeros((imax),dtype=FLOAT)
    cdef Num.ndarray[FLOAT_t, ndim=1] imFwater = Num.zeros((imax),dtype=FLOAT)
    for i in range(imax):
        f = Auc * d * 0.033456 * fs[i]* exp(-2 * pi**2 * qs[i]**2 * sig)
        x = pi * qs[i] * d
        al = 2 * pi**2 * qs[i]**2 * sig_bar
        wert = 2*pi*ls[i]*zwater
        rez = cos(wert)
        imz = sin(wert)
        wert = 1-2*exp(-al)*cos(2*x)+exp(-2*al)
        relayer = (1-exp(-al)*cos(2*x))/wert
        imlayer = exp(-al)*sin(2*x)/wert
        reFwater[i] = f* (relayer * rez - imlayer * imz)
        imFwater[i] = f* (relayer * imz + imlayer * rez)
    return reFwater, imFwater