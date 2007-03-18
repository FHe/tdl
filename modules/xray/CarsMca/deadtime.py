# T. Trainor fftpt@uaf.edu
# Deadtime calculations for CARS MED/MCA library
# T. Trainor, 6-10-2006
#
# --------------
# Modifications
# --------------
#
#
##################################################################################################

"""
Calculate detector deadtime correction factors:

The objective is to calculate a factor 'cor' that will provide
corrected counts via:
   counts_corrected = counts * cor

Background:
------------
A correction factor can be defined as:

     cor = (icr_t/ocr_s)*(rt/lt)

Here icr_t = true input count rate (actual counts hitting the detector
per detector live time - ie icr is a rate)

ocr_s is the output rate from detector slow channel.  ocr_s = TOC_s/lt, TOC_s
are the total counts output from the slow filter (~ the total counts output
by the detector).

rt and lt are the detector real time and live time respectively.

icr_t is an uknown quantity.  It can be determined or approximated in a few
different ways.

A) icr_t may be determined by inverting the following:

    ocr_f = icr_t * exp( -icr_t * t_f)

Here ocr_f is the fast count rate.  ocr_f =TOC_f/lt, TOC_f are the total counts
output from the fast filter.

t_f is the fast filter deadtime.  If the detector reports TOC_f and t_f is known
(from fit to a deatime curve) then icr_t may be calculated from the above.

Note: a detector/file may report InputCounts or ICR.
This is likely = ocr_f rather than icr_t.

Note: The fast filter deatime is much less substantial than the slow filter
deadtime.  Therefore for low total count rates the approximation icr_t ~ ocr_f
may be reasonable.

B) alternatively icr_t may be determined by inverting the following:

    ocr_s = icr_t * exp( -icr_t * t_s)

Here ocr_s is the slow count rate.  ocr_s =TOC_s/lt, TOC_s are the total counts
output from the slow filter.

t_s is the slow filter deadtime.  If the detector reports TOC_s and t_s is known
(from fit to a deatime curve) then icr_t may be calculated from the above.
If the detector does not report TOC_s ocr_s may be approximated from:

    ocr_s = ( Sum_i ( cnts_i) ) / lt ~ TOC_s/lt

Note the above discussion referes primarily to digital couting electronics,
using slow and fast filters. Analog electronics using an SCA may be corrected
using a similiar procedure.  Here the analog amplifier may
report total input counts.  In this case icr_t ~ icr_a (icr amplifier).  


Measuring t's:
--------------

To perform a deadtime correction the characteristic deadtimes must be determined.
In each case (t_f or t_s) a deadtime scan is run, where the input counts are
varied (eg scanning a slit) and output counts monitored.

If a measure of icr_t is reported (or approximated ie = ocr_f or icr_a), then
t_f may be deterimined directly from a fit of
  * x = icr_t
  * y = ocr_s

More likely a detector not suffering from deadtime will be used as a proxy for
icr_t (eg Io -> ion chamber)

In this case plot:
  * x = Io/rt
  * y = ocr_f   -> TOC_f/lt reported in file (may be called icr or total counts)
  * icr_t = a*Io/rt
  * fit varying t_f and a (prop const btwn Io and icr_t)
      ocr_f = icr_t*exp(-icr_t*t_f) 

If the detector does not report ocr_f, then use slow counts to correct:
  * x = Io/rt
  * y = ocr_s   -> TOC_s/lt ~ ( Sum_i ( cnts_i) ) / lt
  * icr_t = a*Io/rt
  * fit varying t_s and a (prop const btwn Io and icr_t)
         ocr_s = icr_t*exp(-icr_t*t_s)
    

Summary of correction methods:
------------------------------
To apply a deadtime correction then depends on what data can be accessed from
the detector.

1) detector reports actual icr_t (or reports ocr_f and total counts are low so
   ocr_f ~icr_t). Then the correction may be calculated directly from
   detector numbers

2) The detector reports ocr_f and t_f has been determined.  icr_t is calculated
   from the saturation equation to use in the correction.

3) The detector does not report ocr_f, rather ocr_s is reported (or approximated
   from total counts) and t_f has been determined.  icr_t is again calculated
   from the saturation equation and used for the correction.
   This is probably the most straightforward method since icr_s can be
   approximated directly from the sum of the counts in the detector (norm by lt),
   and this approach should work for analog and digital electronics.

4) icr_t is uknown or icr_t ~ ocr_s then just assume that icr_t/ocr_s = 1
   (ie just correct for livetime effects)

"""

#from Num import Num
import numpy as Num
import scipy
from scipy.optimize import leastsq

def correction_factor(rt,lt,icr = None,ocr = None):
    """
    Calculate the deadtime correction factor.
        cor = (icr/ocr)*(rt/lt)
        rt  = real time, time the detector was requested to count for
        lt  = live time, actual time the detector was active and
              processing counts
        icr = true input count rate (TOC_t/lt, where TOC_t = true total counts
              impinging the detector)
        ocr = output count rate (TOC_s/lt, where TOC_s = total processed
              {slow filter for dxp} counts output by the detecotr)  

        If icr and/or ocr are None then only lt correction is applied

        the correction is applied as:
            corrected_counts = counts * cor
    """
    if (ocr != None) and (icr != None):
        cor = (icr/ocr)*(rt/lt)
    else:
        cor = (rt/lt)

    #percent_dead_time = 100.0 * (1 - 1/cor)
    
    return cor

def correct_data(data,rt,lt,icr=None,ocr=None):
    """
    Apply deatime correction to data
    """
    cor = correction_factor(rt,lt,icr,ocr)
    return data * cor

def calc_icr(ocr,tau):
    """
    Calculate the true icr from a given ocr and corresponding deadtime factor
    tau using a Newton-Raphson algorithm to solve the following expression.  

        ocr = icr * exp(-icr*tau)    

    Returns None if the loop cannot converge
    
    Note below could be improved!
    """

    # error checks
    if ocr == None: return None
    if ocr <= 0: return None
    if tau == None: return None
    if tau <=0: return None

    
    # max_icr is icr val at top of deadtime curve
    # max_icr is the corresponding ocr value
    # we cannot correct the data if ocr > ocr_max
    max_icr = 1/tau
    max_ocr = max_icr*Num.exp(-1)
    if ocr >= Num.exp(-1)/tau:
        print 'ocr exceeds maximum correctible value of %g cps' % max_ocr
        return None

    # Newton loop
    x1 = ocr
    cnt = 0
    while (1):
        f  = ocr - x1*Num.exp(-x1*tau)
        df = Num.exp(-x1*tau) * (x1*tau - 1)
        x2 = x1 - f/df
        check = abs(  x2 - x1  )
        if ( check < 0.01) :
            icr = x1
            break
        else:
            x1 = x2
            if (x1 > max_icr):
                # went over the top, we assume that
                # the icr is less than 1/tau
                x1 = 1.1 * ocr
        if cnt > 100:
            print 'Warning: icr calculation failed to converge'
            icr = None
            break
        else:
            cnt = cnt + 1

    return icr


def fit_deadtime_curve(Io,ocr):
    """
    Fit a deatime curve and return optimized value of tau

    Io is an array from a linear detector.  This should be in counts/sec
    (ie Io/scaler_count_time)

    ocr is an array corresponding to the output count rate (either slow or fast).
    This should be in counts per second where
        ocr = TOC/lt,
    TOC is the total output counts and lt is the detector live time

    This fits the data to the following:
          x = Io
          y = ocr   -> TOC/lt
          icr_t = a*Io

     fit varying 'tau' and 'a' (prop const btwn Io and icr_t)
         ocr = icr_t*exp(-icr_t*tau)

    This function is appropriate for fitting either slow or fast ocr's.
    If Io is icr_t the the optimized 'a' should be ~1.

    Example:
        (params,msg) = fit_deadtime_curve(Io,ocr_meas)
        a = params[0]
        tau = params[1]

    """
    Io = Num.array(Io)
    ocr = Num.array(ocr)
    
    npts = len(Io)
    if len(ocr) != npts: return None

    # make a gues at a
    # assume that Io and ocr are arranged in ascending order, and the first 5%
    # have little deadtime effects (or maybe a bit so scale up the average 20%).
    idx = max(int(0.05*npts), 3)
    Io_avg  = Num.average(Io[0:idx])
    ocr_avg = Num.average(ocr[0:idx])
    a = 1.2*ocr_avg/Io_avg

    # make a guess at tau, assume max(ocr) is the top of the deadtime curve
    tau = 1/ (Num.exp(1) * max(ocr) )

    params = (a,tau)
    result = leastsq(deadtime_residual,params,args = (Io,ocr))

    return result


def calc_ocr(params,Io):
    a,tau = params
    icr = a*Io
    ocr = icr * Num.exp(-icr*tau)
    return ocr

def deadtime_residual(params,Io,ocr):
    ocr_calc = calc_ocr(params,Io)
    return ocr - ocr_calc


##########
if __name__ == '__main__':
    # test fit
    Io = 10000 * Num.arange(500.0)
    a = 0.1
    tau = 0.00001
    print 'a= ', a, ' tau= ', tau
    ocr = a*Io*Num.exp(-a*Io*tau)
    ocr_meas = ocr + 2*Num.randn(len(ocr))

    (params,msg) = fit_deadtime_curve(Io,ocr_meas)
    a = params[0]
    tau = params[1]
    #print msg
    print 'a_fit= ',a,' tau_fit=', tau

    ocr = 0.3 * 1/tau
    icr = calc_icr(ocr,tau)
    print 'max icr = ', 1/tau
    print 'max ocr = ', Num.exp(-1)/tau
    print 'ocr= ', ocr, ' icr_calc= ',icr

    rt = 1
    lt = 1
    cor = correction_factor(rt,lt,icr,ocr)
    print 'cor= ', cor
    