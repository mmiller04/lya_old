'''Routines to produce robust modified tanh fits, particularly suited for pedestal/edge analysis.

sciortino,2021
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import interp1d
from scipy import optimize

def mtanh_profile(x_coord, edge=0.08, ped=0.4, core=2.5, expin=1.5, expout=1.5, widthp=0.04, xphalf=None):
    """
     This function generates H-mode-like  density and temperature profiles on the input x_coord grid.

    Parameters
    ----------
    x_coord : 1D array
        Radial coordinate of choice
    edge : float
        Separatrix height
    ped : float
        pedestal height
    core : float
        On-axis profile height
    expin : float
        Inner core exponent for H-mode pedestal profile
    expout : float
        Outer core exponent for H-mode pedestal profile
    widthp : float
        Width of pedestal
    xphalf : float
        Position of tanh

    Returns
    -------
    val : 1D array
        modified tanh profile

    Notes
    -----
    This function is inspired by an OMFIT function, somewhere in the framework (math_utils.py?).
    """

    w_E1 = 0.5 * widthp  # width as defined in eped
    if xphalf is None:
        xphalf = 1.0 - w_E1

    xped = xphalf - w_E1

    pconst = 1.0 - np.tanh((1.0 - xphalf) / w_E1)
    a_t = 2.0 * (ped - edge) / (1.0 + np.tanh(1.0) - pconst)

    coretanh = 0.5 * a_t * (1.0 - np.tanh(-xphalf / w_E1) - pconst) + edge

    #xpsi = np.linspace(0, 1, rgrid)
    #ones = np.ones(rgrid)
    
    val = 0.5 * a_t * (1.0 - np.tanh((x_coord - xphalf) / w_E1) - pconst) + edge * np.ones_like(x_coord)

    xtoped = x_coord / xped
    for i in range(0, len(x_coord)):
        if xtoped[i] ** expin < 1.0:
            val[i] = val[i] + (core - coretanh) * (1.0 - xtoped[i] ** expin) ** expout

    return val
def Osbourne_Tanh(x,C):
    """
    adapted from Osborne via Hughes idl script
    tanh function with cubic or quartic inner and linear
    to quadratic outer extensions and 

    INPUTS: 
    c = vector of coefficients defined as such:
    c[0] = pedestal center position
    c[1] = pedestal full width
    c[2] = Pedestal top
    c[3] = Pedestal bottom
    c[4] = inboard slope
    c[5] = inboard quadratic term
    c[6] = inboard cubic term
    c[7] = outboard linear term
    c[8] = outbard quadratic term
    x = x-axis
    """

    z = 2. * ( C[0] - x ) / C[1]
    P1 = 1. + C[4] * z + C[5] * z**2 + C[6] * z**3
    P2 = 1. + C[7] * z + C[8] * z**2
    E1 = np.exp(z)
    E2 = np.exp(-1.*z)
    F = 0.5 * ( C[2] + C[3] + ( C[2] - C[3] ) * ( P1 * E1 - P2 * E2 ) / ( E1 + E2 ) )

    return F

def super_fit_osbourne(x_coord, vals, vals_unc=None, x_out=None,
              guess=None, plot=False,maxfev=2000):
    '''Fast and complete 1D full-profile fit.
    
    Parameters
    ----------
    x_coord : 1D array
        Radial coordinate on which profile is given
    vals : 1D array
        Profile values to be fitted
    vals_unc : 1D array
        Array containing uncertainties for values. If these are not given, the fit uses
        only the values, i.e. it uses least-squares rather than chi^2 minimization.
    x_out : 1D array
        Desired output coordinate array. If not given, this is set equal to the x_coord array.
    plot : bool
        If True, plot the raw data and the fit. 

    Returns
    -------
    res_fit : 1D array
        Fitted profile, on the x_out grid (which may be equal to x_coord if x_out was not provided)
    c = vector of coefficients defined as such:
    c[0] = pedestal center position
    c[1] = pedestal full width
    c[2] = Pedestal top
    c[3] = Pedestal bottom
    c[4] = inboard slope
    c[5] = inboard quadratic term
    c[6] = inboard cubic term
    c[7] = outboard linear term
    c[8] = outbard quadratic term

    Notes
    -----
    This is adopted from Jerry Hughes script on c-mod in idl. Tested on Psi_n grid...should be agnostic  
    '''
    
    if isinstance(vals, (int,float)) or isinstance(x_coord, (int, float)):
        raise ValueError('Input profile to super_fit is a scalar! This function requires 1D profiles.')

    def func(x,c0,c1,c2,c3,c4,c5,c6,c7,c8):
        c = np.asarray([c0,c1,c2,c3,c4,c5,c6,c7,c8])
        nval = Osbourne_Tanh(x,c)
        return nval

    # guesses for minimization
    width = 0.03
    xphalf0 = 1 - width * 0.5
    ped_height_guess = interp1d(x_coord, vals)([1 - 2 * width])[0]
    ped_slope_guess = np.abs((vals[0]-ped_height_guess)/(np.min(x_coord)-np.max(x_coord)))/ped_height_guess #expect positive
    sol_slope_guess = -ped_slope_guess/10.0

    # see parameters order in the function docstring
    if guess==None:
        guess = [xphalf0,width, ped_height_guess, vals[-1], ped_slope_guess,0,0,sol_slope_guess,0]
    """
    # relatively agnostic bounds that should work in all radial coordinate choices
    if bounds is None:
        #bounds = [(0.8,1.1), (0.01,0.1), (0,None), (0, None),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)]
        bounds = [(0.8,1.1), (0.01,0.1), (0,None), (0, None),(None,None),(None,None),(None,None),(None,None),(None,None)]
    """
    # run fit/minimization    

    popt,popc = optimize.curve_fit(func,x_coord,vals,p0=guess,sigma = vals_unc,method = 'lm',maxfev=maxfev)        

    if x_out is None:
        x_out = x_coord
    res_fit = Osbourne_Tanh(x_out,popt)

    # ensure positivity/minima
    res_fit[res_fit<np.nanmin(vals)] = np.nanmin(vals)
    
    if plot:
        plt.figure()
        if vals_unc is None:
            plt.plot(x_coord, vals, 'r.', label='raw')
        else:
            plt.errorbar(x_coord, vals, vals_unc, fmt='.',c='r', label='raw')
        plt.plot(x_out, res_fit, 'b-', label='fit')
        plt.legend()

    return res_fit, popt


def super_fit(x_coord, vals, vals_unc=None, x_out=None, edge_focus=True,
              bounds=None, plot=False):
    '''Fast and complete 1D full-profile fit.
    
    Parameters
    ----------
    x_coord : 1D array
        Radial coordinate on which profile is given
    vals : 1D array
        Profile values to be fitted
    vals_unc : 1D array
        Array containing uncertainties for values. If these are not given, the fit uses
        only the values, i.e. it uses least-squares rather than chi^2 minimization.
    x_out : 1D array
        Desired output coordinate array. If not given, this is set equal to the x_coord array.
    edge_focus : bool
        If True, the fit takes special care to fit the pedestal and SOL and may give a poor core
        fit. If False, a weaker weight will be assigned to the optimal pedestal match.
    bounds : array of 2-tuple
        Bounds for optimizer. Must be of the right shape! See c array of parameters below.
        If left to None, a default set of bounds will be used.
    plot : bool
        If True, plot the raw data and the fit. 

    Returns
    -------
    res_fit : 1D array
        Fitted profile, on the x_out grid (which may be equal to x_coord if x_out was not provided)
    c : 1D array
        Fitting parameters to the `mtanh_profile` function, in the following order:
        :param edge: (float) separatrix height
        :param ped: (float) pedestal height
        :param core: (float) on-axis profile height
        :param expin: (float) inner core exponent for H-mode pedestal profile
        :param expout (float) outer core exponent for H-mode pedestal profile
        :param widthp: (float) width of pedestal
        :param xphalf: (float) position of tanh

    Notes
    -----
    Note that this function doesn't care about what radial coordinate you pass in x_coord,
    but all fitted parameters will be consistent with the coordinate choice made.    
    '''
    
    if isinstance(vals, (int,float)) or isinstance(x_coord, (int, float)):
        raise ValueError('Input profile to super_fit is a scalar! This function requires 1D profiles.')

    def func(c):
        if any(c < 0):
            return 1e10
        nval = mtanh_profile(x_coord, edge=c[0], ped=c[1], core=c[2], expin=c[3], expout=c[4], widthp=c[5], xphalf=c[6])
        if vals_unc is None:
            cost = np.sqrt(sum(((vals - nval) ** 2 / vals[0] ** 2) * weight_func))
        else:
            cost = np.sqrt(sum(((vals - nval) ** 2 / vals_unc ** 2) * weight_func))
        return cost

    if edge_focus:
        weight_func = ((x_coord > 0.85)*(x_coord<1.1) * x_coord + 0.8)**2 #* x_coord
        #weight_func = ((x_coord > 0.85) * x_coord + 0.001) * x_coord
    else:
        weight_func = 0.1 + x_coord #(x_coord + 0.001) * x_coord

    # guesses for minimization
    width = 0.03
    xphalf0 = 1 - width * 0.5
    ped_height_guess = interp1d(x_coord, vals)([1 - 2 * width])[0]

    # see parameters order in the function docstring
    guess = [vals[-1], ped_height_guess, vals[0], 2.0, 2.0, width, xphalf0]
    #print(guess)

    # relatively agnostic bounds that should work in all radial coordinate choices
    if bounds is None:
        #bounds = [(0, None), (0,None), (0,None), (None,None), (None,None), (0.01,0.05), (0.9,1.1)]
        bounds = [(0, None), (0,None), (0,None), (None,None), (None,None), (0.01,0.1), (0.8,1.1)]

    
    # run fit/minimization            
    c = list(map(float, optimize.minimize(func, guess,
                                          method='L-BFGS-B', #'Nelder-Mead',# NM cannot take bounds
                                          bounds=bounds, jac=False).x))

    if x_out is None:
        x_out = x_coord
    res_fit = mtanh_profile(x_out, edge=c[0], ped=c[1], core=c[2], expin=c[3], expout=c[4], widthp=c[5], xphalf=c[6])

    # ensure positivity/minima
    res_fit[res_fit<np.nanmin(vals)] = np.nanmin(vals)
    
    if plot:
        plt.figure()
        if vals_unc is None:
            plt.plot(x_coord, vals, 'r.', label='raw')
        else:
            plt.errorbar(x_coord, vals, vals_unc, fmt='.',c='r', label='raw')
        plt.plot(x_out, res_fit, 'b-', label='fit')
        plt.legend()

    return res_fit, c
