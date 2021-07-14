'''
Methods to load and plot C-Mod Ly-alpha data.

sciortino, August 2020
'''
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import xarray
from scipy.interpolate import interp1d, interp2d
from omfit_classes import omfit_eqdsk, omfit_mds
import shutil, os, scipy, copy
from IPython import embed
import MDSplus
from omfit_classes.omfit_mds import OMFITmdsValue

from scipy.constants import Boltzmann as kB, e as q_electron
from scipy.optimize import curve_fit
import aurora
import sys
sys.path.append('/home/sciortino/usr/python3modules/profiletools3')
sys.path.append('/home/sciortino/usr/python3modules/eqtools3')
import profiletools

# from mitlya repo
import mtanh_fitting
import fit_2D as fit
import LLAMA_tomography as tomo


def get_cmod_kin_profs(shot,tmin,tmax,gfiles_loc=None):
    '''Function to load and fit modified-tanh functions to C-Mod ne and Te.

    This function is designed to be robust for operation within the construction of
    the C-Mod Ly-a database. It makes use of the profiletools package just to conveniently
    collect data and time-average it. An updated profiletools (python 3+) is available 
    from one of the github repos of sciortinof.

    Returns
    -------
    roa_kp : 1D array
        r/a grid
    ne : 1D array
        Electron density in units of :math:`10^{20} m^{-3}`
    ne_std : 1D array
        Uncertainties on electron density in units of :math:`10^{20} m^{-3}`
    Te : 1D array
        Electron temperature in units of :math:`keV`
    Te_std : 1D array
        Uncertainties on electron temperature in units of :math:`keV`
    p_ne : profiletools object
        Electron density object containing experimental data from all loaded diagnostics.
    p_Te : profiletools object
        Electron temperature object containing experimental data from all loaded diagnostics.
    '''   
    
    # require edge Thomson to be available
    p_Te= profiletools.Te(int(shot), include=['ETS'],
                              abscissa='r/a',t_min=tmin,t_max=tmax)
    p_ne= profiletools.ne(int(shot), include=['ETS'],
                          abscissa='r/a',t_min=tmin,t_max=tmax)

    try:
        # try to add core Thomson, not strictly necessary 
        p_Te_CTS = profiletools.Te(int(shot), include=['CTS'],
                                   abscissa='r/a',t_min=tmin,t_max=tmax)
        p_ne_CTS = profiletools.ne(int(shot), include=['CTS'],
                                   abscissa='r/a',t_min=tmin,t_max=tmax)
        #p_Te.remove_points(p_Te.X[:,1]>0.9
        p_Te.add_profile(p_Te_CTS)
        p_ne.add_profile(p_ne_CTS)
    except Exception:
        pass

    try:
        # try to add GPC and GPC2 for Te, not strictly necessary
        p_Te_GPC= profiletools.Te(int(shot), include=['GPC','GPC2'],
                                  abscissa='r/a',t_min=tmin,t_max=tmax)

        # downsample to fewer points in time interval
        p_Te_GPC.keep_slices(0, np.linspace(tmin,tmax, 300))
        p_Te_GPC.remove_points(p_Te_GPC.X[:,1]>0.8)  # doubtful about ECE opacity
        p_Te.add_profile(p_Te_GPC)
    except:
        pass

    # consider only flux surface on which points were measured, regardless of LFS or HFS
    p_Te.X=np.abs(p_Te.X)
    p_ne.X=np.abs(p_ne.X)

    # set some minimum uncertainties. Recall that units in objects are 1e20m^{-3} and keV
    p_ne.y[p_ne.y<=0.] = 0.01  # 10^18 m^-3
    p_Te.y[p_Te.y<=0.01] = 0.01 # 10 eV
    p_ne.err_y[p_ne.err_y<=0.01] = 0.01 # 10^18 m^-3
    p_Te.err_y[p_Te.err_y<=0.02] = 0.02 # 20 eV

    # points in the pedestal that have x uncertainties larger than 0.1 don't help at all
    # do this filtering here because filtering of err_X only works before time-averaging
    #p_ne.remove_points(np.logical_and(p_ne.X[:,1]>0.9, p_ne.err_X[:,1]>0.1))
    #p_Te.remove_points(np.logical_and(p_Te.X[:,1]>0.9, p_Te.err_X[:,1]>0.1))
    
    # time average now, before trying to add time-independent probe data
    p_ne.time_average(weighted=True)
    p_Te.time_average(weighted=True)
    #p_ne.average_data(axis=0, weighted=True)
    #p_Te.average_data(axis=0, weighted=True)

    num_ne_TS = len(p_ne.X)
    num_Te_TS = len(p_Te.X)

    ## use two point model to get T_sep

    # need to fit TS first 
    min_TS_X = p_Te.X.min() if p_Te.X.min() < p_ne.X.min() else p_ne.X.min()
    max_TS_X = p_Te.X.max() if p_Te.X.max() > p_ne.X.max() else p_ne.X.max()
    X_TS_fit = np.linspace(min_TS_X,max_TS_X,100)

    try: # sometimes get underflow error when errorbars put in 
        _out = mtanh_fitting.super_fit(p_Te.X[:,0],p_Te.y,vals_unc=p_Te.err_y,x_out=X_TS_fit)
    except:
        _out = mtanh_fitting.super_fit(p_Te.X[:,0],p_Te.y,x_out=X_TS_fit)
    
    # try osbourne fit as well
    try:
        _out = mtanh_fitting.super_fit_osbourne(p_Te.X[:,0], p_Te.y, x_out=X_TS_fit)
    except:
        pass
    Te_TS_fit, c_Te = _out

    try:
        _out = mtanh_fitting.super_fit(p_ne.X[:,0],p_ne.y,vals_unc=p_ne.err_y,x_out=X_TS_fit)
    except:
        _out = mtanh_fitting.super_fit(p_ne.X[:,0],p_ne.y,x_out=X_TS_fit)
    try:
        _out = mtanh_fitting.super_fit_osbourne(p_ne.X[:,0], p_ne.y, x_out=X_TS_fit)
    except:
        pass
    ne_TS_fit, c_ne = _out

    Te_lcfs_eV = fit.Teu_2pt_model(shot,tmin,tmax,ne_TS_fit,Te_TS_fit,X_TS_fit,gfiles_loc=gfiles_loc)
    print('Te LCFS eV', Te_lcfs_eV)

    ## try to shift TS profiles using T_lcfs

    xSep_TS = fit.shift_profs([1],X_TS_fit,Te_TS_fit[None,:]*1e3,Te_LCFS=Te_lcfs_eV)
    p_ne.X += 1 - xSep_TS
    p_Te.X += 1 - xSep_TS

    if gfiles_loc is None:
        gfiles_loc = '/home/sciortino/EFIT/gfiles'

    
    # attempt to fetch ASP and FSP data if available
    # NB: min and max rhop are imposed here to avoid problematic data!
    p_ne_p, p_Te_p  = fetch_edge_probes(shot, (tmin+tmax)/2., Te_lcfs_eV, roa_min=0.96, roa_max=1.02, gfiles_loc=gfiles_loc)

    if p_ne_p is not None: # either both are None or neither is None
        
        # remove TS points where probe data < cutoff
        #cutoff = 0.02 # in keV
        #lt_ts = np.where(p_Te_p.y < cutoff)
        #x_lt_ts = p_Te_p.X[lt_ts[0][0],0]
        #p_ne.remove_points(p_ne.X[:,0] > x_lt_ts)
        #p_Te.remove_points(p_Te.X[:,0] > x_lt_ts)

        # add cleaned profiles

        p_ne.add_profile(p_ne_p)
        p_Te.add_profile(p_Te_p)

        num_ne_SP = len(p_ne_p.X)
        num_Te_SP = len(p_Te_p.X)
    
    ne_X_before = p_ne.X[:,0]
    Te_X_before = p_Te.X[:,0]


    # for some reason this is needed again after time averaging; applies to any probe data too
    p_ne.y[p_ne.y<=0.] = 0.01  # 10^18 m^-3
    p_Te.y[p_Te.y<=0.01] = 0.01 # 10 eV
    p_ne.err_y[p_ne.err_y<=0.1] = 0.1 # 10^19 m^-3
    p_Te.err_y[p_Te.err_y<=0.02] = 0.02 # 20 eV

    # remove points with stupidly large error bars
    p_ne.remove_points(p_ne.err_y>1) # 10^20 m^-3
    
    # Remove points with too high values in the SOL:
    p_ne.remove_points(np.logical_and(p_ne.X[:,0]>1.0, p_ne.y>1)) # 1e20
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>1.0, p_Te.y>0.25)) # 250 eV

    # Remove ne points with too high uncertainty in the pedestal and SOL:
    p_ne.remove_points(np.logical_and(p_ne.X[:,0]>0.9, p_ne.err_y>0.5)) # 5e19 m^-3

    # Remove Te points with too high uncertainty in the SOL (better not to filter in the pedestal, high variability)
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>1.0, p_Te.err_y>0.1))  # 100 eV
   
    # Substitute pedestal and SOL fit with tanh fit
    roa_kp = np.linspace(0.0, 1.16, 300) # can't extend too much past 1.16

    # check if there exist any data point in the SOL
    if np.all(p_ne.X[:,0]<1.0):
        raise ValueError(f'No SOL ne data points in shot {shot}!')
    if np.all(p_Te.X[:,0]<1.0):
        raise ValueError(f'No SOL Te data points in shot {shot}!')

    # force fits to go down in the far SOL (r/a=1.2) to make the routine more robust
    p_Te.add_data( np.array([[1.2]]), np.array([10e-3]), err_X=np.array([0.001]), err_y=np.array([0.001]))
    p_ne.add_data( np.array([[1.2]]), np.array([0.1]), err_X=np.array([0.001]), err_y=np.array([0.001]))

    # Now fit:    
    maxfev = 2000 # number of iterations for osbourne fit to converge
    idxs = np.argsort(p_ne.X[:,0])

    try:
        ne, ne_c = mtanh_fitting.super_fit_osbourne(p_ne.X[idxs,0], p_ne.y[idxs], p_ne.err_y[idxs], x_out=roa_kp,
                                                    maxfev=maxfev)
    except:
        ne, ne_c = mtanh_fitting.super_fit(p_ne.X[idxs,0], p_ne.y[idxs], p_ne.err_y[idxs], x_out=roa_kp,
                                       edge_focus=True, plot=False, bounds=None)
    
    idxs = np.argsort(p_Te.X[:,0])
    
    try:
        Te, Te_c = mtanh_fitting.super_fit_osbourne(p_Te.X[idxs,0], p_Te.y[idxs], p_Te.err_y[idxs], x_out=roa_kp,
                                                    maxfev=maxfev)
    except:
        Te, Te_c = mtanh_fitting.super_fit(p_Te.X[idxs,0], p_Te.y[idxs], p_Te.err_y[idxs], x_out=roa_kp,
                                       edge_focus=True, plot=False, bounds=None)

    # impose positivity
    ne[ne<np.nanmin(p_ne.y)] = np.nanmin(p_ne.y)
    Te[Te<np.nanmin(p_Te.y)] = np.nanmin(p_Te.y)
    
    # eliminate points that are more than 3 sigma away and fit again
    p_ne.remove_points(p_ne.X[:,0]>1.1) # remove artificial point
    chi_ne = (p_ne.y - interp1d(roa_kp, ne)(p_ne.X[:,0]))/p_ne.err_y
    p_ne.remove_points(chi_ne>3)
    p_Te.remove_points(p_Te.X[:,0]>1.1) # remove artificial point
    chi_Te = (p_Te.y - interp1d(roa_kp, Te)(p_Te.X[:,0]))/p_Te.err_y
    p_Te.remove_points(chi_Te>3)

    # add xtra points in the far SOL again to force fits to go down
    p_Te.add_data( np.array([[1.2]]), np.array([10e-3]), err_X=np.array([0.001]), err_y=np.array([0.001]))
    p_ne.add_data( np.array([[1.2]]), np.array([0.1]), err_X=np.array([0.001]), err_y=np.array([0.001]))

    # Fit again:    
    idxs = np.argsort(p_ne.X[:,0])
    try:
        ne, ne_c = mtanh_fitting.super_fit_osbourne(p_ne.X[idxs,0], p_ne.y[idxs], p_ne.err_y[idxs], x_out=roa_kp,
                                                    maxfev=maxfev)
    except:
        ne, ne_c = mtanh_fitting.super_fit(p_ne.X[idxs,0], p_ne.y[idxs], p_ne.err_y[idxs], x_out=roa_kp,
                                    edge_focus=True, plot=False, bounds=None)
    idxs = np.argsort(p_Te.X[:,0])
    try:
        Te, Te_c = mtanh_fitting.super_fit_osbourne(p_Te.X[idxs,0], p_Te.y[idxs], p_Te.err_y[idxs], x_out=roa_kp,
                                                    maxfev=maxfev)
    except:
        Te, Te_c = mtanh_fitting.super_fit(p_Te.X[idxs,0], p_Te.y[idxs], p_Te.err_y[idxs], x_out=roa_kp,
                                       edge_focus=True, plot=False, bounds=None)
    
    ne_std = ne*0.2 # 20% -- tanh fitting not set up to provide good uncertainties
    Te_std = Te*0.2 # 20%

    p_ne.remove_points(p_ne.X[:,0]>1.1) # remove artificial point
    p_Te.remove_points(p_Te.X[:,0]>1.1) # remove artificial point
    
    ne_X_after = p_ne.X[:,0]
    Te_X_after = p_Te.X[:,0]
    
    ne_pts_removed = np.setdiff1d(ne_X_before,ne_X_after)
    Te_pts_removed = np.setdiff1d(Te_X_before,Te_X_after)
   
    for pt in ne_pts_removed:
        if pt in p_ne_p.X:
            num_ne_SP -= 1
        else:
            num_ne_TS -= 1
    
    for pt in Te_pts_removed:
        if pt in p_Te_p.X:
            num_Te_SP -= 1
        else:
            num_Te_TS -= 1
   
    #p_ne.plot_data()
    #plt.xlim([0.86,1.16])
    #plt.ylim([-0.01, 0.75])
    #plt.gca().plot(roa_kp, ne2)
    #plt.gca().plot(roa_kp, ne)
    #plt.legend(['super_fit', 'super_fit_osbourne'])

    #p_Te.plot_data()
    #plt.xlim([0.86,1.16])
    #plt.ylim([-0.01, 0.5])
    #plt.gca().plot(roa_kp, Te2)
    #plt.gca().plot(roa_kp, Te)
    #plt.legend(['super_fit', 'super_fit_osbourne'])

    kpnum = {}
    num_ne = {}; num_Te = {}
    kpnum['ne'] = num_ne; kpnum['Te'] = num_Te
    num_ne['TS'] = num_ne_TS; num_ne['SP'] = num_ne_SP
    num_Te['TS'] = num_Te_TS; num_Te['SP'] = num_Te_SP

    # output fits + profiletool objects for ne and Te so that experimental data points are passed too
    return roa_kp, ne, ne_std, Te, Te_std, p_ne, p_Te, kpnum

            



def fetch_edge_probes(shot, time, Te_lcfs_eV, roa_min=0.995, roa_max=1.05, gfiles_loc=None):
    '''Load data for the ASP and FSP probes on Alcator C-Mod. 
    Time in seconds.

    rhop_min and rhop_max are used to subselect the radial range of the data.

    This function returns profiletools data structures.
    See https://profiletools.readthedocs.io/en/latest/#

    '''
    import afsp_probes
    import lyman_data
    import eqtools
        
    # will be used for conversions
    eq = eqtools.CModEFITTree(shot)

    has_A = False
    has_F = False
    try:
        # if available, add A-Side Probe (ASP) data
        out = afsp_probes.get_clean_data(shot, time, probe='A', plot=False, gfiles_loc=gfiles_loc)
        roa_asp, roa_asp_unc, t_range, ne_prof_asp, ne_unc_prof_asp, Te_prof_asp, Te_unc_prof_asp, ax = out

        # probe data is returned in SI units, change to units of 1e20m^-3 and keV
        ne_prof_asp /= 1e20
        ne_unc_prof_asp /= 1e20
        Te_prof_asp /= 1e3
        Te_unc_prof_asp /= 1e3

        # mask out data points outside of given range
        mask_asp = (roa_asp>roa_min)&(roa_asp<roa_max)

        ASP_X = np.ones((len(roa_asp),1))
        ASP_X[:,0] = roa_asp

        p_ne_ASP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='$10^{20}$ m$^{-3}$',
                                                       X_labels=r'$r/a$', y_label=r'$n_e$, ASP')
        p_ne_ASP.efit_tree = eq
        p_ne_ASP.abscissa = 'r/a'
        p_ne_ASP.shot = shot
        p_ne_ASP.t_min = t_range[0]; p_ne_ASP.t_max = t_range[1]
        channels = range(0, len(ne_prof_asp[mask_asp]))
        p_ne_ASP.add_data(ASP_X[mask_asp], ne_prof_asp[mask_asp], channels={0: channels}, err_y=ne_unc_prof_asp[mask_asp])

        p_Te_ASP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='keV',
                                                       X_labels=r'$r/a$', y_label=r'$T_e$, ASP')
        p_Te_ASP.efit_tree = eq
        p_Te_ASP.abscissa = 'r/a'
        p_Te_ASP.shot = shot
        p_Te_ASP.t_min = t_range[0]; p_Te_ASP.t_max = t_range[1]
        channels = range(0, len(Te_prof_asp[mask_asp]))
        p_Te_ASP.add_data(ASP_X[mask_asp], Te_prof_asp[mask_asp], channels={0: channels}, err_y=Te_unc_prof_asp[mask_asp])
        
        has_A = True
    except Exception:
        print('ASP fetch failed')
        pass
        
    ######
    try:
        # if available, add F-Side Probe (FSP) data
        out = afsp_probes.get_clean_data(shot, time, probe='F', plot=False, gfiles_loc=gfiles_loc)
        rhop_fsp, rhop_fsp_unc, t_range, ne_prof_fsp, ne_unc_prof_fsp, Te_prof_fsp, Te_unc_prof_fsp, ax = out

        # probe data is returned in SI units, change to units of 1e20m^-3 and keV
        ne_prof_fsp /= 1e20
        ne_unc_prof_fsp /= 1e20
        Te_prof_fsp /= 1e3
        Te_unc_prof_fsp /= 1e3

        # mask out data points outside of given range
        mask_fsp = (rhop_fsp>rhop_min)&(rhop_fsp<rhop_max)

        roa_fsp = eq.rho2rho('psinorm','r/a',rhop_fsp**2,time)
        FSP_X = np.ones((len(roa_fsp),1))
        FSP_X[:,0] = roa_fsp

        p_ne_FSP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='$10^{20}$ m$^{-3}$',
                                                       X_labels=r'$\sqrt{\psi_n}$', y_label=r'$n_e$, FSP')
        p_ne_FSP.efit_tree = eqtools.CModEFITTree(shot)
        p_ne_FSP.abscissa = 'sqrtpsinorm'
        p_ne_FSP.shot = shot
        p_ne_FSP.t_min = t_range[0]; p_ne_FSP.t_max = t_range[1]        
        channels = range(0, len(ne_prof_fsp[mask_fsp]))
        p_ne_FSP.add_data(FSP_X[mask_fsp], ne_prof_fsp[mask_fsp], channels={0: channels}, err_y=ne_unc_prof_fsp[mask_fsp])

        p_Te_FSP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='keV',
                                                       X_labels=r'$\sqrt{\psi_n}$', y_label=r'$T_e$, FSP')
        p_Te_FSP.efit_tree = eqtools.CModEFITTree(shot)
        p_Te_FSP.abscissa = 'sqrtpsinorm'
        p_Te_FSP.shot = shot
        p_Te_FSP.t_min = t_range[0]; p_Te_FSP.t_max = t_range[1]
        channels = range(0, len(Te_prof_fsp[mask_fsp]))
        p_Te_FSP.add_data(FSP_X[mask_fsp], Te_prof_fsp[mask_fsp], channels={0: channels}, err_y=Te_unc_prof_fsp[mask_fsp])
        
        has_F = True
    except Exception:
        print('FSP fetch failed')
        pass


    fit_A = False
    fit_F = False
    # try to fit probes for the shift
    if has_A and has_F:
        try:
            _out = fit_probes(p_Te_ASP,p_ne_ASP)
            Te_ASP_fit, ne_ASP_fit = _out
            fit_A = True
        except:
            print('Could not fit ASP data')
        try:
            _out = fit_probes(p_Te_FSP,p_ne_FSP)
            Te_FSP_fit, ne_FSP_fit = _out
            fit_F = True
        except:
            print('Could not fit FSP data')
    elif has_A and not has_F:
        try:
            _out = fit_probes(p_Te_ASP,p_ne_ASP)
            Te_ASP_fit, ne_ASP_fit = _out
            fit_A = True
        except:
            print('Could not fit ASP data')
    elif not has_A and has_F:
        try:
            _out = fit_probes(p_Te_FSP,p_ne_FSP)
            Te_FSP_fit, ne_FSP_fit = _out
            fit_F = True
        except:
            print('Could not fit FSP data')

    shift_A = False
    shift_F = False
    #try to shift probe data
    if fit_A and fit_F:
        xSep_ASP = fit.shift_profs([1],p_Te_ASP.X[:,0],Te_ASP_fit[None,:]*1e3,Te_LCFS=Te_lcfs_eV)
        shift_A = True if xSep_ASP != 1 else False
        xSep_FSP = fit.shift_profs([1],p_Te_FSP.X[:,0],Te_FSP_fit[None,:]*1e3,Te_LCFS=Te_lcfs_eV)
        shift_F = True if xSep_FSP != 1 else False
    elif fit_A and not fit_F:
        xSep_ASP = fit.shift_profs([1],p_Te_ASP.X[:,0],Te_ASP_fit[None,:]*1e3,Te_LCFS=Te_lcfs_eV)
        shift_A = True if xSep_ASP != 1 else False
    elif not fit_A and fit_F:
        xSep_FSP = fit.shift_profs([1],p_Te_FSP.X[:,0],Te_FSP_fit[None,:]*1e3,Te_LCFS=Te_lcfs_eV)
        shift_F = True if xSep_FSP != 1 else False

    p_ne = None
    p_Te = None

    # multiply probe ne by some factor to match TS
    mult_factor = 2

    if shift_A and shift_F:
        # shift and add data from the two probes
        p_ne_ASP.X += 1 - xSep_ASP
        p_ne_FSP.X += 1 - xSep_FSP
        p_ne = p_ne_ASP
        p_ne.add_profile(p_ne_FSP)
        p_Te_ASP.X += 1 - xSep_ASP
        p_Te_FSP.X += 1 - xSep_FSP
        p_Te = p_Te_ASP
        p_Te.add_profile(p_Te_FSP)

        #p_ne.y*=mult_factor

    elif shift_A and not shift_F:
        p_ne_ASP.X += 1 - xSep_ASP
        p_ne = p_ne_ASP
        p_Te_ASP.X += 1 - xSep_ASP
        p_Te = p_Te_ASP
        
        #p_ne.y*=mult_factor

    elif not shift_A and shift_F:
        p_ne_FSP.X += 1 - xSep_FSP
        p_ne = p_ne_FSP
        p_Te_FSP.X += 1 - xSep_FSP
        p_Te = p_Te_FSP
        
        #p_ne.y*=mult_factor

    # remove probe points for psi < 1
    if p_ne is not None:
        p_ne.remove_points(p_ne.X[:,0] < 1)
        p_Te.remove_points(p_Te.X[:,0] < 1)
  
    return p_ne, p_Te


# def fav_vs_unfav(shot,time,geqdsk = None):
#     '''Determine whether grad-B field direction is favorable or unfavorable.
    
#     This function ignores the possibility of having a double null, use with care!
#     The active x-point is taken to be the point along the LCFS that is furthest from Z=0.

#     '''
#     if geqdsk is None:
#         geqdsk = get_geqdsk_cmod(shot,time*1e3)
#         geqdsk = get_geqdsk_cmod(shot,time*1e3)  # repeat to make sure it's loaded...
    
#     # find ion grad(B)-drift direction (determined by B field dir, since radial grad(B) is always inwards )
#     magTree = MDSplus.Tree('magnetics',shot)
#     nodeBt = magTree.getNode('\magnetics::Bt')
#     Bt = nodeBt.data()
#     time_Bt = nodeBt.dim_of().data()
#     tidx = np.argmin(np.abs(time_Bt - time)) 
#     gradB_drift_up = False if Bt[tidx]<0 else True
    
#     # find whether shot is USN or LSN -- assume not DN...
#     maxZ = np.max(geqdsk['ZBBBS'])
#     minZ = np.min(geqdsk['ZBBBS'])
    
#     #  pretty sure that the X-point is where the LCFS is furthest from the magnetic axis
#     USN = True if np.abs(maxZ)==np.max([np.abs(maxZ),np.abs(minZ)]) else False
    
#     # favorable or unfavorable grad-B drift direction?
#     favorable = (gradB_drift_up and USN) or (gradB_drift_up==False and USN==False)
    
#     return gradB_drift_up, USN, favorable


def fit_probes(Te,ne):
    ''' Fit probe data with exponentially decaying functions'''

    _out = curve_fit(probe_func,Te.X[:,0]-1,Te.y*1e3) # fit works better if x shifted to 0
    popt_Te,pcov_Te = _out
    Te_fit = probe_func(Te.X[:,0]-1,*popt_Te)/1e3

    _out = curve_fit(probe_func,ne.X[:,0]-1,ne.y*10) # fit works better if data O~(1)
    popt_ne,pcov_ne = _out
    ne_fit = probe_func(ne.X[:,0]-1,*popt_ne)/10

    return Te_fit, ne_fit


def probe_func(x,a,k,b):
    ''' Fucntion used to fit probe data'''
    return a*np.exp(-k*x)+b

        
def get_vol_avg_pressure(shot,time,rhop,ne,Te):
    ''' Calculate volume-averaged pressure given some ne,Te radial profiles.

    ne must be in cm^-3 units and Te in eV.
    '''
    # find volume-averaged pressure    
    p_Pa = (ne*1e6) * (Te*q_electron)
    p_atm = p_Pa/101325.  # conversion factor between Pa and atm

    # load geqdsk dictionary
    geqdsk = get_geqdsk_cmod(shot,time*1e3)
    
    # find volume average within LCFS
    indLCFS = np.argmin(np.abs(rhop-1.0))
    p_Pa_vol_avg = aurora.vol_average(p_Pa[:indLCFS], rhop[:indLCFS], method='omfit',geqdsk = geqdsk)[-1]
    #p_atm_vol_avg = p_Pa_vol_avg/101325.

    return p_Pa_vol_avg



def get_geqdsk_cmod(shot,time_ms, gfiles_loc = '/home/sciortino/EFIT/gfiles/'):
    ''' Get a geqdsk file in omfit_eqdsk format by loading it from disk, if available, 
    or from MDS+ otherwise.  

    This function tries to first load a EFIT20, if available.

    time must be in ms!

    Currently, the omfit_eqdsk class struggles to connect to MDS+ sometimes and returns None.
    Calling this function twice usually avoids the problem.
    '''
    time_ms = np.floor(time_ms)   # TODO: better to do this outside!!
    file_name=f'g{shot}.{str(int(time_ms)).zfill(5)}'

    def fetch_and_move():
        try:
            # try to fetch EFIT20 first
            geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
                device='CMOD',shot=shot, time=time_ms, SNAPfile='EFIT20',
                fail_if_out_of_range=True,time_diff_warning_threshold=20
            )
        except:
            # if EFIT20 is not available, look for default ANALYSIS EFIT
            geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
                device='CMOD',shot=shot, time=time_ms, SNAPfile='ANALYSIS',
                fail_if_out_of_range=True,time_diff_warning_threshold=20
            )
    
        geqdsk.save(raw=True)
        shutil.move(file_name, gfiles_loc+file_name)
    
    if os.path.exists(gfiles_loc + file_name):
        # fetch local g-file if available
        try:
            geqdsk = omfit_eqdsk.OMFITgeqdsk(gfiles_loc + file_name)
            kk = geqdsk.keys()  # quick test
        except:
            geqdsk = fetch_and_move()
    else:
        geqdsk = fetch_and_move()
        
    return geqdsk

    
def get_Greenwald_frac(shot, tmin,tmax, roa, ne, Ip_MA, a_m=0.22):
    ''' Calculate Greenwald density fraction by normalizing volume-averaged density.

    INPUTS
    ------
    shot : int, shot number
    tmin and tmax: floats, time window (in [s]) to fetch equilibrium.
    ne: 1D array-like, expected as time-independent. Units of 1e20 m^-3.
    Ip_MA: float, plasma current in MA. 
    a_m : minor radius in units of [m]. Default of 0.69 is for C-Mod. 

    OUTPUTS
    -------
    n_by_nG : float
        Greenwald density fraction, defined with volume-averaged density.
    '''
    tmin *= 1000.  # change to ms
    tmax *= 1000.  # change to ms
    time = (tmax+tmin)/2.
    geqdsk = get_geqdsk_cmod(shot,time)
    
    # find volume average within LCFS
    rhop = aurora.rad_coord_transform(roa,'r/a','rhop', geqdsk)

    indLCFS = np.argmin(np.abs(rhop-1.0))
    n_volavg = aurora.vol_average(ne[:indLCFS], rhop[:indLCFS], geqdsk=geqdsk)[-1]

    # Greenwald density
    n_Gw = Ip_MA/(np.pi * a_m**2)   # units of 1e20 m^-3, same as densities above

    # Greenwald fraction:
    f_gw = n_volavg/n_Gw

    return f_gw



def get_CMOD_gas_fueling(shot, plot=False):
    '''Load injected gas amounts and give a grand total in Torr-l.
    Translated from gas_input2_ninja.dat scope. 
    '''

    _c_side = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                             TDI='\\plen_cside').data()[0,:],31)
    _t = omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                 TDI='dim_of(\\plen_cside)').data()
    _b_sideu = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                              TDI='\\plen_bsideu').data()[0,:],31)
    _b_top = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                            TDI='\\plen_btop').data()[0,:],31)

    plen_bot_time = omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='edge',
                                            TDI='\edge::gas_ninja.plen_bot').dim_of(0)
    plen_bot = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='edge',
                                              TDI='\edge::gas_ninja.plen_bot').data()[0,:],31)

    # only work with quantities within [0,2]s interval
    ind0 = np.argmin(np.abs(_t)); ind1 = np.argmin(np.abs(_t-2.0))

    time = _t[ind0:ind1]
    c_side = _c_side[ind0:ind1]
    b_sideu = _b_sideu[ind0:ind1]
    b_top = _b_top[ind0:ind1]

    # ninja system is on a different time base than the other measurements
    ninja2 = interp1d(plen_bot_time, plen_bot, bounds_error=False)(time)
    
    gas_tot = c_side + b_sideu + b_top + ninja2

    if plot:
        fig,ax = plt.subplots()
        ax.plot(time, gas_tot, label='total')
        ax.plot(time, c_side, label='c-side')
        ax.plot(time, b_sideu, label='b-side u')
        ax.plot(time, b_top, label='b-top')
        ax.plot(time, ninja2, label='ninja2')
        ax.legend(loc='best').set_draggable(True)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Total injected gas [Torr-l]')
        
    return time, gas_tot

    
def get_Lya_data(shot=1080416024, systems=['LYMID'], plot=True):
    ''' Get Ly-alpha data for C-Mod from any (or all) of the systems:
    ['LYMID','WB1LY','WB4LY','LLY','BPLY']
    '''

    bdata = {} # BRIGHT node
    edata = {} # EMISS node

    if systems=='all':
        systems=['LYMID','WB1LY','WB4LY','LLY','BPLY']
        
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(13,8))
        ls = ['-','--','-.',':','--']

    for ss,system in enumerate(systems):
        fetched_0=True; fetched_1=True
        try:
            bdata[system] = fetch_bright(shot, system)
            
            if plot:
                for ch in np.arange(bdata[system].values.shape[1]):
                    ax[0].plot(bdata[system].time, bdata[system].values[:,ch],
                               label=system+', '+str(ch), ls=ls[ss])
        except:
            print('Could not fetch C-Mod Ly-alpha BRIGHT data from system '+system)
            fetched_0=False
            pass
        
        try:
            edata[system] = fetch_emiss(shot, system)
            if plot:
                for ch in np.arange(edata[system].values.shape[1]):
                    ax[1].plot(edata[system].time, edata[system].values[:,ch],
                               label=system+', '+str(ch), ls=ls[ss])
        except:
            print('Could not fetch C-Mod Ly-alpha EMISS data from system '+system)
            fetched_1=False
            pass
        
        if plot:
            ax[0].set_xlabel('time [s]'); ax[1].set_xlabel('time [s]')
            if fetched_0: ax[0].set_ylabel(r'Brightness [$'+str(bdata[system].units)+'$]')
            if fetched_1: ax[1].set_ylabel(r'Emissivity [$'+str(edata[system].units)+'$]')
            ax[0].legend(); ax[1].legend()
            
    return bdata,edata


def fetch_bright(shot,system):

    _bdata = {}
    node = omfit_mds.OMFITmdsValue(server='CMOD', shot=shot, treename='SPECTROSCOPY',
                         TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
                             '{:s}:BRIGHT'.format(system))
    _bdata = xarray.DataArray(
        node.data(), coords={'time':node.dim_of(1),'R':node.dim_of(0)},
        dims=['time','R'],
        attrs={'units': node.units()})

    return _bdata


def fetch_emiss(shot,system):

    _edata = {}
    node = omfit_mds.OMFITmdsValue(server='CMOD', shot=shot, treename='SPECTROSCOPY',
                         TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
                             '{:s}:EMISS'.format(system))
    _edata = xarray.DataArray(
        node.data(), coords={'time':node.dim_of(1),'R':node.dim_of(0)},
        dims=['time','R'],
        attrs={'units': node.units()})

    #print('Emissivity units: ' , node.units())

    return _edata


def fetch_tomo_emiss(shot,system,r_end=0.93,sys_err=5,shift=0):

    _out = tomo.tomoCMOD(shot,system,r_end=r_end,sys_err=sys_err)
    tvec,R_grid,y,y_err,backprojection = _out

    _edata = xarray.DataArray(
        y, coords={'time':tvec,'R':R_grid+shift},
        dims=['time','R',],
        attrs={'units': '$W/m^{3}$'})

    return _edata, y_err


def get_CMOD_1D_geom(shot,time):

    # right gap
    tmp = omfit_mds.OMFITmdsValue(server='CMOD', treename='ANALYSIS', shot=shot,
                        TDI='\\ANALYSIS::TOP.EFIT.RESULTS.A_EQDSK.ORIGHT')
    time_vec = tmp.dim_of(0)
    _gap_R = tmp.data()
    gap_R = _gap_R[time_vec.searchsorted(time)-1]

    # R location of LFS LCFS
    tmp = omfit_mds.OMFITmdsValue(server='CMOD', treename='ANALYSIS', shot=shot,
                        TDI='\\ANALYSIS::TOP.EFIT.RESULTS.G_EQDSK.RBBBS')
    time_vec = tmp.dim_of(0)
    _rbbbs = tmp.data()*1e2 # m --> cm
    rbbbs = _rbbbs[:,time_vec.searchsorted(time)-1]

    Rsep = np.max(rbbbs)

    return Rsep,gap_R




def plot_emiss(edata, shot, time, ax=None):
    ''' Plot emissivity profile '''

    # get Rsep and gap
    Rsep, gap = get_CMOD_1D_geom(shot,time)
    Rwall = Rsep+gap
    print('Rwall,Rsep,gap:',Rwall, Rsep,gap)

    if ax is None:
        fig,ax = plt.subplots()

    tidx = np.argmin(np.abs(edata.time.values - time))
    ax.plot(edata.R.values, edata.values[tidx,:], '.-')  #*100 - Rwall

    ax.set_ylabel(r'emissivity [${:}$]'.format(edata.units))
    ax.set_xlabel(r'R [cm]')
    return ax


def plot_bright(bdata, shot, time,ax=None):
    ''' Plot brightness over chords profile '''

    # get Rsep and gap
    Rsep, gap = get_CMOD_1D_geom(shot,time)
    Rwall = Rsep+gap
    print('Rwall,Rsep,gap:',Rwall, Rsep,gap)

    if ax is None:
        fig,ax = plt.subplots()

    tidx = bdata.time.values.searchsorted(time)-1
    mask = np.nonzero(bdata.values[tidx,:])[0]
    ax.plot(bdata.R.values[mask], bdata.values[tidx,mask], '.-')  #*100-Rwall

    ax.set_ylabel(r'brightness [${:}$]'.format(bdata.units))
    ax.set_xlabel(r'R [cm]')

    return ax

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_P_ohmic(shot):
    ''' Get Ohmic power

    Translated/adapted from scopes:
    _vsurf =  deriv(smooth1d(\ANALYSIS::EFIT_SSIBRY,2))*$2pi ;
    _ip=abs(\ANALYSIS::EFIT_AEQDSK:CPASMA);
    _li = \analysis::efit_aeqdsk:ali;
    _L = _li*6.28*67.*1.e-9;
    _vi = _L*deriv(smooth1d(_ip,2));
    _poh=_ip*(_vsurf-_vi)/1.e6
    '''

    # psi at the edge:
    ssibry_node = OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::efit_ssibry')
    time = ssibry_node.dim_of(0)
    ssibry = ssibry_node.data()
    
    # total voltage associated with magnetic flux inside LCFS
    vsurf = np.gradient(smooth(ssibry,5),time) * 2 * np.pi

    # calculated plasma current
    ip_node= OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::EFIT_AEQDSK:CPASMA')
    ip = np.abs(ip_node.data())

    # internal inductance
    li = OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::EFIT_AEQDSK:ali').data()

    R_cm = 67.0 # value chosen/fixed in scopes
    L = li*2.*np.pi*R_cm*1e-9  # total inductance (nH)
    
    vi = L * np.gradient(smooth(ip,2),time)   # induced voltage
    
    P_oh = ip * (vsurf - vi)/1e6 # P=IV   #MW
    return time, P_oh

    
def get_CMOD_var(var,shot, tmin=None, tmax=None, plot=False):
    ''' Get tree variable for a CMOD shot. If a time window is given, the value averaged over that window is returned,
    or else the time series is given.  See list below for acceptable input variables.
    '''

    if var=='Bt':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='magnetics',TDI='\\magnetics::Bt')
    elif var=='Bp':
        # use Bpolav, average poloidal B field --> see definition in Silvagni NF 2020
        node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:bpolav')
    elif var=='Ip':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='magnetics',TDI='\\magnetics::Ip')
    elif var=='nebar':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons',TDI='\\electrons::top.tci.results:nl_04')
    elif var=='P_RF':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='RF',TDI='\\RF::RF_power_net')
    elif var=='P_ohmic':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='RF',TDI='\\RF::RF_power_net')
    elif var=='P_rad':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='spectroscopy',TDI='\\spectroscopy::top.bolometer:twopi_diode') # kW
    elif var=='p_D2':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::TOP.GAS.RATIOMATIC.F_SIDE')  # mTorr
    elif var=='p_E_BOT_MKS':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::E_BOT_MKS')  # mTorr   #lower divertor
    elif var=='p_B_BOT_MKS':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::B_BOT_MKS')     # mTorr  # lower divertor
    elif var=='p_F_CRYO_MKS':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::F_CRYO_MKS')     # mTorr, only post 2006
    elif var=='q95':
        node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:qpsib')
    elif var=='Wmhd':
        node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:wplasm')
    elif var=='areao':
        node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:areao')
    elif var=='betat':
        node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:betat')
    elif var=='betap':
        node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:betap')
    elif var=='P_oh':
        t,data = get_P_ohmic(shot)   # accurate routine to estimate Ohmic power
    elif var=='h_alpha':
        node = OMFITmdsValue(server='CMOD',shot=shot, treename='spectroscopy', TDI='\ha_2_bright')
    elif var=='cryo_on':
        node = OMFITmdsValue(server='CMOD',shot=shot, treename='EDGE', TDI='\EDGE::TOP.CRYOPUMP:MESSAGE')
    elif var=='ssep':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:ssep')
    else:
        raise ValueError('Variable '+var+' was not recognized!')

    if var not in ['P_oh']: 
        data = node.data()
        t = node.dim_of(0)

        if var=='p_E_BOT_MKS' or var=='p_B_BOT_MKS' or var=='p_F_CRYO_MKS':  # anomalies in data storage
            data = data[0,:]
    
    if var=='P_rad':
        # From B.Granetz's matlab scripts: factor of 4.5 from cross-calibration with 2pi_foil during flattop
        # NB: Bob's scripts mention that this is likely not accurate when p_rad (uncalibrated) <= 0.5 MW
        data *= 4.5
        # data from the twopi_diode is output in kW. Change to MW for consistency
        data /= 1e3
        
    if plot:
        plt.figure()
        plt.plot(t,data)
        plt.xlabel('time [s]')
        plt.ylabel(var)

    if tmin is not None and tmax is not None:
        tidx0 = np.argmin(np.abs(t - tmin))
        tidx1 = np.argmin(np.abs(t - tmax))
        return np.mean(data[tidx0:tidx1])
    else:
        return t,data


def load_fmp_neTe(shot, get_max=False, plot=False):
    '''Load slow ne and Te from Flush Mounted Probes (FMP) on the divertor from the nodes
    \EDGE::top.probes.fmp.osd_0{ii}.p0.*e_Slow

    If get_max=True, returns the maximum of all the loaded probe signals over time. Otherwise, return
    individual signals.
    '''
    ne_fmp = []
    Te_fmp = []
    t_fmp = []

    ii=1
    while True:
        node_ne = OMFITmdsValue(server='CMOD',shot=shot,treename='EDGE', TDI=f'\EDGE::top.probes.fmp.osd_0{ii}.p0.ne_Slow')
        node_Te = OMFITmdsValue(server='CMOD',shot=shot,treename='EDGE', TDI=f'\EDGE::top.probes.fmp.osd_0{ii}.p0.Te_Slow')

        if node_ne.data() is None:
            break
        
        ne_fmp.append(node_ne.data())
        t_fmp.append(node_ne.dim_of(0))
        Te_fmp.append(node_Te.data())
        ii+=1

    
    if get_max:
        ne_fmp_interp = np.zeros((len(ne_fmp),200)) # 200 time points is enough, usually ~100 in signals
        Te_fmp_interp = np.zeros((len(Te_fmp),200)) # 200 time points is enough, usually ~100 in signals
        
        # each probe has a different time base. Interpolate and then sum
        tmin = np.min([np.min(tlist) for tlist in t_fmp])
        tmax= np.max([np.max(tlist) for tlist in t_fmp])
        time = np.linspace(tmin, tmax, ne_fmp_interp.shape[1])
        
        for ii in np.arange(len(ne_fmp)):
            ne_fmp_interp[ii,:] = interp1d(t_fmp[ii], ne_fmp[ii], bounds_error=False)(time)
            Te_fmp_interp[ii,:] = interp1d(t_fmp[ii], Te_fmp[ii], bounds_error=False)(time)
            
        ne_fmp_max = np.nanmax(ne_fmp_interp, axis=0)
        Te_fmp_max = np.nanmax(Te_fmp_interp, axis=0)

        if plot:
            fig,ax = plt.subplots()
            ax.plot(time, ne_fmp_max)
            ax.set_xlabel('time [s]')
            ax.set_ylabel(r'$n_e$ FMP max [m$^{-3}$]')
            
            fig,ax = plt.subplots()
            ax.plot(time, Te_fmp_max)
            ax.set_xlabel('time [s]')
            ax.set_ylabel(r'$T_e$ FMP max [eV]')

        return time, ne_fmp_max, Te_fmp_max
    
    if not get_max and plot:
        fig1,ax1 = plt.subplots()
        fig2,ax2 = plt.subplots()
        
        for elem in np.arange(len(ne_fmp)):
            ax1.plot(t_fmp[elem], ne_fmp[elem], label=f'{elem}')
            ax2.plot(t_fmp[elem], Te_fmp[elem], label=f'{elem}')
            
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel(r'$n_e$ [m$^{-3}$]')
        plt.tight_layout(); plt.legend()
        ax2.set_xlabel('time [s]')
        ax2.set_ylabel(r'$T_e$ [m$^{-3}$]')
        plt.tight_layout(); plt.legend()

    return t_fmp, ne_fmp, Te_fmp


def Lya_to_ion_rate(emiss_prof, ne, Te, ni=None, plot=True, rhop=None,
                    rates_source='adas', axs=None):
    '''Estimate ionization rate measured from ground state density and emissivity profiles.'''


    assert len(emiss_prof)==len(ne) and len(ne)==len(Te)
    if ni is None:
        ni=copy.deepcopy(ne)
    else:
        assert len(ne)==len(ni)

    nn,ax = aurora.Lya_to_neut_dens(emiss_prof, ne, Te, plot=False, rhop=rhop, rates_source=rates_source)

    atom_data = aurora.get_atom_data('H')
    ion_func = aurora.interp_atom_prof(atom_data['scd'], np.log10(ne), np.log10(Te), x_multiply=True)
    ion_rate = ion_func[:,0]*nn

    return ion_rate,ax




