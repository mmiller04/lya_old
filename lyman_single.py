'''
Obtain C-Mod neutral density profiles for a single shot and time interval. 
sciortino, Aug 2020
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import os, copy
#import fit_2D
import pickle as pkl
from scipy.interpolate import interp1d, RectBivariateSpline
import lyman_data
import aurora
from IPython import embed
import mtanh_fitting
import eqtools


def single_case(shot,tmin,tmax, roa_kp, ne, ne_std, Te, Te_std,
                p_ne, p_Te, gfiles_loc=None, lya_shift=0,
                tomo_inversion=True, zero_pos=0.93, tomo_err=5,
                SOL_exp_decay=True, decay_from_LCFS=False):
    ''' Process Lyman-alpha data for a single shot/time interval. 

    Parameters
    ----------
    shot : int
        CMOD shot number
    tmin : float
        Lower bound of time window
    tmax : float
        Upper bound of time window
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
    gfiles_loc : str or None
        Location to save g-file
    lya_shift : float
        Allows for manual setting of relative shift between Ly-alpha data and kinetic profiles.
        Comes from uncertainty in EFIT position of separatrix.
    tomo_inversion : bool
        If True, use Tomas' tomographic inversion (else use data from tree - Matt's inversion)
    zero_pos : float
        Position at which 0 for brightness data is placed if using Tomas' tomographic inversion
    tomo_err : float
        Systematic error applied to brightness data for Tomas' tomographic inversion (in units of %)
    SOL_exp_decay : bool
        If True, apply an exponential decay (fixed to 1cm length scale) in the region outside of the 
        last radius covered by experimental data.
    decay_from_LCFS : bool
        If True, force exponential decay everywhere outside of the LCFS, ignoring any potential
        data in that region.

    Returns
    -------
    R : 1D array
        Major radius grid
    roa : 1D array
        r/a grid
    rhop : 1D array
        rhop grid
    ne_prof : 1D array
        Interpolated electron density in units of :math:`cm^{-3}`
    ne_prof_unc : 1D array
        Interpolated electron density uncertaities in units of :math:`cm^{-3}`
    Te_prof : 1D array
        Interpolated electron temperature in units of :math:`eV`
    Te_prof_unc : 1D array
        Interpolated electron temperature uncertainties in units of :math:`eV`
    nn_prof : 1D array
        Neutral density in units of :math:`cm^{-3}`
    nn_prof_unc : 1D array
        Neutral density uncertainties in units of :math:`cm^{-3}`
    emiss_prof : 1D array
        Emissivity in units of :math:` W/cm^3`
    emiss_prof_unc : 1D array
        Emissivity uncertainties in units of :math:` W/cm^3`
    emiss_min : float
        Minimum emissivity imposed internally. Ideally, this should be an argument in the future, not a returned value.
    Te_min : float
        Minimum Te imposed internally. Ideally, this should be an argument in the future, not a returned value.

    '''
    time = (tmax+tmin)/2.
    
    ne_decay_len_SOL = 0.01 # m
    Te_decay_len_SOL = 0.01 # m

    # transform coordinates:
    if gfiles_loc is None:
        gfiles_loc = '/home/sciortino/EFIT/gfiles/'
    
    # convert using aurora rad_coord_transform function
    #geqdsk = lyman_data.get_geqdsk_cmod(shot,time*1e3,gfiles_loc=gfiles_loc)
    #geqdsk = lyman_data.get_geqdsk_cmod(shot,time*1e3,gfiles_loc=gfiles_loc) # read it back in (second time to prevent funny issues)
    #R_kp = aurora.rad_coord_transform(roa_kp, 'r/a', 'rhop', geqdsk)
    #Rsep = aurora.rad_coord_transform(1.0, 'r/a', 'Rmid', geqdsk)
    #print('EFIT R_sep',Rsep)

    # try to convert using eqtools rho2rho
    eq = eqtools.CModEFITTree(shot)
    R_kp = eq.rho2rho('r/a', 'psinorm', roa_kp, time, sqrt=True) # returns rhop if sqrt = True
    Rsep = eq.rho2rho('r/a', 'Rmid', 1, time)
    print('EFIT R_sep',Rsep)


    # exponential decays of kinetic profs from last point of experimental data:
    # ne and Te profiles can miss last point depending on filtering?
    max_roa_expt = np.maximum(np.max(p_ne.X[:,0]), np.max(p_Te.X[:,0]))  
    #max_rhop_expt = float(aurora.rad_coord_transform(max_roa_expt, 'r/a', 'rhop', geqdsk))
    max_rhop_expt = float(eq.rho2rho('r/a', 'psinorm', max_roa_expt, time, sqrt=True))
    print('Experimental TS data extending to r/a={:.4},rhop={:.4}'.format(max_roa_expt,max_rhop_expt))

    indLCFS = np.argmin(np.abs(R_kp - Rsep))
    ind_max = np.argmin(np.abs(roa_kp - max_roa_expt))
    if SOL_exp_decay and decay_from_LCFS: # exponential decay from the LCFS
        ind_max = indLCFS
        
    ne_std_av = copy.deepcopy(ne_std)
    Te_std_av = copy.deepcopy(Te_std)
    
    if SOL_exp_decay:
        # apply exponential decay outside of region covered by data
        ne_sol = ne[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/ne_decay_len_SOL)
        ne_av = np.concatenate((ne[:ind_max], ne_sol))
        Te_sol = Te[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/Te_decay_len_SOL)
        Te_av = np.concatenate((Te[:ind_max], Te_sol))

        # set all ne/Te std outside of experimental range to mean of outer-most values
        edge_unc = np.mean(ne_std[ind_max-3:ind_max])
        edge_unc = np.mean(Te_std[ind_max-3:ind_max])
        ne_std_av[ind_max:] = edge_unc if edge_unc<5e13 else 5e13
        Te_std_av[ind_max:] = edge_unc if edge_unc<30e-3 else 30e-3
    else:
        ne_av = copy.deepcopy(ne)
        Te_av = copy.deepcopy(Te)

    # no uncertainties larger than 30 eV outside of LCFS
    Te_std_av[np.logical_and(R_kp>Rsep, Te_std_av>30e-3)] = 30e-3
    #Te_std_av_lcfs = Te_std_av[indLCFS:]
    #Te_std_av_lcfs[Te_std_av_lcfs>30e-3] = 30e-3

    # set ne to cm^-3 and Te in eV
    ne_av *= 1e14
    ne_std_av *= 1e14
    Te_av *= 1e3
    Te_std_av *= 1e3

    # set appropriate minima
    Te_min=10.0    # intended to be a better approximation overall than 3eV
    Te_av[Te_av<Te_min] = Te_min  
    ne_av[ne_av<1e12] = 1e12

    # load Lyman emissivity from MDS+
    if tomo_inversion:
        _out = lyman_data.fetch_tomo_emiss(shot, 'LYMID', r_end=zero_pos, sys_err=tomo_err, shift=lya_shift)
        edata, emiss_err = _out
    else:
        edata = lyman_data.fetch_emiss(shot, 'LYMID')

    # time average through 50 ms for each point
    interp = RectBivariateSpline(
        edata.R.values, edata.time.values, edata.values.T, s=0 # no smoothing
        )
    time_vec_av = np.linspace(tmin,tmax,100)
    _emiss_ = interp(edata.R.values,time_vec_av)

    _emiss_prof = np.mean(_emiss_, axis=1)*1e-6  # W/m^3 --> W/cm^3
    # tomo inversion returns error from inversion, so no need to recalculate it
    if tomo_inversion:
        interp_err = RectBivariateSpline(
            edata.R.values, edata.time.values, emiss_err.T, s=0 # no smoothing
            )
        _emiss_unc_ = interp_err(edata.R.values,time_vec_av)
        _emiss_prof_unc = np.mean(_emiss_unc_, axis=1)*1e-6
    else:
        _emiss_prof_unc = np.std(_emiss_, axis=1)*1e-6  # W/m^3 --> W/cm^3
    
    # interpolate kinetic profiles on emissivity radial grid
    R = np.linspace(np.min(edata.R.values), np.max(edata.R.values), 200)
    #roa = aurora.rad_coord_transform(R, 'Rmid', 'r/a', geqdsk)
    roa = eq.rho2rho('Rmid', 'r/a', R, time)
    #rhop = aurora.rad_coord_transform(R, 'Rmid', 'rhop', geqdsk)
    rhop = eq.rho2rho('Rmid', 'psinorm', R, time, sqrt=True)

    print('Ly-a data extending r/a={:.3}-{:.3},rhop={:.3}-{:.3f}'.format(np.min(roa),np.max(roa),
                                                                         np.min(rhop),np.max(rhop)))

    # interpolate emissivity on a finer radial grid
    emiss_prof = interp1d(edata.R.values, _emiss_prof)(R)
    emiss_prof_unc = interp1d(edata.R.values, _emiss_prof_unc)(R)

    # Don't believe too small Ly-a emissivities
    emiss_min = 5e-3 # W/cm^3
    #emiss_prof[emiss_prof<emiss_min] = np.nan

    # explore sensitivity of nn_prof within 1 sigma of ne and Te  
    ne_prof = np.exp(interp1d(roa_kp,np.log(ne_av), bounds_error=False, fill_value=None)(roa))
    Te_prof = interp1d(roa_kp,Te_av, bounds_error=False, fill_value=None)(roa)
   
    ne_prof_up = np.exp(interp1d(roa_kp,np.log(ne_av+ne_std_av), bounds_error=False, fill_value=None)(roa))
    Te_prof_up = interp1d(roa_kp,Te_av+Te_std_av, bounds_error=False, fill_value=None)(roa)
    
    ne_prof_down = np.exp(interp1d(roa_kp,np.log(np.maximum(ne_av-ne_std_av,1e12)),
                                   bounds_error=False, fill_value=None)(roa))
    Te_prof_down = interp1d(roa_kp,np.maximum(Te_av-Te_std_av,Te_min),
                            bounds_error=False, fill_value=None)(roa)

    # useful for uncertainty estimation on nn/ne:
    ne_prof_unc = interp1d(roa_kp,ne_std_av, bounds_error=False, fill_value=None)(roa)
    Te_prof_unc = interp1d(roa_kp,Te_std_av, bounds_error=False, fill_value=None)(roa)

    
    ## neutral density

    # mean profile
    nn_prof,ax = aurora.Lya_to_neut_dens(
        emiss_prof, ne_prof, Te_prof, plot=False, rhop=rhop)

    # testing up and down shifts of ne,Te profiles within uncertainties
    nn_prof_uu,ax = aurora.Lya_to_neut_dens(emiss_prof, ne_prof_up, Te_prof_up,
                                            plot=False, rhop=rhop)
    nn_prof_ud,ax = aurora.Lya_to_neut_dens(emiss_prof, ne_prof_up, Te_prof_down,
                                            plot=False, rhop=rhop)
    nn_prof_du,ax = aurora.Lya_to_neut_dens(emiss_prof, ne_prof_down, Te_prof_up,
                                            plot=False, rhop=rhop)
    nn_prof_dd,ax = aurora.Lya_to_neut_dens(emiss_prof, ne_prof_down, Te_prof_down,
                                            plot=False, rhop=rhop)

    nn_prof_low = np.min([nn_prof_uu,nn_prof_ud,nn_prof_du,nn_prof_dd],axis=0)
    nn_prof_high = np.max([nn_prof_uu,nn_prof_ud,nn_prof_du,nn_prof_dd],axis=0)
    nn_prof_unc1 = (nn_prof_high-nn_prof_low)/2.

    # linear propagation of uncertainty:
    nn_prof_unc = nn_prof * np.sqrt((nn_prof_unc1/nn_prof)**2+(emiss_prof_unc/emiss_prof)**2)

    
    ## ionization rate

    # mean_profile
    ion_rate_prof,ax = lyman_data.Lya_to_ion_rate(
            emiss_prof, ne_prof, Te_prof, plot=False, rhop=rhop)

    # testing up and down shifts of ne,Te profiles within uncertainties
    ion_prof_uu,ax = lyman_data.Lya_to_ion_rate(emiss_prof, ne_prof_up, Te_prof_up,
                                                plot=False, rhop=rhop)
    ion_prof_ud,ax = lyman_data.Lya_to_ion_rate(emiss_prof, ne_prof_up, Te_prof_down,
                                                plot=False, rhop=rhop)
    ion_prof_du,ax = lyman_data.Lya_to_ion_rate(emiss_prof, ne_prof_down, Te_prof_up,
                                                plot=False, rhop=rhop)
    ion_prof_dd,ax = lyman_data.Lya_to_ion_rate(emiss_prof, ne_prof_down, Te_prof_down,
                                                plot=False, rhop=rhop)

    ion_prof_low = np.min([ion_prof_uu,ion_prof_ud,ion_prof_du,ion_prof_dd],axis=0)
    ion_prof_high = np.max([ion_prof_uu,ion_prof_ud,ion_prof_du,ion_prof_dd],axis=0)
    ion_prof_unc1 = (ion_prof_high-ion_prof_low)/2.

    # linear propagation of uncertainty:
    ion_rate_prof_unc = ion_rate_prof * np.sqrt((ion_prof_unc1/ion_rate_prof)**2+(emiss_prof_unc/emiss_prof)**2)
   

    ## calculate also from raw data points
    # set ne to cm^-3 and Te in eV

    ne_raw = p_ne.y*1e14
    ne_err_raw = p_ne.err_y*1e14
    ne_roa_raw = p_ne.X[:,0]

    Te_raw = p_Te.y*1e3
    Te_err_raw = p_Te.err_y*1e3
    Te_roa_raw = p_Te.X[:,0]
    
    # map Te onto ne points
    roa_raw = ne_roa_raw
    Te_raw = interp1d(Te_roa_raw, Te_raw, fill_value='extrapolate')(roa_raw)
    Te_err_raw = interp1d(Te_roa_raw, Te_err_raw, fill_value='extrapolate')(roa_raw)

    # map kps and emiss to midplane
    R_raw = eq.rho2rho('r/a', 'Rmid', roa_raw, time)
    rhop_raw = eq.rho2rho('r/a', 'psinorm', roa_raw, time, sqrt=True)

    emiss_mapped = interp1d(R, emiss_prof, bounds_error=False)(R_raw)
    emiss_unc_mapped = interp1d(R, emiss_prof_unc, bounds_error=False)(R_raw)

    # repeat same procedure as above to quantiy errors in nn/ion_rate
    
    # don't need to interpolate these since calculating on kp grid
    ne_raw_up = ne_raw+ne_err_raw
    Te_raw_up = Te_raw+Te_err_raw 

    ne_raw_down = np.maximum(ne_raw-ne_err_raw,1e12)
    Te_raw_down = np.maximum(Te_raw-Te_err_raw,Te_min)

    ## neutral density

    # mean profile
    nn_raw,ax = aurora.Lya_to_neut_dens(
        emiss_mapped, ne_raw, Te_raw, plot=False, rhop=rhop_raw)
   
    # testing up and down shifts of ne,Te profiles within uncertainties
    nn_raw_uu,ax = aurora.Lya_to_neut_dens(emiss_mapped, ne_raw_up, Te_raw_up,
                                            plot=False, rhop=rhop_raw)
    nn_raw_ud,ax = aurora.Lya_to_neut_dens(emiss_mapped, ne_raw_up, Te_raw_down,
                                            plot=False, rhop=rhop_raw)
    nn_raw_du,ax = aurora.Lya_to_neut_dens(emiss_mapped, ne_raw_down, Te_raw_up,
                                            plot=False, rhop=rhop_raw)
    nn_raw_dd,ax = aurora.Lya_to_neut_dens(emiss_mapped, ne_raw_down, Te_raw_down,
                                            plot=False, rhop=rhop_raw)

    nn_raw_low = np.min([nn_raw_uu,nn_raw_ud,nn_raw_du,nn_raw_dd],axis=0)
    nn_raw_high = np.max([nn_raw_uu,nn_raw_ud,nn_raw_du,nn_raw_dd],axis=0)
    nn_raw_unc1 = (nn_raw_high-nn_raw_low)/2.

    # linear propagation of uncertainty:
    nn_raw_unc = nn_raw * np.sqrt((nn_raw_unc1/nn_raw)**2+(emiss_unc_mapped/emiss_mapped)**2)


    ## ionization rate

    # mean profile 
    ion_rate_raw,ax = lyman_data.Lya_to_ion_rate(
            emiss_mapped, ne_raw, Te_raw, plot=False, rhop=rhop_raw)

    # testing up and down shifts of ne,Te profiles within uncertainties
    ion_raw_uu,ax = lyman_data.Lya_to_ion_rate(emiss_mapped, ne_raw_up, Te_raw_up,
                                                plot=False, rhop=rhop)
    ion_raw_ud,ax = lyman_data.Lya_to_ion_rate(emiss_mapped, ne_raw_up, Te_raw_down,
                                                plot=False, rhop=rhop)
    ion_raw_du,ax = lyman_data.Lya_to_ion_rate(emiss_mapped, ne_raw_down, Te_raw_up,
                                                plot=False, rhop=rhop)
    ion_raw_dd,ax = lyman_data.Lya_to_ion_rate(emiss_mapped, ne_raw_down, Te_raw_down,
                                                plot=False, rhop=rhop)

    ion_raw_low = np.min([ion_raw_uu,ion_raw_ud,ion_raw_du,ion_raw_dd],axis=0)
    ion_raw_high = np.max([ion_raw_uu,ion_raw_ud,ion_raw_du,ion_raw_dd],axis=0)
    ion_raw_unc1 = (ion_raw_high-ion_raw_low)/2.

    # linear propagation of uncertainty:
    ion_rate_raw_unc = ion_rate_raw * np.sqrt((ion_raw_unc1/ion_rate_raw)**2+(emiss_unc_mapped/emiss_mapped)**2)

    return R, roa, rhop, ne_prof,ne_prof_unc, Te_prof, Te_prof_unc,\
        nn_prof, nn_prof_unc, ion_rate_prof, ion_rate_prof_unc,\
        R_raw, roa_raw, rhop_raw,\
        nn_raw, nn_raw_unc, ion_rate_raw, ion_rate_raw_unc,\
        emiss_prof, emiss_prof_unc, emiss_mapped, emiss_unc_mapped, emiss_min, Te_min






if __name__=='__main__':
    # I-mode:
    #shot=1080416025   # 24/25
    #tmin=0.8 #1.2 # gas puffs around 1.0s
    #tmax=1.4

    # problematic
    #shot=1120917011
    #tmin=0.9
    #tmax=1.1

    # L-mode:
    #shot=1100308004
    #tmin=0.7
    #tmax=1.4

    # EDA H-mode:
    #shot=1100305023
    #tmin=0.85
    #tmax=1.3
    
    # Other L-modes:
    #shot=1080110005
    #tmin=1.0
    #tmax=1.3

    # L-mode:
    shot=1070511002
    tmin=0.7
    tmax=1.4

    gfiles_loc = '.'
    kp_out = lyman_data.get_cmod_kin_profs(shot,tmin,tmax,gfiles_loc=gfiles_loc)
    roa_kp, ne, ne_std, Te, Te_std, p_ne, p_Te, kpnum = kp_out
    
    #geqdsk = lyman_data.get_geqdsk_cmod(shot,(tmin+tmax)/2.*1e3,gfiles_loc=gfiles_loc)
    eq = eqtools.CModEFITTree(shot)
    time = (tmin+tmax)/2

    # gather Lyman-alpha profiles:
    res = single_case(shot,tmin,tmax,roa_kp, ne, ne_std, Te, Te_std,
                      p_ne,p_Te, gfiles_loc=gfiles_loc, tomo_inversion=True,
                      SOL_exp_decay=False)

    R, roa, rhop, ne_prof,ne_prof_unc, Te_prof, Te_prof_unc,\
        nn_prof, nn_prof_unc, ion_rate_prof, ion_rate_prof_unc,\
        R_raw, roa_raw, rhop_raw,\
        nn_raw, nn_raw_unc, ion_rate_raw, ion_rate_raw_unc,\
        emiss_prof, emiss_prof_unc, emiss_mapped, emiss_mapped_unc, emiss_min, Te_min = res


    #####
    grad_logn0 = np.gradient(np.log10(np.maximum(nn_prof,1e-10)), R)

    # estimate neutral scale length and radial sensitivity as a std between rhop=[0.97,0.99]
    def interp_to_rhop(rhop_fine):
        return float(interp1d(rhop, 1./(grad_logn0+1e-10),bounds_error=False)(rhop_fine))*1e3 # m-->mm
    L0_ped = np.mean(list(map(interp_to_rhop, np.linspace(0.97,0.99,100))))
    L0_ped_unc = np.std(list(map(interp_to_rhop, np.linspace(0.97,0.99,100))))

    nn_by_ne_prof = nn_prof/ne_prof
    # ne uncertainty in the SOL also goes to ne<0.... ignore it
    nn_by_ne_prof_unc = np.sqrt((nn_prof_unc/ne_prof)**2) #+(nn_prof/ne_prof**2)**2*ne_prof_unc**2)  

    geqdsk = lyman_data.get_geqdsk_cmod(shot,(tmin+tmax)/2.*1e3,gfiles_loc=gfiles_loc)
    
    style=1
    #fig,ax = plt.subplots(4,1, figsize=(8,12), sharex=True)
    num=10

    ff = 1./np.log(10.)
   
    # plot only n0 and n0/ne
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].plot(rhop, np.log10(nn_prof), c='k')
    for ij in np.arange(num):
        ax[0].fill_between(rhop, np.log10(nn_prof)+3*ff*nn_prof_unc/nn_prof*ij/num,
                           np.log10(nn_prof)-3*ff*nn_prof_unc/nn_prof*ij/num, alpha=0.3*(1.-ij/num), color='k')
        
    ax[1].plot(rhop, np.log10(nn_by_ne_prof),c='k')
    for ij in np.arange(num):
        ax[1].fill_between(rhop,np.log10(nn_by_ne_prof)+3*ff*nn_by_ne_prof_unc/nn_by_ne_prof*ij/num,
                           np.log10(nn_by_ne_prof)-3*ff*nn_by_ne_prof_unc/nn_by_ne_prof*ij/num, alpha=0.3*(1.-ij/num),
                           color='k')
    
    ax[0].set_ylabel(r'$log_{10}(n_n$ [$cm^{-3}$])')
    ax[1].set_ylabel(r'$log_{10}(n_n/n_e)$')
    ax[1].set_xlabel(r'$\rho_p$')
    ax[0].set_xlabel(r'$\rho_p$')
    fig.suptitle(f'C-Mod shot {shot}')
    fig.tight_layout()

    # complete analysis layout:
    ne_prof[ne_prof<1e10] = 1e10
    nn_prof[nn_prof<1.] = 1.0
    nn_by_ne_prof[nn_by_ne_prof<1e-8] = 1e-8

    #############

    fig,ax = plt.subplots(4,1, figsize=(8,12), sharex=True)
    ax[0].plot(rhop, np.maximum(emiss_prof,0),'k')
    ax[0].axhline(emiss_min, c='r', ls='--')

    for ij in np.arange(num):
        ax[0].fill_between(rhop, emiss_prof+3*emiss_prof_unc*ij/num,
                           np.maximum(emiss_prof-3*emiss_prof_unc*ij/num,0), alpha=0.3*(1.-ij/num), color='k')

    ax[1].plot(rhop,np.maximum(ne_prof,1e11), c='b')
    for ij in np.arange(num):
        ax[1].fill_between(rhop, ne_prof+3*ne_prof_unc*ij/num,
                           np.maximum(ne_prof-3*ne_prof_unc*ij/num,1e11), alpha=0.1*(1.-ij/num), color='b')
    
    ne_ts_mask = p_ne.X[:,0] > np.min(rhop); ne_ts_mask[kpnum['ne']['TS']:] = False
    ne_sp_mask = p_ne.X[:,0] > np.min(rhop); ne_sp_mask[:kpnum['ne']['TS']] = False
    # aurora transform doesn't seem to work outside of LCFS?
    #p_ne_rhop = aurora.rad_coord_transform(p_ne.X[mask,0], 'r/a', 'rhop', geqdsk)
    p_ne_rhop = eq.rho2rho('r/a', 'psinorm', p_ne.X[:,0], time, sqrt=True)
    ax[1].errorbar(p_ne_rhop[ne_sp_mask], p_ne.y[ne_sp_mask]*1e14, p_ne.err_y[ne_sp_mask]*1e14, color='navy', fmt='.')
    ax[1].errorbar(p_ne_rhop[ne_ts_mask], p_ne.y[ne_ts_mask]*1e14, p_ne.err_y[ne_ts_mask]*1e14, color='royalblue', fmt='.')

    ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(rhop, np.maximum(Te_prof,Te_min), color='r')
    for ij in np.arange(num):
        ax2.fill_between(rhop, Te_prof+3*Te_prof_unc*ij/num,
                         np.maximum(Te_prof-3*Te_prof_unc*ij/num,Te_min), color='r', alpha=0.1*(1.-ij/num))
    ax2.tick_params(axis='y', labelcolor='r')
    
    Te_ts_mask = p_Te.X[:,0] > np.min(rhop); Te_ts_mask[kpnum['Te']['TS']:] = False
    Te_sp_mask = p_Te.X[:,0] > np.min(rhop); Te_sp_mask[:kpnum['Te']['TS']] = False
    #p_Te_rhop = aurora.rad_coord_transform(p_Te.X[mask], 'r/a', 'rhop', geqdsk)
    p_Te_rhop = eq.rho2rho('r/a', 'psinorm', p_Te.X[:,0], time, sqrt=True)
    ax2.errorbar(p_Te_rhop[Te_sp_mask], p_Te.y[Te_sp_mask]*1e3, p_Te.err_y[Te_sp_mask]*1e3, color='firebrick', fmt='.')
    ax2.errorbar(p_Te_rhop[Te_ts_mask], p_Te.y[Te_ts_mask]*1e3, p_Te.err_y[Te_ts_mask]*1e3, color='salmon', fmt='.')

    #ax[2].plot(rhop,np.maximum(Te_prof,Te_min))
    #ax[2].axhline(Te_min, c='r', ls='--')
    #ax[2].fill_between(rhop, Te_prof+Te_prof_unc,
    #                   np.maximum(Te_prof-Te_prof_unc,Te_min), alpha=0.3)

    ax[2].plot(rhop, np.log10(nn_prof), c='k')
    ax[2].errorbar(rhop_raw[ne_sp_mask], np.log10(nn_raw[ne_sp_mask]), ff*nn_raw_unc[ne_sp_mask]/nn_raw[ne_sp_mask], color='darkorange', fmt='.')
    ax[2].errorbar(rhop_raw[ne_ts_mask], np.log10(nn_raw[ne_ts_mask]), ff*nn_raw_unc[ne_ts_mask]/nn_raw[ne_ts_mask], color='wheat', fmt='.')
    #ax[2].plot(rhop_raw, np.log10(nn_raw), 'o')
    for ij in np.arange(num):
        ax[2].fill_between(rhop, np.log10(nn_prof)+3*ff*nn_prof_unc/nn_prof*ij/num,
                           np.log10(nn_prof)-3*ff*nn_prof_unc/nn_prof*ij/num, alpha=0.3*(1.-ij/num), color='k')

    #ax[3].plot(rhop, np.log10(nn_by_ne_prof),c='k')
    #for ij in np.arange(num):
    #    ax[3].fill_between(rhop,np.log10(nn_by_ne_prof)+3*ff*nn_by_ne_prof_unc/nn_by_ne_prof*ij/num,
    #                       np.log10(nn_by_ne_prof)-3*ff*nn_by_ne_prof_unc/nn_by_ne_prof*ij/num, alpha=0.3*(1.-ij/num),
    #                       color='k')
    
    ax[3].plot(rhop, np.log10(ion_rate_prof), c='k')
    ax[3].errorbar(rhop_raw[ne_sp_mask], np.log10(ion_rate_raw[ne_sp_mask]), ff*ion_rate_raw_unc[ne_sp_mask]/ion_rate_raw[ne_sp_mask], color='darkorange', fmt='.')
    ax[3].errorbar(rhop_raw[ne_ts_mask], np.log10(ion_rate_raw[ne_ts_mask]), ff*ion_rate_raw_unc[ne_ts_mask]/ion_rate_raw[ne_ts_mask], color='wheat', fmt='.')
    #ax[3].plot(rhop_raw, np.log10(ion_rate_raw), 'o')
    for ij in np.arange(num):
        ax[3].fill_between(rhop, np.log10(ion_rate_prof)+3*ff*ion_rate_prof_unc/ion_rate_prof*ij/num,
                           np.log10(ion_rate_prof)-3*ff*ion_rate_prof_unc/ion_rate_prof*ij/num, alpha=0.3*(1.-ij/num), color='k')

    ax[0].set_ylabel('Ly-a emiss [$W/cm^3$]')
    ax[1].set_ylabel(r'$n_e$ [$cm^{-3}$]', color='b')
    ax[1].tick_params(axis='y', colors='b')
    ax2.set_ylabel(r'$T_e$ [$eV$]', color='r')
    ax[2].set_ylabel(r'$log_{10}(n_n$ [$cm^{-3}$])')
    #ax[2].set_ylim([5,15])
    #ax[3].set_ylabel(r'$log_{10}(n_n/n_e)$')
    ax[3].set_ylabel(r'$log_{10}(S_{ion}$ [$cm^{-3}s^{-1}$])')
    #ax[3].set_ylim([-8,1])
    
    ax[-1].set_xlabel(r'$\rho_p$')
    ax[0].set_xlim([np.min(rhop[~np.isnan(emiss_prof)]),np.max(rhop[~np.isnan(emiss_prof)])])
    fig.suptitle(f'C-Mod shot {shot}')
    fig.tight_layout()

    out2 = [rhop,roa,R, nn_prof,nn_prof_unc,ion_rate_prof,ion_rate_prof_unc,ne_prof,ne_prof_unc,Te_prof,Te_prof_unc]
    with open(f'Dicts/lyman_data_{shot}.pkl','wb') as f:
        pkl.dump(out2,f)

    
