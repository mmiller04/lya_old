import numpy as np
import matplotlib.pyplot as plt
import lyman_data
import lyman_single

def scan_zero(R1,R2,plot=False):

    R_end_vec = np.linspace(R1,R2,5)
    out_list = []

    for each_R in R_end_vec:

        # gather Lyman-alpha profiles
        _out = lyman_single.single_case(shot,tmin,tmax,roa_kp,ne,ne_std,Te,Te_std,
                p_ne,p_Te, gfiles_loc=gfiles_loc,
                tomo_inversion=True, zero_pos=each_R,
                SOL_exp_decay=False)

        out_list.append(_out)

    
    if plot:

        fig, ax = plt.subplots(3, sharex=True)

        for i in range(len(out_list)):

            R = out_list[i][0]
            emiss = out_list[i][11]
            emiss_unc = out_list[i][12]
            nn = out_list[i][7]
            nn_unc = out_list[i][8]

            #ax[0].errorbar(R,emiss,emiss_unc,fmt='.')
            ax[0].plot(R,emiss)
            #ax[2].errorbar(R,nn,nn_unc,fmt='.')
            ax[2].plot(R,nn)

        # these are the same for scan
        ne_prof = out_list[0][3]
        ne_unc = out_list[0][4]
        #ax[1].errorbar(R,ne_prof,ne_unc,fmt='.')
        ax[1].plot(R,ne_prof)

        Te_prof = out_list[0][5]
        Te_unc = out_list[0][6]
        ax2 = ax[1].twinx()
        #ax2.errorbar(R,Te_prof,Te_unc,fmt='.')
        ax2.plot(R,Te_prof)

        ax[0].legend(['0.92 m','0.9225 m','0.925 m','0.9275 m','0.93 m'])
        
    return out_list


def scan_error(plot):

    sys_err_vec = np.array([5,7.5,10,20,50,100])
    out_list = []

    for each_err in sys_err_vec:

        # gather Lyman-alpha profiles
        _out = lyman_single.single_case(shot,tmin,tmax,roa_kp,ne,ne_std,Te,Te_std,
                p_ne,p_Te, gfiles_loc=gfiles_loc,
                tomo_inversion=True, tomo_err=each_err,
                SOL_exp_decay=False)

        out_list.append(_out)

    
    if plot:

        fig, ax = plt.subplots(3, sharex=True)

        for i in range(len(out_list)):

            R = out_list[i][0]
            emiss = out_list[i][11]
            emiss_unc = out_list[i][12]
            nn = out_list[i][7]
            nn_unc = out_list[i][8]

            #ax[0].errorbar(R,emiss,emiss_unc,fmt='.')
            ax[0].plot(R,emiss)
            #ax[2].errorbar(R,nn,nn_unc,fmt='.')
            ax[2].plot(R,nn)

        # these are the same for scan
        ne_prof = out_list[0][3]
        ne_unc = out_list[0][4]
        #ax[1].errorbar(R,ne_prof,ne_unc,fmt='.')
        ax[1].plot(R,ne_prof)

        Te_prof = out_list[0][5]
        Te_unc = out_list[0][6]
        ax2 = ax[1].twinx()
        #ax2.errorbar(R,Te_prof,Te_unc,fmt='.')
        ax2.plot(R,Te_prof)

        ax[0].legend(['5%','7.5%','10%','20%','50%','100%'])
    
    return out_list



def scan_shift(shift1,shift2,plot=False):

    shift_vec = np.linspace(shift1,shift2,3)
    out_list = []

    for each_shift in shift_vec:

        # gather Lyman-alpha profiles
        _out = lyman_single.single_case(shot,tmin,tmax,roa_kp,
                ne,ne_std,Te,Te_std, p_ne,p_Te, 
                lya_shift=each_shift, gfiles_loc=gfiles_loc,
                tomo_inversion=True,
                SOL_exp_decay=False)

        out_list.append(_out)


    if plot:

        fig, ax = plt.subplots(3, sharex=True)

        for i in range(len(out_list)):

            R = out_list[i][0]
            emiss = out_list[i][11]
            emiss_unc = out_list[i][12]
            nn = out_list[i][7]
            nn_unc = out_list[i][8]

            #ax[0].errorbar(R,emiss,emiss_unc,fmt='.')
            ax[0].plot(R,emiss)
            ax[2].plot(R,nn)

       
        # use quantities without shift
        R = out_list[1][0]
        emiss = out_list[1][11]
        nn = out_list[1][7]
        
        efit_unc = calc_efit_unc(R,emiss,lcfs_err=500)*nn
        ax[2].errorbar(R,nn,efit_unc,fmt='.')

        # these are the same for scan
        ne_prof = out_list[0][3]
        ne_unc = out_list[0][4]
        #ax[1].errorbar(R,ne_prof,ne_unc,fmt='.')
        ax[1].plot(R,ne_prof,c='b')

        Te_prof = out_list[0][5]
        Te_unc = out_list[0][6]
        ax2 = ax[1].twinx()
        #ax2.errorbar(R,Te_prof,Te_unc,fmt='.')
        ax2.plot(R,Te_prof,c='r')

        ax[0].legend(['-2.5 mm','0 mm','2.5 mm'])
        
        ax[0].set_ylabel('Ly-a emiss [$W/cm^{3}$]')
        ax[1].set_ylabel('$n_{e}$ [$cm^{-3}$]',c='b')
        ax2.set_ylabel('$T_{e}$ [$eV$]',c='r')
        ax[2].set_ylabel('$n_{n}$ [$cm^{-3}$]')
    
    return out_list


def calc_efit_unc(R, emiss, lcfs_err=5):

    demiss_dr = np.zeros_like(emiss)
    demiss_dr[:-1] = np.diff(emiss)/np.diff(R) # array length one less
    demiss_dr[-1] = demiss_dr[-2]

    prop_err = (lcfs_err/2/1000)*demiss_dr

    return prop_err


if __name__ == '__main__':

    shot = 1070511002
    tmin = 0.7
    tmax = 1.4

    gfiles_loc = '.'
    kp_out = lyman_data.get_cmod_kin_profs(shot,tmin,tmax,gfiles_loc=gfiles_loc)
    roa_kp, ne, ne_std, Te, Te_std, p_ne, p_Te = kp_out

    geqdsk = lyman_data.get_geqdsk_cmod(shot,(tmin+tmax)/2.*1e3,gfiles_loc=gfiles_loc)

#    out_zero = scan_zero(0.92,0.93,plot=True)
#    out_err = scan_error(plot=True)
    out_shift = scan_shift(-0.0025,0.0025,plot=True)

