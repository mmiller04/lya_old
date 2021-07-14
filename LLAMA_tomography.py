import matplotlib as mpl
#mpl.rcParams['keymap.back'].remove('left')
#mpl.rcParams['keymap.forward'].remove('right')

import numpy as np

from scipy.stats.mstats import mquantiles
import matplotlib.pylab as plt

from time import time
from matplotlib.widgets import Slider, MultiCursor,Button

from IPython import embed
import MDSplus as mds
from multiprocessing import Pool
import argparse

from  scipy.linalg import eigh, solve_banded
from  scipy.interpolate import interp1d

from omfit_classes.omfit_mds import OMFITmdsValue

chanVar= np.asarray([ 0.00237986,  0.00450433,  0.0115627 ,  0.00488375,  0.00378624,\
0.00514737,  0.00504275,  0.00584001,  0.00533578,  0.00558195,\
0.00513933,  0.00541877,  0.0048475 ,  0.00506704,  0.00493994,\
0.00519987,  0.00466806,  0.00280573,  0.00380159,  0.00337836,\
0.00398431,  0.00263873,  0.00356935,  0.00296274,  0.00335741,\
0.00405468,  0.00339026,  0.00407109,  0.00302094,  0.00282334,\
0.00284611,  0.16003643,  0.00338876,  0.00375883,  0.00350299,\
0.00383726,  0.00287146,  0.00351086,  0.00329034,  0.00357102])

        

#Plotting
def update_fill_between(fill,x,y_low,y_up,min,max ):
    paths, = fill.get_paths()
    nx = len(x)
    
    y_low = np.maximum(y_low, min)
    y_low[y_low==max] = min
    y_up = np.minimum(y_up,max)
    
    vertices = paths.vertices.T
    vertices[:,1:nx+1] = x,y_up
    vertices[:,nx+1] =  x[-1],y_up[-1]
    vertices[:,nx+2:-1] = x[::-1],y_low[::-1]
    vertices[:, 0] = x[0],y_up[0]
    vertices[:,-1] = x[0],y_up[0]

#Plotting    
def update_errorbar(err_plot, x,y,yerr):
    
    plotline, caplines, barlinecols = err_plot

    # Replot the data first
    plotline.set_data(x,y)

    # Find the ending points of the errorbars
    error_positions = (x,y-yerr), (x,y+yerr)

    # Update the caplines
    if len(caplines) > 0:
        for j,pos in enumerate(error_positions):
            caplines[j].set_data(pos)

    # Update the error bars
    barlinecols[0].set_segments(list(zip(list(zip(x,y-yerr)), list(zip(x,y+yerr))))) 
        

#Loading individual channels
def mds_load(tmp):
    (mds_server,   TDI) = tmp
    MDSconn = mds.Connection(mds_server )
    output = [MDSconn.get(tdi).data() for tdi in TDI]

    return output

#This allows parallel loading of each LLAMA channel
#on DIII-D each channel is stored independentl
def mds_par_load(mds_server,   TDI,  numTasks):

    #load a junks of a single vector
    TDI = np.array_split(TDI, min(numTasks, len(TDI)))

    args = [(mds_server,  tdi) for tdi in TDI]

    pool = Pool(len(args))
    

    out = pool.map(mds_load,args)
    pool.close()
    pool.join()
    
    #join lists
    output = [j for i in out for j in i]
 
    return  output
    

    


class LLAMA_tomography():
    inner_wall_R = 1.013 #m
    outer_wall_R = 2.371 #m

    def __init__(self, shot, system, regularisation='GCV', time_avg= 0.0005):
        self.shot = shot
        self.regularisation = regularisation
        self.time_avg = time_avg
        self.system = system # which Ly-a array brightness data is taken from

    #Loads gemoetry excluding channels in LFScList
    #basically the same as load_geometry if LFScList is empty
    def load_geometry_removeChan(self,LFScList):
        #LFScList is the index of the channels to skip
                
        coord_path = '/fusion/projects/diagnostics/llama/coordinates/'
        #coord_path = ''
        
        
        import h5py as h5
        data=h5.File(coord_path+'Jan2020Coords.h5','r')
        
        
        LFS_coords = np.array( data['Data_LFS']['rProfCoords']) 
        LFS_weights = np.array( data['Data_LFS']['rProf']) 

        HFS_coords = np.array( data['Data_HFS']['rProfCoords']) 
        HFS_weights = np.array( data['Data_HFS']['rProf']) 

        #tangential rardius
        lfs_r = np.hypot(LFS_coords[:,0], LFS_coords[:,1])/1e3 #m
        hfs_r = np.hypot(HFS_coords[:,0], HFS_coords[:,1])/1e3 #m
        lfs_z =  LFS_coords[:,2].mean(1)/1e3 #m
        hfs_z =  HFS_coords[:,2].mean(1)/1e3 #m

        #We now deleta the specified indexes...only for LFS
        LFS_coords = np.delete(LFS_coords,LFScList,axis=0) #I dont think this is necessary since it isn't used again

        lfs_r = np.delete(lfs_r,LFScList,axis=0)
        lfs_z = np.delete(lfs_z,LFScList,axis=0)

   
        self.nch_hfs = len(hfs_r)
        self.nch_lfs = len(lfs_r)
                
        ## calculation 2019_11_01
        #HFScalf = 1./np.array([7.80179e-23,7.77613e-23,7.74548e-23,7.70985e-23,7.66927e-23,7.62375e-23,7.57334e-23,7.5181e-23,7.45806e-23,7.39331e-23,\
                            #7.30678e-23,7.19921e-23,7.08806e-23,6.97349e-23,6.85565e-23,6.7347e-23,6.61082e-23,6.48416e-23,6.35492e-23,6.22325e-23])
        #LFScalf = 1./np.array([2.67382e-22,2.71539e-22,2.75228e-22,2.78458e-22,2.81242e-22,2.83593e-22,2.85523e-22,2.87049e-22,2.88183e-22,2.88941e-22,\
                            #2.88663e-22,2.87375e-22,2.85775e-22,2.83885e-22,2.81726e-22,2.79319e-22,2.76683e-22,2.7384e-22,2.70808e-22,2.67608e-22])
        
        # revised 2020_05_05 #laggnerf
        HFScalf = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/HFS_calibration.dat')[:,1]
        LFScalf = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/LFS_calibration.dat')[:,1]
        
        HFScalfErr = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/HFS_calibration.dat')[:,2]
        LFScalfErr = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/LFS_calibration.dat')[:,2]
        
        LFScalf = np.delete(LFScalf,LFScList,axis=0)
        LFScalfErr = np.delete(LFScalfErr,LFScList,axis=0)
        self.calf = np.hstack((HFScalf,LFScalf))
        self.calfErr = np.hstack((HFScalfErr,LFScalfErr))

        ##create response matrix

        ##account for  finite LOS width 
        R_tg_virtual = np.vstack((hfs_r, lfs_r)).T
        weight = np.vstack((HFS_weights, LFS_weights)).T
        weight /= np.sum(weight,0)

        #We have to delete some of the LFS channels here due to the vstack in weight must
        #be the same length
        weight = np.delete(weight,LFScList+self.nch_hfs,axis =1)
        LFS_weights = np.delete(LFS_coords,LFScList,axis=0)

        #center of mass of the LOS
        self.R_tg = np.average(R_tg_virtual,0,weight) 
        self.Z_tg = np.hstack((hfs_z,lfs_z))

        self.lfs_min = self.R_tg[self.nch_hfs:].min()
        self.lfs_max = self.R_tg[self.nch_hfs:].max()
        self.hfs_min = self.R_tg[:self.nch_hfs].min()
        self.hfs_max = self.R_tg[:self.nch_hfs].max()
        
        self.nr = 200
        self.R_grid = np.hstack((np.linspace(self.hfs_min-.01,self.hfs_max+.05,self.nr),
                                 np.linspace(self.lfs_min-.01,self.lfs_max+.02,self.nr)))
        
        dL = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-R_tg_virtual[:,:,None]**2,0))       
               -np.sqrt(np.maximum( self.R_grid[:-1]**2-R_tg_virtual[:,:,None]**2,0)))
        self.dL = np.sum(dL*weight[:,:,None],0)
    

        #add more LOS in between existing for a better plotting of the results
        clipped_grid = np.clip(self.R_grid,self.R_tg.min(),self.R_tg.max())
        weights_grid = interp1d(self.R_tg,weight)(clipped_grid) 
        R_virtual_grid = interp1d(self.R_tg,R_tg_virtual,fill_value='extrapolate')(self.R_grid)

        dL_back = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-R_virtual_grid[:,:,None]**2,0))       
                    -np.sqrt(np.maximum( self.R_grid[:-1]**2-R_virtual_grid[:,:,None]**2,0)))
        self.dL_back = np.sum(dL_back*weights_grid[:,:,None],0)  
        
        
        
    def load_geometry(self,r_end=False,sys_err=5):
        
        node = OMFITmdsValue(server='CMOD',shot=self.shot,treename='SPECTROSCOPY',
            TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
            '{:s}:BRIGHT'.format(self.system))

        # check which channels are empty

        bright = node.data()
        self.good_chans = np.where(bright[0] != 0)[0]
        
        # LFS_coords = np.array( data['Data_LFS']['rProfCoords']) 
        # LFS_weights = np.array( data['Data_LFS']['rProf'])

        # HFS_coords = np.array( data['Data_HFS']['rProfCoords']) 
        # HFS_weights = np.array( data['Data_HFS']['rProf']) 

        #tangential rardius
        # lfs_r = np.hypot(LFS_coords[:,0], LFS_coords[:,1])/1e3 #m
        # hfs_r = np.hypot(HFS_coords[:,0], HFS_coords[:,1])/1e3 #m
        # lfs_z =  LFS_coords[:,2].mean(1)/1e3 #m
        # hfs_z =  HFS_coords[:,2].mean(1)/1e3 #m

        lfs_r = node.dim_of(0)[self.good_chans]
        lfs_r = np.flip(lfs_r) # stored as decreasing
        
        if r_end:
            lfs_r = np.insert(lfs_r,len(lfs_r),r_end) # want to insert a 0 at r_end

        LFS_weights = np.ones(len(lfs_r))

        # lfs_z = z_midplane - 0.125

        # self.nch_hfs = len(hfs_r)
        self.nch_lfs = len(lfs_r)
                
        ## calculation 2019_11_01
        #HFScalf = 1./np.array([7.80179e-23,7.77613e-23,7.74548e-23,7.70985e-23,7.66927e-23,7.62375e-23,7.57334e-23,7.5181e-23,7.45806e-23,7.39331e-23,\
                            #7.30678e-23,7.19921e-23,7.08806e-23,6.97349e-23,6.85565e-23,6.7347e-23,6.61082e-23,6.48416e-23,6.35492e-23,6.22325e-23])
        #LFScalf = 1./np.array([2.67382e-22,2.71539e-22,2.75228e-22,2.78458e-22,2.81242e-22,2.83593e-22,2.85523e-22,2.87049e-22,2.88183e-22,2.88941e-22,\
                            #2.88663e-22,2.87375e-22,2.85775e-22,2.83885e-22,2.81726e-22,2.79319e-22,2.76683e-22,2.7384e-22,2.70808e-22,2.67608e-22])
        
        # # revised 2020_05_05 #laggnerf
        # HFScalf = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/HFS_calibration.dat')[:,1]
        # LFScalf = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/LFS_calibration.dat')[:,1]
        
        # HFScalfErr = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/HFS_calibration.dat')[:,2]
        # LFScalfErr = np.loadtxt('/fusion/projects/diagnostics/llama/Calibration/calFactors/2020_01/LFS_calibration.dat')[:,2]
        

        ### ignore claibration for now

        self.calf = np.ones(lfs_r.shape)
        self.calfErr = np.ones(lfs_r.shape)*sys_err/100

        # self.calf = np.hstack((HFScalf,LFScalf))
        # self.calfErr = np.hstack((HFScalfErr,LFScalfErr))

        ##create response matrix

        ##account for  finite LOS width 
        # R_tg_virtual = np.vstack((hfs_r, lfs_r)).T
        R_tg_virtual = lfs_r
        # weight = np.vstack((HFS_weights, LFS_weights)).T
        #weight = LFS_weights
        #weight /= np.sum(weight,0)

        #center of mass of the LOS
        self.R_tg = R_tg_virtual
        #self.R_tg = np.average(R_tg_virtual,0,weight) 
        self.Z_tg = np.zeros_like(self.R_tg) # assume at midplane (z = 0)

        self.lfs_min = self.R_tg[0]
        self.lfs_max = self.R_tg[-1]

        # self.lfs_min = self.R_tg[self.nch_hfs:].min() # first r value
        # self.lfs_max = self.R_tg[self.nch_hfs:].max() # last r value
        # self.hfs_min = self.R_tg[:self.nch_hfs].min()
        # self.hfs_max = self.R_tg[:self.nch_hfs].max()
       
        self.nr = 50
        #self.R_grid = np.linspace(self.lfs_min-.01,self.lfs_max+.01,self.nr)
        self.R_grid = np.linspace(self.lfs_min,self.lfs_max,self.nr)
        # self.R_grid = np.hstack((np.linspace(self.hfs_min-.01,self.hfs_max+.05,self.nr),
        #                          np.linspace(self.lfs_min-.01,self.lfs_max+.02,self.nr)))
        
        dL = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-R_tg_virtual[:,None]**2,0))       
               -np.sqrt(np.maximum( self.R_grid[:-1]**2-R_tg_virtual[:,None]**2,0)))
        self.dL = dL # no need to sum over spot size
    
        # #add more LOS in between existing for a better plotting of the results
        # clipped_grid = np.clip(self.R_grid,self.eR_tg.min(),self.R_tg.max())
        # weights_grid = interp1d(self.R_tg,weight)(clipped_grid) 
        # R_virtual_grid = interp1d(self.R_tg,R_tg_virtual,fill_value='extrapolate')(self.R_grid)

        # dL_back = 2*(np.sqrt(np.maximum((self.R_grid[1:])**2-R_virtual_grid[:,:,None]**2,0))       
        #             -np.sqrt(np.maximum( self.R_grid[:-1]**2-R_virtual_grid[:,:,None]**2,0)))
        # self.dL_back = np.sum(dL_back*weights_grid[:,:,None],0)

    #Simplest data load
    #smooths data for an entire shot
    def load_data(self,r_end=False):

        # #mds_server = 'localhost'

        # MDSconn = mds.Connection(mds_server)

        # PTNAME = 'PTDATA2("LYA1%s%.2dRAW",%d,1)'
        
        # TDI  = [PTNAME%('H',n+1,self.shot) for n in range(n_los)]
        # TDI += [PTNAME%('L',n+1,self.shot) for n in range(n_los)]
        # TDI += ['dim_of(%s)'%TDI[-1]]
        
        # raw = mds_par_load(mds_server,  TDI, 8)
        # tvec = raw.pop(-1)

        node = OMFITmdsValue(server='CMOD',shot=self.shot,treename='SPECTROSCOPY',
            TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
            '{:s}:BRIGHT'.format(self.system))

        # raw_data = np.vstack(raw).T
        raw_data = node.data()
        raw_data = raw_data[:,self.good_chans]
        raw_data = np.flip(raw_data)

        # add a zero at desired r_end value (set in load_geometry)
        if r_end:
            _zeros = np.zeros(len(raw_data[:,0]))[:,None]
            raw_data = np.concatenate((raw_data,_zeros),axis=1)

        n_los = len(raw_data[0])

        tvec = node.dim_of(1)

        offset = slice(0,tvec.searchsorted(0))
        dt = (tvec[-1]-tvec[0])/(len(tvec)-1)

        #n_smooth = int(20*self.time_avg/dt)
        n_smooth = 1

        nt,nch = raw_data.shape

        data_low = raw_data
        tvec_low = tvec

        nt = nt//n_smooth*n_smooth
        
        tvec_low = tvec[:nt].reshape(-1,n_smooth).mean(1)

        data_low = raw_data[:nt].reshape(-1,n_smooth, nch).mean(1)-raw_data[offset].mean(0)

        #data = np.load('data.npz')
        #data_low = data['data_low']
        #tvec_low = data['tvec_low']

        #estimate noise from the signal before the plasma
        error_low1 = np.zeros_like(data_low)
        error_low2 = np.std(data_low[tvec_low<0],0)
        error_low21 = np.std(data_low[tvec_low<0],0)[None,:]
        error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]/3


        #guess errorbarss from the variation between neighboring channels
        #ind1 = np.r_[1,0:n_los-1,n_los+1,  n_los:n_los*2-1]
        #ind2 = np.r_[  1:n_los  ,n_los-2,n_los+1:n_los*2  ,n_los*2-2]
        ind1 = np.r_[1,0:n_los-1]
        ind2 = np.r_[  1:n_los  ,n_los-2]

        #the ind1 and ind2, basically shift data_low up and down by one to find the average
        #value of the neighboring channels. That is then subtracting from the original array
        # to get the average difference from neighboring hcannels
        
        #the difference between the followin neighbor is calculated and then the standard error
        #is calculated for each channel
        error_low += np.std(np.diff(data_low-(data_low[:,ind1]+data_low[:,ind2])/2,axis=0),axis=0)/np.sqrt(2)

        #remove offset estimated from the edge most detector 
        offset_time = data_low[:,[-1]]
        data_low -= offset_time
        #make sure that zero value is within errorbars when data are negative
        error_low = np.maximum(error_low, -data_low)
        
        self.data = data_low *self.calf#[ph/m^2s]
        #self.err  = error_low*self.calf#[ph/m^2s] # laggnerf
        self.err  = np.sqrt(\
                    (error_low*self.calf)**2+\
                    (data_low*self.calfErr)**2
                   ) #[ph/m^2s] # laggnerf
        self.tvec = tvec_low #[s]
        self.scale = np.median(self.data) #just a normalisation to aviod calculation with so huge exponents
        
        #BUG corrupted channel
        #self.err[ :,20+11] *= 10
        #self.data[:,20+11] *= 0.8
        self.nt = len(self.tvec)
        #print(self.nt)

    """breaks data into time from ELMS
    """
    def load_data_ELMt(self,elmT,LFScList):
        
        #fast fetch of the experimental data
        mds_server = 'atlas.gat.com'
        #mds_server = 'localhost'

        MDSconn = mds.Connection(mds_server )

        PTNAME = 'PTDATA2("LYA1%s%.2dRAW",%d,1)'
        n_los = 20 
        
        TDI  = [PTNAME%('H',n+1,self.shot) for n in range(n_los)]
        TDI += [PTNAME%('L',n+1,self.shot) for n in range(n_los)]
        TDI += ['dim_of(%s)'%TDI[-1]]
        
        raw = mds_par_load(mds_server,  TDI, 8)
        tvec = raw.pop(-1)


        
        raw_data = np.vstack(raw).T

        offset = slice(0,tvec.searchsorted(0))
       
        dt = (tvec[-1]-tvec[0])/(len(tvec)-1)

        n_smooth = int(1000*self.time_avg/dt)


        nt,nch = raw_data.shape


        

        
        nt = nt//n_smooth*n_smooth

        data_low = raw_data[:nt].reshape(-1,n_smooth, nch).mean(1)-raw_data[offset].mean(0)
        tvec_low = tvec[:nt].reshape(-1,n_smooth).mean(1)
        
        #estimate noise from the signal before the plasma 
        #error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]/np.sqrt(data_low[tvec_low<0].shape[0])
        #error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]
        error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]/np.sqrt(3)
        """
        f,a = plt.subplots(2,20)

        for i in range(40):
            if i<20:

                a[0,i%20].hist(data_low[tvec_low<0][:,i],bins = 20)
            else:
                a[1,i%20].hist(data_low[tvec_low<0][:,i],bins = 20)
        plt.show()
        """

        error_low = error_low[0,:]


        #guess errorbarss from the variation between neighboring channels

        ind1 = np.r_[1,0:n_los-1,n_los+1,  n_los:n_los*2-1]
        ind2 = np.r_[  1:n_los  ,n_los-2,n_los+1:n_los*2  ,n_los*2-2]

        edges = np.zeros(len(elmT),dtype =int)

        for i in range(len(elmT)):
            cT = elmT[i]

            iT = np.argmin(np.abs(tvec-cT))#start index


            print('desired time: '+str(cT))
            print('actual :'+str(tvec[iT]))


            edges[i] = iT

        
        # we now grab the windows specified in edges by splitting the aray
        # at the indicies specified in edges. every other part of the array
        # will be one of our windows



        #we make a new tvec from 0 to max t from elm 
        maxT = np.max(np.diff(tvec[edges]))
        maxI = np.max(np.diff(edges))

        data_elmT = np.empty(shape = (len(elmT)-1,maxI,nch))
        data_elmT.fill(np.nan)

        #split does not return a numpy array since the sections are of different length
        raw_split = np.split(raw_data,edges)[1:len(elmT)]


        for i in range(len(raw_split)):
            cDat = np.asarray(raw_split[i])

            #since cDat varies in shape we set the zeros which match
            #the shape of cDat equal to cDat
            #effectively cDat is padded with zeros so that each of raw_split entries
            #can be combined into one numpy array
            data_elmT[i,:cDat.shape[0],:cDat.shape[1]] = cDat

        #we now want to take the mean of all points in the timte_avg window we can first collapse
        #take the mean along axis 0 

        data_low = np.zeros(shape = (maxI//n_smooth,40))
        error_low_elmT = np.zeros(shape = (maxI//n_smooth,40))


        #the speed of this could likely be improved using reshape but I can't figure
        #out how to improve it
        for i in range(len(data_low)):

            subMat = data_elmT[:,n_smooth*i:n_smooth*i+n_smooth,:]
            mSubMat = np.ma.masked_array(subMat,np.isnan(subMat))

            #we take the mean ignoring the nan entries added by paddig

            #this would work if we had an up to date version of numpy
            #data_low[i,:] = subMat.nanmean(axis = (0,1))

            #error_low_elmT = subMat.std(axis = (0,1))

            #instead we have to mask
            data_low[i,:] = mSubMat.mean(axis = (0,1)).filled(np.nan)

            #coutn the number of non Nan etnries
            nPts = (~np.isnan(subMat)).sum(axis=(0,1))
            error_low_elmT[i,:] = mSubMat.std(axis = (0,1)).filled(np.nan)/np.sqrt(nPts)



        tvec_elm = np.arange((dt*n_smooth)/2,maxT,(dt*n_smooth))

        #removing the offset from t<0
        data_low = data_low-raw_data[offset].mean(0)


       
        #if you use the outside channel you can get negatives which can cause issues
        #with the inversion near the peak
        #outerChan = np.min(data_low)

        #remove offset estimated from the edge most detector 
        #offset_time = data_low[:,[-1]]
        offset_time = np.asarray([data_low.min(axis = 1)]).T


        self.rawData = np.delete(data_low,20+LFScList,axis=1)*self.calf
        data_low -= offset_time


        #make sure that zero value is within errorbars when data are negative
        errorRes = np.maximum(np.sqrt(error_low**2+error_low_elmT**2+chanVar**2), -data_low)
        errorRes = np.delete(errorRes,20+LFScList,axis=1)

        data_low = np.delete(data_low,20+LFScList,axis=1)


        self.data = data_low*self.calf#[ph/m^2s]


        #self.err  = error_low*self.calf#[ph/m^2s] # laggnerf

        self.err  = np.sqrt(\
                    (errorRes*self.calf)**2+\
                    (data_low *self.calfErr)**2
                   ) #[ph/m^2s] # laggnerf
        self.tvec = tvec_elm/1e3 #[s]
        self.scale = np.median(self.data) #just a normalisation to aviod calculation with so huge exponents
        
        
        #BUG corrupted channel
        #self.err[:,20+11] *= 10
        #self.data[:,20+11] *= 0.8
        #self.rawData[:,20+11] *= 0.8
        self.nt = len(self.tvec)





    """breaks data into windows specified by tWindows
    inversion will be perfomed on each window 
    """
    def load_data_specWindow(self,tWindows):
        #fast fetch of the experimental data
        mds_server = 'atlas.gat.com'
        #mds_server = 'localhost'

        MDSconn = mds.Connection(mds_server )

        PTNAME = 'PTDATA2("LYA1%s%.2dRAW",%d,1)'
        n_los = 20 
        
        TDI  = [PTNAME%('H',n+1,self.shot) for n in range(n_los)]
        TDI += [PTNAME%('L',n+1,self.shot) for n in range(n_los)]
        TDI += ['dim_of(%s)'%TDI[-1]]
        
        raw = mds_par_load(mds_server,  TDI, 8)
        tvec = raw.pop(-1)


        
        raw_data = np.vstack(raw).T

        offset = slice(0,tvec.searchsorted(0))
       
        dt = (tvec[-1]-tvec[0])/(len(tvec)-1)

        n_smooth = int(1000*self.time_avg/dt)


        nt,nch = raw_data.shape
        
        nt = nt//n_smooth*n_smooth

        data_low = raw_data[:nt].reshape(-1,n_smooth, nch).mean(1)-raw_data[offset].mean(0)
        tvec_low = tvec[:nt].reshape(-1,n_smooth).mean(1)
        
        #estimate noise from the signal before the plasma 
        error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]/3

        error_low = error_low[0,:]



        #guess errorbarss from the variation between neighboring channels

        ind1 = np.r_[1,0:n_los-1,n_los+1,  n_los:n_los*2-1]
        ind2 = np.r_[  1:n_los  ,n_los-2,n_los+1:n_los*2  ,n_los*2-2]

        nWindows = tWindows.shape[-1]


        edges = np.zeros(nWindows*2)


        for i in range(nWindows):
            tW = tWindows[:2,i]

            iS = np.argmin(np.abs(tvec-tW[0]))#start index
            iE = np.argmin(np.abs(tvec-tW[1]))#end index
            """
            print('desired time start: '+str(tW[0]))
            print('actual :'+str(tvec[iS]))

            print('desired time end: '+str(tW[1]))
            print('actual :'+str(tvec[iE]))
            """
            edges[2*i] = iS
            edges[2*i+1] = iE

        
        # we now grab the windows specified in edges by splitting the aray
        # at the indicies specified in edges. every other part of the array
        # will be one of our windows

        #added Jan 7 2021, edges were becoming a float for some reason
        edges=edges.astype(int)

        tvec_window = np.split(tvec,edges)[1::2]


        dWindow = np.split(raw_data,edges,axis = 0)[1::2]


        tvec_window = np.mean(tvec_window,axis = 1)
        data_low = np.mean(dWindow,axis = 1)

        error_low = np.tile(error_low,(len(data_low),1))

        error_low += np.std(np.diff(data_low-(data_low[:,ind1]+data_low[:,ind2])/2,axis=0),axis=0)/np.sqrt(2)

        #remove offset estimated from the edge most detector 
        offset_time = data_low[:,[-1]]
        data_low -= offset_time
        #make sure that zero value is within errorbars when data are negative
        error_low = np.maximum(error_low, -data_low)

        self.data = data_low *self.calf#[ph/m^2s]


        #self.err  = error_low*self.calf#[ph/m^2s] # laggnerf
        self.err  = np.sqrt(\
                    (error_low*self.calf)**2+\
                    (data_low*self.calfErr)**2
                   ) #[ph/m^2s] # laggnerf
        self.tvec = tvec_window/1e3 #[s]
        self.scale = np.median(self.data) #just a normalisation to aviod calculation with so huge exponents
        
        #BUG corrupted channel
        self.err[ :,20+11] *= 10
        self.data[:,20+11] *= 0.8
        self.nt = len(self.tvec)


    def load_data_Window(self,tWindows):
        
        #fast fetch of the experimental data
        mds_server = 'atlas.gat.com'

        #mds_server = 'localhost'

        MDSconn = mds.Connection(mds_server)

        PTNAME = 'PTDATA2("LYA1%s%.2dRAW",%d,1)'
        n_los = 20 
        
        TDI  = [PTNAME%('H',n+1,self.shot) for n in range(n_los)]
        TDI += [PTNAME%('L',n+1,self.shot) for n in range(n_los)]
        TDI += ['dim_of(%s)'%TDI[-1]]
        
        raw = mds_par_load(mds_server,  TDI, 8)
        tvec = raw.pop(-1)


        
        raw_data = np.vstack(raw).T

        offset = slice(0,tvec.searchsorted(0))
       
        dt = (tvec[-1]-tvec[0])/(len(tvec)-1)

        n_smooth = int(1000*self.time_avg/dt)


        nt,nch = raw_data.shape

        data_low = raw_data
        tvec_low = tvec

        
        nt = nt//n_smooth*n_smooth

        data_low = raw_data[:nt].reshape(-1,n_smooth, nch).mean(1)-raw_data[offset].mean(0)
        tvec_low = tvec[:nt].reshape(-1,n_smooth).mean(1)
        
        #estimate noise from the signal before the plasma 
        #error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]/np.sqrt(data_low[tvec_low<0].shape[0])
        #error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]
        error_low = np.zeros_like(data_low)+np.std(data_low[tvec_low<0],0)[None,:]/np.sqrt(3)
        """
        f,a = plt.subplots(2,20)

        for i in range(40):
            if i<20:

                a[0,i%20].hist(data_low[tvec_low<0][:,i],bins = 20)
            else:
                a[1,i%20].hist(data_low[tvec_low<0][:,i],bins = 20)
        plt.show()
        """

        error_low = error_low[0,:]


        #guess errorbarss from the variation between neighboring channels

        ind1 = np.r_[1,0:n_los-1,n_los+1,  n_los:n_los*2-1]
        ind2 = np.r_[  1:n_los  ,n_los-2,n_los+1:n_los*2  ,n_los*2-2]


        nWindows = tWindows.shape[-1]


        edges = np.zeros(nWindows*2)


        for i in range(nWindows):
            tW = tWindows[:2,i]

            iS = np.argmin(np.abs(tvec-tW[0]))#start index
            iE = np.argmin(np.abs(tvec-tW[1]))#end index

            print('desired time start: '+str(tW[0]))
            print('actual :'+str(tvec[iS]))

            print('desired time end: '+str(tW[1]))
            print('actual :'+str(tvec[iE]))

            edges[2*i] = iS
            edges[2*i+1] = iE

        
        # we now grab the windows specified in edges by splitting the aray
        # at the indicies specified in edges. every other part of the array
        # will be one of our windows


        tvec_window = np.concatenate(np.split(tvec,edges)[1::2])


        dWindow = np.concatenate(np.split(raw_data,edges,axis = 0)[1::2])


        data_low = dWindow.mean(0)

        #removing the offset from t<0
        data_low = data_low-raw_data[offset].mean(0)

        error_low_window = np.std(dWindow-raw_data[offset].mean(0),axis = 0)/np.sqrt(len(dWindow[:,0]))
        #error_low_window = np.std(dWindow-raw_data[offset].mean(0),axis = 0)


        outerChan = data_low[-1]
        #if you use the outside channel you can get negatives which can cause issues
        #with the inversion near the peak
        #outerChan = np.min(data_low)

        #remove offset estimated from the edge most detector 
        offset_time = data_low - outerChan


        data_low = offset_time



        #make sure that zero value is within errorbars when data are negative
        errorRes = np.maximum(np.sqrt(error_low**2+error_low_window**2+chanVar**2), -data_low)



        self.data = data_low *self.calf#[ph/m^2s]


        #self.err  = error_low*self.calf#[ph/m^2s] # laggnerf
        self.err  = np.sqrt(\
                    (errorRes*self.calf)**2+\
                    (data_low *self.calfErr)**2
                   ) #[ph/m^2s] # laggnerf
        self.tvec = np.mean(tvec_window)/1e3 #[s]
        self.scale = np.median(self.data) #just a normalisation to aviod calculation with so huge exponents
        
        
        #BUG corrupted channel
        
        self.err[20+11] *= 10
        self.data[20+11] *= 0.8
        self.nt = 1
        





        

    def regul_matrix(self, biased_edges = True):
        #regularization band matrix

        bias = .1 if biased_edges else 1e-5
        D = np.ones((3,self.nr-1))
        D[1,:] *= -2
        D[1,-1] = bias
        D[1,[0,self.nr-3]] = -1
        D[2,[-2,-3]] = 0

        #D = inv(solve_banded((1,1),D, eye( self.nr-1)))
        #imshow(D, interpolation='nearest',   cmap='seismic');show()
        return D
    
    def PRESS(self,g, prod,S,U):
        #predictive sum of squares        
        w = 1./(1.+np.exp(g)/S**2)
        ndets = len(prod)
        return np.sum((np.dot(U, (1-w)*prod)/np.einsum('ij,ij,j->i', U,U, 1-w))**2)/ndets
    
        
    def GCV(self,g, prod,S,U):
        #generalized crossvalidation        
        w = 1./(1.+np.exp(g)/S**2)
        ndets = len(prod)
        return (np.sum((((w-1)*prod))**2)+1)/ndets/(1-np.mean(w))**2
    
    
    def FindMin(self,F, x0,dx0,prod,S,U,tol=0.01):
        #stupid but robust minimum searching algorithm.

        fg = F(x0, prod, S,U)
        while abs(dx0) > tol:
            fg2 = F(x0+dx0, prod,S,U)
                                
            if fg2 < fg:
                fg = fg2
                x0 += dx0                
                continue
            else:
                dx0/=-2.
                
        return x0, np.log(fg2)

    def calc_tomo_window(self, n_blocks = 1):
        #calculate tomography of data splitted in n_blocks using optimised minimum fisher regularisation
        #Odstrcil, T., et al. "Optimized tomography methods for plasma 
        #emissivity reconstruction at the ASDEX  Upgrade tokamak.
        #" Review of Scientific Instruments 87.12 (2016): 123505.
        
        #defined independently for LFS and HFS
        reg_level_guess = .7,.6
        reg_level_min = .4, .4

        nfisher = 4
        
        #prepare regularisation operator
        D = self.regul_matrix(biased_edges=True)
        
        
        self.y = np.zeros((self.nt, 2*self.nr-1))
        self.y_err = np.zeros((self.nt, 2*self.nr-1))
        self.chi2lfs = np.zeros(self.nt)
        self.chi2hfs = np.zeros(self.nt)
        self.gamma_lfs = np.zeros(self.nt) 
        self.gamma_hfs = np.zeros(self.nt) 
        self.backprojection = np.zeros_like(self.data)

        itime = np.arange(self.nt)
        tind = 1


        T = self.dL/self.err[:,None]*self.scale
        mean_d = self.data/self.err
        d = self.data/self.err


        #reconstruct both sides independently, just substract LFS contribution from HFS
        for iside,side in enumerate(('LFS', 'HFS')): 
            W = np.ones(self.nr-1)

            
            if side == 'LFS': 
                ind_los = slice(self.nch_hfs,self.nch_hfs+self.nch_lfs)
                ind_space = slice(self.nr,None)
                lfs_contribution = [0]
            else:
                ind_los = slice(0,self.nch_hfs)
                ind_space = slice(0,self.nr-1)
                lfs_contribution = np.dot(T[ind_los],self.y.T).T
            
            Q = np.linspace(0,1,ind_los.stop-ind_los.start)



            for ifisher in range(nfisher):
                #multiply tridiagonal regularisation operator by a diagonal weight matrix W
                WD = np.copy(D)
                
                WD[0,1:]*=W[:-1]
                WD[1]*=W
                WD[2,:-1]*=W[1:]
                
                #transpose the band matrix 
                DTW = np.copy(WD) 
                DTW[0,1:],DTW[2,:-1] = WD[2,:-1],WD[0,1:]
                
                #####    solve Tikhonov regularization (optimised for speed)
                H = solve_banded((1,1),DTW,T[ind_los,ind_space].T, overwrite_ab=True,check_finite=False)
                #fast method to calculate U,S,V = svd(H.T) of rectangular matrix 
                LL = np.dot(H.T, H)
                S2,U = eigh(LL,overwrite_a=True, check_finite=False,lower=True)  
                S2 = np.maximum(S2,1) #singular values S can be negative due to numerical uncertainty 

                #K = dot(dot(U,diag(1/S)),V.T).T
                #print( 'Decomposition accuracy',linalg.norm(dot(T[ind_los,ind_space],K)-eye(len(S))))
                #substract LFS contribution, calculate projection                       

                mean_p = np.dot(mean_d[ind_los]-np.mean(lfs_contribution,0),U)
                
                #guess for regularisation - estimate quantile of log(S^2)
                g0 = np.interp(reg_level_guess[iside], Q, np.log(S2))

                if ifisher == nfisher -1:
                    #last step - find optimal regularisation
                    S = np.sqrt(S2)
                    
                    g0, log_fg2 = self.FindMin(self.GCV, g0 ,1,mean_p,S,U.T) #slowest step
                    #avoid too small regularisation when min of GCV is not found
                    
                    gmin = np.interp(reg_level_min[iside], Q, np.log(S2))
                    g0 = max(g0, gmin)
                    
                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)
                    
                    V = np.dot(H,U/S)  
                    V = solve_banded((1,1),WD,V, overwrite_ab=True,overwrite_b=True,check_finite=False) 
                else:
                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)
                    
                    #calculate y without evaluating V explicitly
                    y = np.dot(H,np.dot(U/S2,w*mean_p))
                    #final inversion of mean solution , reconstruction
                    y = solve_banded((1,1),WD,y, overwrite_ab=True,overwrite_b=True,check_finite=False) 
                    
                    #plt.plot(y)
                    #weight matrix for the next iteration
                    W = 1/np.maximum(y,1e-10)**.5
            
            p = np.dot(d[ind_los]-lfs_contribution,U)
            y = np.dot((w/S)*p,V.T)
    
            self.backprojection[ind_los] = fit = np.dot(p*w,U.T)+lfs_contribution
            chi2 = np.sum((d[ind_los]-fit)**2)/np.size(fit)
            gamma = np.interp(g0,np.log(S2),Q)

            
            if side == 'LFS':
                self.chi2lfs = chi2
                self.gamma_lfs = gamma
            else:
                self.chi2hfs = chi2
                self.gamma_hfs = gamma

            self.y[:,ind_space] = y
            #correction for under/over estimated data uncertainty
            self.y_err[:,ind_space] = np.sqrt(np.dot(V**2,(w/S)**2))#*chi2[:,None])

        
        self.backprojection *= self.err
            
            
        self.y *= self.scale
        self.y_err *= self.scale
        self.R_grid_b = (self.R_grid[1:]+ self.R_grid[:-1])/2
 
        return self.R_grid_b, self.y,self.y_err, self.backprojection


    #Calculates inversion for input data only
    def calc_tomo_sig(self, data, err, rVec, n_blocks = 100):
        #calculate tomography of data splitted in n_blocks using optimised minimum fisher regularisation
        #Odstrcil, T., et al. "Optimized tomography methods for plasma 
        #emissivity reconstruction at the ASDEX  Upgrade tokamak.
        #" Review of Scientific Instruments 87.12 (2016): 123505.
        
        #defined independently for LFS and HFS
        reg_level_guess = .7,.6
        reg_level_min = .4, .4

        nfisher = 4

        self.nr = 200
        self.nt = 1
        self.err = err

        self.scale = np.median(self.data)
        
        #prepare regularisation operator
        D = self.regul_matrix(biased_edges=True)
        
        
        self.y = np.zeros((self.nt, 2*self.nr-1))
        self.y_err = np.zeros((self.nt, 2*self.nr-1))
        self.chi2lfs = np.zeros(self.nt)
        self.chi2hfs = np.zeros(self.nt)
        self.gamma_lfs = np.zeros(self.nt) 
        self.gamma_hfs = np.zeros(self.nt) 
        self.backprojection = np.zeros_like(data)

        itime = np.arange(self.nt)
        tinds = np.array_split(itime, n_blocks)


        
        for ib, tind in enumerate(tinds):

            T = self.dL/self.err[tind].mean(0)[:,None]*self.scale
            mean_d = self.data[tind].mean(0)/self.err[tind].mean(0)
            d = self.data[tind]/self.err[tind]


            #reconstruct both sides independently, just substract LFS contribution from HFS
            for iside,side in enumerate(('LFS', 'HFS')): 
                W = np.ones(self.nr-1)

                
                if side == 'LFS': 
                    ind_los = slice(self.nch_hfs,self.nch_hfs+self.nch_lfs)
                    ind_space = slice(self.nr,None)
                    lfs_contribution = [0]
                else:
                    ind_los = slice(0,self.nch_hfs)
                    ind_space = slice(0,self.nr-1)
                    lfs_contribution = np.dot(T[ind_los],self.y[tind].T).T
                
                Q = np.linspace(0,1,ind_los.stop-ind_los.start)

    
    
                for ifisher in range(nfisher):
                    #multiply tridiagonal regularisation operator by a diagonal weight matrix W
                    WD = np.copy(D)
                    
                    WD[0,1:]*=W[:-1]
                    WD[1]*=W
                    WD[2,:-1]*=W[1:]
                    
                    #transpose the band matrix 
                    DTW = np.copy(WD) 
                    DTW[0,1:],DTW[2,:-1] = WD[2,:-1],WD[0,1:]
                    
                    #####    solve Tikhonov regularization (optimised for speed)
                    H = solve_banded((1,1),DTW,T[ind_los,ind_space].T, overwrite_ab=True,check_finite=False)
                    #fast method to calculate U,S,V = svd(H.T) of rectangular matrix 
                    LL = np.dot(H.T, H)
                    S2,U = eigh(LL,overwrite_a=True, check_finite=False,lower=True)  
                    S2 = np.maximum(S2,1) #singular values S can be negative due to numerical uncertainty 

                    #K = dot(dot(U,diag(1/S)),V.T).T
                    #print( 'Decomposition accuracy',linalg.norm(dot(T[ind_los,ind_space],K)-eye(len(S))))
                    #substract LFS contribution, calculate projection                       

                    mean_p = np.dot(mean_d[ind_los]-np.mean(lfs_contribution,0),U)
                    
                    #guess for regularisation - estimate quantile of log(S^2)
                    g0 = np.interp(reg_level_guess[iside], Q, np.log(S2))

                    if ifisher == nfisher -1:
                        #last step - find optimal regularisation
                        S = np.sqrt(S2)
                        
                        g0, log_fg2 = self.FindMin(self.GCV, g0 ,1,mean_p,S,U.T) #slowest step
                        #avoid too small regularisation when min of GCV is not found
                        
                        gmin = np.interp(reg_level_min[iside], Q, np.log(S2))
                        g0 = max(g0, gmin)
                        
                        #filtering factor
                        w = 1./(1.+np.exp(g0)/S2)
                        
                        V = np.dot(H,U/S)  
                        V = solve_banded((1,1),WD,V, overwrite_ab=True,overwrite_b=True,check_finite=False) 
                    else:
                        #filtering factor
                        w = 1./(1.+np.exp(g0)/S2)
                        
                        #calculate y without evaluating V explicitly
                        y = np.dot(H,np.dot(U/S2,w*mean_p))
                        #final inversion of mean solution , reconstruction
                        y = solve_banded((1,1),WD,y, overwrite_ab=True,overwrite_b=True,check_finite=False) 
                        
                        #plt.plot(y)
                        #weight matrix for the next iteration
                        W = 1/np.maximum(y,1e-10)**.5
                
                p = np.dot(d[:,ind_los]-lfs_contribution,U)
                y = np.dot((w/S)*p,V.T)
        
                self.backprojection[tind,ind_los] = fit = np.dot(p*w,U.T)+lfs_contribution
                chi2 = np.sum((d[:,ind_los]-fit)**2,1)/np.size(fit,1)
                gamma = np.interp(g0,np.log(S2),Q)
  
                
                if side == 'LFS':
                    self.chi2lfs[tind] = chi2
                    self.gamma_lfs[tind] = gamma
                else:
                    self.chi2hfs[tind] = chi2
                    self.gamma_hfs[tind] = gamma

                self.y[tind,ind_space] = y
                #correction for under/over estimated data uncertainty
                self.y_err[tind,ind_space] = np.sqrt(np.dot(V**2,(w/S)**2))#*chi2[:,None])

            
            self.backprojection[tind] *= self.err[tind].mean(0)
            
            
        self.y *= self.scale
        self.y_err *= self.scale
        self.R_grid_b = (self.R_grid[1:]+ self.R_grid[:-1])/2
 
        return self.R_grid_b, self.y,self.y_err, self.backprojection

    def calc_tomo(self, n_blocks = 10):
        #calculate tomography of data splitted in n_blocks using optimised minimum fisher regularisation
        #Odstrcil, T., et al. "Optimized tomography methods for plasma 
        #emissivity reconstruction at the ASDEX  Upgrade tokamak.
        #" Review of Scientific Instruments 87.12 (2016): 123505.
        
        #defined independently for LFS and HFS
        reg_level_guess = .7,.6
        reg_level_min = .4, .4

        nfisher = 4
        
        #prepare regularisation operator
        D = self.regul_matrix(biased_edges=True)
        
        
        self.y = np.zeros((self.nt, self.nr-1))
        self.y_err = np.zeros((self.nt, self.nr-1))
        self.chi2lfs = np.zeros(self.nt)
        # self.chi2hfs = np.zeros(self.nt)
        self.gamma_lfs = np.zeros(self.nt) 
        # self.gamma_hfs = np.zeros(self.nt) 
        self.backprojection = np.zeros_like(self.data)

        itime = np.arange(self.nt)
        tinds = np.array_split(itime, n_blocks)

        for ib, tind in enumerate(tinds):

            ## cmod mod: see where the error is 0 and replace to avoid dividing by 0
            mean_err_zero_inds = np.where(self.err[tind].mean(0) == 0)
            err_zero_inds = np.where(self.err[tind] == 0)

            T = self.dL/self.err[tind].mean(0)[:,None]*self.scale
            mean_d = self.data[tind].mean(0)/self.err[tind].mean(0)
            d = self.data[tind]/self.err[tind]
            
            ## replace infinities and nans with 0
            T[mean_err_zero_inds] = 0
            mean_d[mean_err_zero_inds] = 0
            d[err_zero_inds] = 0

            #reconstruct both sides independently, just substract LFS contribution from HFS
            # for iside,side in enumerate(('LFS', 'HFS')): 

            iside = 0
            side = 'LFS'

            W = np.ones(self.nr-1)

                
                # if side == 'LFS': 
            ind_los = slice(0,self.nch_lfs)
            ind_space = slice(0,self.nr-1)
            lfs_contribution = [0]
                # else:
                #     ind_los = slice(0,self.nch_hfs)
                #     ind_space = slice(0,self.nr-1)
                #     lfs_contribution = np.dot(T[ind_los],self.y[tind].T).T
                
            Q = np.linspace(0,1,ind_los.stop-ind_los.start)

    
    
            for ifisher in range(nfisher):
                #multiply tridiagonal regularisation operator by a diagonal weight matrix W
                WD = np.copy(D)
                
                WD[0,1:]*=W[:-1]
                WD[1]*=W
                WD[2,:-1]*=W[1:]
                
                #transpose the band matrix 
                DTW = np.copy(WD) 
                DTW[0,1:],DTW[2,:-1] = WD[2,:-1],WD[0,1:]

                #####    solve Tikhonov regularization (optimised for speed)
                H = solve_banded((1,1),DTW,T[ind_los,ind_space].T, overwrite_ab=True,check_finite=False)
                #fast method to calculate U,S,V = svd(H.T) of rectangular matrix 
                LL = np.dot(H.T, H)
                S2,U = eigh(LL,overwrite_a=True, check_finite=False,lower=True)  
                S2 = np.maximum(S2,1) #singular values S can be negative due to numerical uncertainty 

                #K = dot(dot(U,diag(1/S)),V.T).T
                #print( 'Decomposition accuracy',linalg.norm(dot(T[ind_los,ind_space],K)-eye(len(S))))
                #substract LFS contribution, calculate projection                       

                mean_p = np.dot(mean_d[ind_los]-np.mean(lfs_contribution,0),U)
                
                #guess for regularisation - estimate quantile of log(S^2)
                g0 = np.interp(reg_level_guess[iside], Q, np.log(S2))

                if ifisher == nfisher -1:
                    #last step - find optimal regularisation
                    S = np.sqrt(S2)
                    
                    g0, log_fg2 = self.FindMin(self.GCV, g0 ,1,mean_p,S,U.T) #slowest step
                    #avoid too small regularisation when min of GCV is not found
                    
                    gmin = np.interp(reg_level_min[iside], Q, np.log(S2))
                    g0 = max(g0, gmin)
                    
                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)
                    
                    V = np.dot(H,U/S)  
                    V = solve_banded((1,1),WD,V, overwrite_ab=True,overwrite_b=True,check_finite=False) 
                else:
                    #filtering factor
                    w = 1./(1.+np.exp(g0)/S2)
                    
                    #calculate y without evaluating V explicitly
                    y = np.dot(H,np.dot(U/S2,w*mean_p))
                    #final inversion of mean solution , reconstruction
                    y = solve_banded((1,1),WD,y, overwrite_ab=True,overwrite_b=True,check_finite=False) 
                    
                    #plt.plot(y)
                    #weight matrix for the next iteration
                    W = 1/np.maximum(y,1e-10)**.5
                
            p = np.dot(d[:,ind_los]-lfs_contribution,U)
            y = np.dot((w/S)*p,V.T)
    
            self.backprojection[tind,ind_los] = fit = np.dot(p*w,U.T)+lfs_contribution
            chi2 = np.sum((d[:,ind_los]-fit)**2,1)/np.size(fit,1)
            gamma = np.interp(g0,np.log(S2),Q)

            
            # if side == 'LFS':
            self.chi2lfs[tind] = chi2
            self.gamma_lfs[tind] = gamma
            # else:
            #     self.chi2hfs[tind] = chi2
            #     self.gamma_hfs[tind] = gamma

            self.y[tind,ind_space] = y
            #correction for under/over estimated data uncertainty
            self.y_err[tind,ind_space] = np.sqrt(np.dot(V**2,(w/S)**2))#*chi2[:,None])

        
        self.backprojection[tind] *= self.err[tind].mean(0)
            
            
        self.y *= self.scale
        self.y_err *= self.scale

        self.R_grid_b = (self.R_grid[1:]+ self.R_grid[:-1])/2
 
        return self.R_grid_b, self.y,self.y_err, self.backprojection
        
        
        
    def show_reconstruction(self):
                
        f,ax = plt.subplots(2,2, sharex='col', figsize=(9,9))
        ax_time = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='y')
        slide_time = Slider(ax_time, 'Time:', 0.0, self.tvec[-1], valinit=0, valstep=0.001, valfmt='%1.4fs')


        f.subplots_adjust(bottom=.2)

        r = self.R_grid_b
        nr = self.nr

        confidence_hfs = ax[0,0].fill_between(r[:nr], r[:nr]*0, r[:nr]*0, alpha=.5, facecolor='b', edgecolor='None')
        tomo_mean_hfs, = ax[0,0].plot([],[], lw=2 )
        
        ind_hfs = slice(0,self.nch_hfs)

        errorbar_hfs = ax[1,0].errorbar(0,np.nan,0,  capsize = 4
                                        ,c='g',marker='o',fillstyle='none',ls='none')
        retro_hfs, = ax[1,0].plot([],[],'b-')
        retro_hfs2, = ax[1,0].plot([],[],'bx')



        confidence_lfs = ax[0,1].fill_between(r[nr:], r[nr:]*0, r[nr:]*0, alpha=.5, facecolor='b', edgecolor='None')
        tomo_mean_lfs, = ax[0,1].plot([],[], lw=2 )
        
        ind_lfs = slice(self.nch_hfs,self.nch_hfs+self.nch_lfs+2)
        errorbar_lfs = ax[1,1].errorbar(0,np.nan,0,  capsize = 4
                                        ,c='g',marker='o',fillstyle='none',ls='none')
        retro_lfs, = ax[1,1].plot([],[],'b-')
        retro_lfs2, = ax[1,1].plot([],[],'bx')

        
        ax[0,0].axvline(self.hfs_min,c='k',ls='--')
        ax[0,0].axvline(self.hfs_max,c='k',ls='--')
        ax[0,1].axvline(self.lfs_max,c='k',ls='--')
        ax[0,1].axvline(self.lfs_min,c='k',ls='--')
        
        ax[0,0].axvline(self.inner_wall_R,c='k',ls='-')
        ax[0,1].axvline(self.outer_wall_R ,c='k',ls='-')

        ax[0,0].axhline(0,c='k')
        ax[0,1].axhline(0,c='k')
        ax[1,0].axhline(0,c='k')
        ax[1,1].axhline(0,c='k')



        self.multi = MultiCursor(f.canvas, ax.flatten(), color='r', lw=1)
        ax[0,0].set_xlim(self.hfs_min-.02, self.hfs_max+.05)
        ax[0,1].set_xlim(self.lfs_min-.01, self.lfs_max+.05)

        ax[0,0].set_ylim(0, mquantiles(self.y[:,:self.nr].max(1),0.99)*1.1)
        ax[0,1].set_ylim(0, mquantiles(self.y[:,self.nr:].max(1),0.99)*1.1)

        ax[1,0].set_ylim(0, (self.data+self.err)[:,ind_hfs].max()*1.1)
        ax[1,1].set_ylim(0, (self.data+self.err)[:,ind_lfs].max()*1.1)
        ax[1,1].set_xlabel('R [m]')
        ax[1,0].set_xlabel('R [m]')
        ax[0,0].set_ylabel('Emissivity [ph/m$^3$s]')
        ax[1,0].set_ylabel('Brightness [ph/m$^2$s]')
        
        title = f.suptitle('')
        titleHFS = ax[0,0].set_title('')
        titleLFS = ax[0,1].set_title('')


        def update(val):
            it = np.argmin(np.abs(self.tvec-val))
            update_fill_between(confidence_hfs,r[:nr],self.y[it,:nr]-self.y_err[it,:nr],self.y[it,:nr]+self.y_err[it,:nr],-np.inf,np.inf)
            update_fill_between(confidence_lfs,r[nr:],self.y[it,nr:]-self.y_err[it,nr:],self.y[it,nr:]+self.y_err[it,nr:],-np.inf,np.inf)
            
            tomo_mean_hfs.set_data(r[:nr],self.y[it,:nr])
            tomo_mean_lfs.set_data(r[nr:],self.y[it,nr:])

            update_errorbar(errorbar_hfs,self.R_tg[ind_hfs], self.data[it,ind_hfs], self.err[it,ind_hfs])
            update_errorbar(errorbar_lfs,self.R_tg[ind_lfs], self.data[it,ind_lfs], self.err[it,ind_lfs])


            backprojection = np.dot(self.dL_back, self.y[it])
            
            retro_lfs.set_data(self.R_grid[nr:], backprojection[nr:])
            retro_hfs.set_data(self.R_grid[:nr], backprojection[:nr])
            retro_lfs2.set_data(self.R_tg[ind_lfs], self.backprojection[it,ind_lfs])
            retro_hfs2.set_data(self.R_tg[ind_hfs], self.backprojection[it,ind_hfs])



            title.set_text('#%d  %.3fs'%(self.shot, self.tvec[it]))
            titleHFS.set_text('HFS: $\chi^2/ndoF$: %.2f  $\gamma$: %.2f'%( self.chi2hfs[it], self.gamma_hfs[it]) )
            titleLFS.set_text('LFS: $\chi^2/ndoF$: %.2f  $\gamma$: %.2f'%( self.chi2lfs[it], self.gamma_lfs[it]) )

            f.canvas.draw_idle()


        def on_key(event):
            dt = (self.tvec[-1]-self.tvec[0])/(len(self.tvec)-1)
            tnew = slide_time.val
            
            if hasattr(event,'step'):
                #scroll_event
                tnew += event.step*dt

            elif 'left' == event.key:
                #key_press_event
                tnew -= dt
                    
            elif 'right' == event.key:
                tnew += dt
                
            tnew = min(max(tnew,self.tvec[0]),self.tvec[-1])
            slide_time.set_val(tnew)
            update(tnew)


        self.cid = f.canvas.mpl_connect('key_press_event',   on_key)
        self.cid_scroll  = f.canvas.mpl_connect('scroll_event',on_key)


        slide_time.on_changed(update)
        update(0)
        axbutton = plt.axes([0.85, 0.1, 0.1, 0.05])
        self.save_button = Button(axbutton, 'Save')
        self.save_button.on_clicked(self.save)

        
        f2,ax2 = plt.subplots(2, sharex = True)
        
        im1 = ax2[0].imshow(self.y[:,:nr].T, extent=(self.tvec[0], self.tvec[-1], self.R_grid[0],self.R_grid[self.nr-1]),
                            aspect='auto',interpolation='nearest',vmin=0, vmax = mquantiles(self.y[:,:nr], .99))
        im2 = ax2[1].imshow(self.y[:,nr:].T, extent=(self.tvec[0], self.tvec[-1], self.R_grid[self.nr],self.R_grid[-1]),
                            aspect='auto',interpolation='nearest',vmin=0, vmax = mquantiles(self.y[:,nr:], .99))
        f2.colorbar(im1,ax=ax2[0])
        f2.colorbar(im2,ax=ax2[1])

        self.multi2 = MultiCursor(f2.canvas, ax2, color='r', lw=1)
        ax2[1].set_xlabel('Time [s]')
        ax2[0].set_ylabel('R [m]')
        ax2[1].set_ylabel('R [m]')
       
       #estimate time when discharge starts and ends 

        signal = np.mean(self.data[:,:self.nch_hfs]-self.data[:,:self.nch_hfs][:,(-1,)],1)\
                +np.mean(self.data[:,self.nch_hfs:]-self.data[:,self.nch_hfs:][:,(-1,)],1)
        ind = signal > .01*np.max(signal)
        imin,imax = np.where(ind)[0][[0,-1]]
        tind = slice(imin,imax)
        ax2[0].set_xlim(0,self.tvec[imax])

        
        plt.show()

        
    def save(self,event,fileLoc = ''):
        
        
        #estimate time when discharge starts and ends 

        signal = np.mean(self.data[:,:self.nch_hfs]-self.data[:,:self.nch_hfs][:,(-1,)],1)\
                +np.mean(self.data[:,self.nch_hfs:]-self.data[:,self.nch_hfs:][:,(-1,)],1)
        ind = signal > .01*np.max(signal)
        imin,imax = np.where(ind)[0][[0,-1]]
        tind = slice(imin,imax)
        
        np.savez_compressed(fileLoc+'LLAMA_%d.npz'%self.shot,
                          time=np.single(self.tvec[tind]),
                          R_tg = np.single(self.R_tg),
                          Z_tg = np.single(self.Z_tg),
                          backprojection = np.single(self.backprojection[tind]),
                          radial_grid = np.single(self.R_grid_b),
                          emiss = np.single(self.y[tind]),
                          emiss_err = np.single(self.y_err[tind]),
                          brightness=np.single(self.data[tind]),
                          brightness_err = np.single(self.err[tind]))
        print('saved')

    def package2return(self,tWindow=False):
        
        dDict = {}

        dDict['R_tg'] = self.R_tg
        dDict['Z_tg'] = self.Z_tg
        dDict['radial_grid'] = self.R_grid_b

        if tWindow:
        
            ind_min = np.where(self.tvec <= tWindow[0])[0][-1]
            ind_max = np.where(self.tvec >= tWindow[1])[0][0]

            dDict['time'] = self.tvec[ind_min:ind_max+1]
            dDict['backprojection'] = self.backprojection[ind_min:ind_max+1]
            dDict['emiss'] = self.y[ind_min:ind_max+1]

            dDict['emiss_err'] = self.y_err[ind_min:ind_max+1]
            dDict['brightness'] = self.data[ind_min:ind_max+1]

            dDict['brightness_err'] = self.err[ind_min:ind_max+1]
            dDict['tAvr'] = self.time_avg

            try: 
                dDict['brightness_raw'] = self.rawData[ind_min:ind_max+1]
                return dDict
            except:

                return dDict

        else:
            dDict['time'] = self.tvec
            dDict['backprojection'] = self.backprojection
            dDict['emiss'] = self.y

            dDict['emiss_err'] = self.y_err
            dDict['brightness'] = self.data

            dDict['brightness_err'] = self.err
            dDict['tAvr'] = self.time_avg

            try: 
                dDict['brightness_raw'] = self.rawData
                return dDict
            except:

                return dDict


    
        


def main():



    parser = argparse.ArgumentParser( usage='You must specify the shot number, use --help for more info')
    
    parser.add_argument('--shot', metavar='S', type=int, help='shot number', default=180916)
    parser.add_argument('--tAvr', metavar='S', type=float, help='averaging time window in s', default=0.005)
    args = parser.parse_args()

    tomo = LLAMA_tomography(args.shot,time_avg=args.tAvr)
    tomo.load_geometry()

    t = time()
    print('Fetching data...')
    tomo.load_data()
    print('Done in %.1fs'%(time()-t))
    t = time()
    print('Computing tomography...')
    tomo.calc_tomo()
    print('Done in %.1fs'%(time()-t))
    
    #embed()

    #removing display for fast saving-AR May 22 2020
    tomo.show_reconstruction()
    #tomo.save(None)

def batchRun():

    tAvr = 0.001
    for i in range(11):

        shot = i+180906
        tomo = LLAMA_tomography(shot,time_avg=tAvr)
        tomo.load_geometry()

        t = time()
        print('Fetching data...')
        tomo.load_data()
        print('Done in %.1fs'%(time()-t))
        t = time()
        print('Computing tomography...')
        tomo.calc_tomo()
        print('Done in %.1fs'%(time()-t))

        tomo.save(None)

def tomoWindow(tWindows):
    #tWindows = np.asarray([[2826.344104,2932.67296143],[2857.94490723,2969.6394873],[ 180910,180910]])

    tAvr = 0.001

    shot = tWindows[2,0]

    tomo = LLAMA_tomography(shot,time_avg=tAvr)
    tomo.load_geometry()

    t = time()
    print('Fetching data...')
    tomo.load_data_Window(tWindows)
    print('Done in %.1fs'%(time()-t))
    t = time()
    print('Computing tomography...')
    tomo.calc_tomo_window()
    print('Done in %.1fs'%(time()-t))
    
    #embed()

    #removing display for fast saving-AR May 22 2020
    #tomo.show_reconstruction()
    #tomo.save(None)

    return tomo.package2return()

def tomoWindowSpec(tWindows,tAvr):
    #tWindows = np.asarray([[2826.344104,2932.67296143],[2857.94490723,2969.6394873],[ 180910,180910]])


    shot = tWindows[2,0]



    tomo = LLAMA_tomography(shot,time_avg=tAvr)
    tomo.load_geometry()

    t = time()
    print('Fetching data...')
    tomo.load_data_specWindow(tWindows)
    print('Done in %.1fs'%(time()-t))
    t = time()
    print('Computing tomography...')
    tomo.calc_tomo()
    print('Done in %.1fs'%(time()-t))
    
    #embed()

    #removing display for fast saving-AR May 22 2020
    #tomo.show_reconstruction()
    #tomo.save(None)

    return tomo.package2return()

def tomoELMt(shot,tAvr,elmT):


    tomo = LLAMA_tomography(shot,time_avg=tAvr)
    LFScList = 11
    tomo.load_geometry_removeChan(LFScList)

    t = time()
    print('Fetching data...')
    tomo.load_data_ELMt(elmT,LFScList)
    print('Done in %.1fs'%(time()-t))
    t = time()
    print('Computing tomography...')
    tomo.calc_tomo()
    print('Done in %.1fs'%(time()-t))
    

    return tomo.package2return()

def tomoReturn(shot, tAvr = 0.01):
    
    #tWindows = np.asarray([[2826.344104,2932.67296143],[2857.94490723,2969.6394873],[ 180910,180910]])



    tomo = LLAMA_tomography(shot,time_avg=tAvr)
    tomo.load_geometry()

    t = time()
    print('Fetching data...')
    tomo.load_data()
    print('Done in %.1fs'%(time()-t))
    t = time()
    print('Computing tomography...')
    tomo.calc_tomo()
    print('Done in %.1fs'%(time()-t))
    


    return tomo.package2return()

def tomoRunSave(shot, tAvr = 0.01,fileLoc=''):




    tomo = LLAMA_tomography(shot,time_avg=tAvr)
    tomo.load_geometry()

    t = time()
    print('Fetching data...')
    tomo.load_data()
    print('Done in %.1fs'%(time()-t))
    t = time()
    print('Computing tomography...')
    tomo.calc_tomo()
    print('Done in %.1fs'%(time()-t))
    


    tomo.save(None, fileLoc)


def tomoCMOD(shot,system,tWindow=False,r_end=0.93,sys_err=5):

    print('Inversion has zero at '+str(r_end) + ' m')
    tomo = LLAMA_tomography(shot,system,time_avg=0) # cmod brightness data already smoothed in time
    tomo.load_geometry(r_end=r_end,sys_err=sys_err)
    tomo.load_data(r_end=r_end)
    tomo.calc_tomo()

    return tomo.tvec,tomo.R_grid_b,tomo.y,tomo.y_err,tomo.backprojection
    
    # # assume there is only one window for the time being
    # return tomo.package2return(tWindow[0])


if __name__ == "__main__":
    #main()
    #batchRun()
    tWindows = np.asarray([[2826.344104,2932.67296143],[2857.94490723,2969.6394873],[ 180910,180910]])
    tomoWindow(tWindows)



