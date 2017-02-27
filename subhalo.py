""" 


"""
import h5py
import time 
import numpy as np

# --- Local --- 
#from treepm import subhalo_io 
import sham_hack as sham
import subhalo_io_hack as subhalo_io
from utilities import utility as wetzel_util
import util as Util


class CentralSH_MergerHistory(object): 
    def __init__(self, scatter=0.2, source='li-march'): 
        ''' Class that tracks the merger history of central subhalos at snapshot = 1
        '''
        # SHAM properties 
        self.scatter = scatter
        self.source = source

    def File(self, nsnap_lookback):
        # file name 
        f = ''.join([
            '/home/users/hahn/projects/central_quenching/CenQue/dat/wetzel_tree/', # directory
            'CentralSubhalo.mergertrack.'
            'SHAM.', self.source, '.scat', str(self.scatter), '.', 
            'lookback', str(nsnap_lookback), '.hdf5'
            ])

        return f 

    def Build(self, nsnap_lookback=23): 
        ''' Track back merger history of centrals at snapshot 1 until nsnap_lookback
        '''
        nsnaps = range(1, nsnap_lookback+1) 
    
        # read in TreePM Subhalo snapshots from z~0.0502 to z~1.0833
        # note: 250 here is the simulation box length (Mpc/h comoving)
        sub = subhalo_io.Treepm.read('subhalo', 250, zis = nsnaps) 

        # assign M* to subhalos by SHAMing onto M_max of subhalos 
        sham.assign(sub, 'm.star', 
                scat = self.scatter, 
                dis_mf = 0.0, 
                source = self.source,
                sham_prop='m.max',
                zis = nsnaps) 

        # identify central subhalos @ snapshot = 1
        snap1_cens = wetzel_util.utility_catalog.indices_ilk(sub[1], ilk = 'cen') 
        tracked = snap1_cens # index of snapshot1 centrals
         
        snap_dicts = {} 
        for i_snap in nsnaps[1:]: 
            # parent subhalo with the highest M_max are designated *the* parent subhalo
            # the "main branch" of the merger tree 
            i_parents = np.repeat(-1, len(tracked))
            w_parents = np.where((tracked >= 0) & (sub[i_snap-1]['par.i'][tracked] >= 0))
            # index snapshot1 central main branch at snapshot i_snap
            i_parents[w_parents] = sub[i_snap-1]['par.i'][tracked[w_parents]] 

            # the other parent subhalos (ones *not* designated the parent)
            i_sub = np.arange(len(sub[i_snap]['m.star']))
            rest = i_sub[np.in1d(i_sub, i_parents, invert=True)] 

            child, i_child = wetzel_util.utility_catalog.indices_tree(sub, i_snap, i_snap-1, cis=rest, get_indices=True)
            
            # tracked subhalos *with* mergers
            track_merg = np.arange(len(tracked))[np.in1d(tracked, child)] 
            
            dM_merg = np.zeros(len(tracked))
            t_start = time.time() 
            for ii in track_merg: # (~6 mins)
                b_cen = np.where(child == tracked[ii]) # subhalos that "merge" to the tracked central 
                
                # calculate the dM_merg, which is the total  M* of galaxies of subhalos that 
                # merge onto the main branch subhalo.
                dM_merg[ii] += np.log10(np.sum(10**sub[i_snap]['m.star'][rest[b_cen]]))
            print 't = ', (time.time()-t_start)/60., ' minutes' 

            snap_dict = {}
            snap_dict['dM_merg'] = dM_merg
            snap_dict['i_parent'] = i_parents
            snap_dict['M_halo.parent'] = np.repeat(-999., len(i_parents))
            snap_dict['M_max.parent'] = np.repeat(-999., len(i_parents))
            snap_dict['M_star.parent'] = np.repeat(-999., len(i_parents))

            snap_dict['M_halo.parent'][w_parents] = sub[i_snap]['halo.m'][i_parents[w_parents]]
            snap_dict['M_max.parent'][w_parents] = sub[i_snap]['m.max'][i_parents[w_parents]]
            snap_dict['M_star.parent'][w_parents] = sub[i_snap]['m.star'][i_parents[w_parents]]

            snap_dicts['snap'+str(i_snap)] = snap_dict

            tracked = i_parents # update for next snapshot step 

        # save to hdf5 file  
        f = h5py.File(self.File(nsnap_lookback), 'w') 
        grp = f.create_group('data') 
        
        # metadata 
        grp.attrs['SHAM.scatter'] = self.scatter
        grp.attrs['SHAM.source'] = self.source
        grp.attrs['nsnap_lookback'] = nsnap_lookback
        
        # snapshot 1 central subhalo data
        for k in sub[1].keys(): 
            if k == 'm.star': 
                key = 'M_star' 
            elif k == 'm.max': 
                key = 'M_max' 
            elif k == 'halo.m': 
                key = 'M_halo' 
            elif k == 'ssfr': 
                continue 
            else: 
                key = k
            grp.create_dataset(key, data=sub[1][k][snap1_cens]) 

        # snapshot n > 1 data
        for isnap in nsnaps[1:]: 
            for k in snap_dicts['snap'+str(isnap)].keys(): 
                grp.create_dataset('snap'+str(isnap)+'.'+k, data=snap_dicts['snap'+str(isnap)][k])

        f.close() 
        return None


def Jeremy_IOWrap(): 
    ''' Reformat CentralSH_MergerHistory output to match Jeremy's 
    pre-historic output specifications (ASCII format). 

    requested data format : 
        - position, velocity, M_gal_merger/M_gal_sham, galaxy mass, halo mass 
    '''
    ST = CentralSH_MergerHistory()
    ST_file = ST.File(23) # file name 
    f = h5py.File(ST_file, 'r')
    grp = f['data'] 

    # central galaxy cuts (number of cuts on the central subhalos/galaxies) 
    cuts = np.where(grp['M_star'].value > 9.4) # hardcoded mass cut 

    data_list = [] 
    for key in ['pos', 'vel']: # position, velocity 
        for ii in range(3): 
            data_list.append(grp[key].value[cuts][:,ii]) 
    data_list.append(grp['M_star'].value[cuts])     # M_* galaxy stellar mass 
    data_list.append(grp['M_halo'].value[cuts])     # M_halo
    data_list.append(grp['M_max'].value[cuts])      # M_max

    # calculate M_gal_merger
    M_merger_tot = np.zeros(len(cuts[0]))
    for i_snap in range(2, 24): 
        has_merge = np.where(grp['snap'+str(i_snap)+'.dM_merg'].value[cuts] > 1.) # has merged and dM_merger > 1 

        M_merger_tot[has_merge] += 10**(grp['snap'+str(i_snap)+'.dM_merg'].value[cuts])[has_merge]

    notzero = np.where(M_merger_tot > 0) 
    M_merger_tot[notzero] = np.log10(M_merger_tot[notzero]) 
    assert M_merger_tot.min() >= 0.

    # calculate f_merger = M_gal_merger / M_sham(z=0)
    f_merger = np.zeros(len(cuts[0]))
    f_merger[notzero] = 10**(M_merger_tot[notzero] - grp['M_star'].value[cuts][notzero])
    data_list.append(f_merger)  
    
    # data column header
    hdr = ', '.join(['pos0 [Mpc]', 'pos1 [Mpc]', 'pos2 [Mpc]', 
        'vel0 [Mpc/Gyr]', 'vel1 [Mpc/Gyr]', 'vel2 [Mpc/Gyr]', 'M_star', 'M_halo', 'M_max', 'f_merger'])
    # data column format  
    fmt = ['%10.5f' for i in range(len(data_list))]
    # file name 
    output_file = ''.join(['/mount/sirocco1/hahn/Gal_agar/catalog/', 
        'Centrals.MergerHistory', '.Lookback', str(23), '.Mcut', str(9.4), '.dat']) 
    # save to ascii file  
    np.savetxt(
            output_file, 
            (np.vstack(np.array(data_list))).T, 
            fmt=fmt, delimiter='\t', header=hdr) 
    return None 

    




if __name__=='__main__': 
    #ST = CentralSH_MergerHistory()
    #ST.Build(nsnap_lookback=23)
    Jeremy_IOWrap()
