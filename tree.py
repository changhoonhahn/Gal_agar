



class TrimmedTree(object): 
    ''' Class that tracks 'trimmed trees' for central halos at snapshot = 1

    Here's how it roughly works
    o*              z = 0 
    | \ 
    o*  o           z = 0.1
    |\  |\ \
    o*o o o o       z = 0.2
    
    * main branch of halo
    we trim the tree and only keep the first level of leaves 
    
    o*              z = 0 
    | \ 
    o*  o           z = 0.1
    |\  
    o*o             z = 0.2


    '''
    def __init__(self): 
        pass

    def Build(self): 
        # read in TreePM Subhalo snapshots from z~0.0502 all the way back to z~4.2798
        # note: 250 here is the simulation box length (Mpc/h comoving)
        sub = subhalo_io.Treepm.read('subhalo', 250, zis=range(1, 35)) 


