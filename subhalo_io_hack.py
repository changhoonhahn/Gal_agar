'''
Reads in subtree.dat & halotree.dat from tree directory.

Masses in log {M_sun}, distances in {Mpc comoving}.

simulation_length {Mpc/h}    particle_number_per_dimension    particle_mass {M_sun}
50    256    7.710e8
50    512    8.976e7
64    800    4.934e7
100   800    1.882e8
125   1024   1.753e8
200   1500   2.284e8
250   2048   1.976e8
720   1500   1.066e10

lcdm250
zi    aexp      redshift  t {Gyr} t_wid {Gyr}
 0    1.0000    0.0000    13.8099 0.6771
 1    0.9522    0.0502    13.1328 0.6604
 2    0.9068    0.1028    12.4724 0.6453
 3    0.8635    0.1581    11.8271 0.6291
 4    0.8222    0.2162    11.1980 0.6087
 5    0.7830    0.2771    10.5893 0.5905
 6    0.7456    0.3412     9.9988 0.5700
 7    0.7100    0.4085     9.4289 0.5505
 8    0.6760    0.4793     8.8783 0.5259
 9    0.6438    0.5533     8.3525 0.5060
10    0.6130    0.6313     7.8464 0.4830
11    0.5837    0.7132     7.3635 0.4587
12    0.5559    0.7989     6.9048 0.4382
13    0.5293    0.8893     6.4665 0.4152
14    0.5040    0.9841     6.0513 0.3916
15    0.4800    1.0833     5.6597 0.3724
16    0.4570    1.1882     5.2873 0.3496
17    0.4352    1.2978     4.9378 0.3298
18    0.4144    1.4131     4.6080 0.3099
19    0.3946    1.5342     4.2980 0.2901
20    0.3758    1.6610     4.0079 0.2735
21    0.3578    1.7949     3.7343 0.2541
22    0.3408    1.9343     3.4802 0.2394
23    0.3245    2.0817     3.2408 0.2235
24    0.3090    2.2362     3.0172 0.2094
25    0.2942    2.3990     2.8078 0.1942
26    0.2802    2.5689     2.6136 0.1821
27    0.2668    2.7481     2.4315 0.1704
28    0.2540    2.9370     2.2611 0.1576
29    0.2419    3.1339     2.1035 0.1466
30    0.2304    3.3403     1.9569 0.1371
31    0.2194    3.5579     1.8198 0.1280
32    0.2089    3.7870     1.6918 0.1192
33    0.1989    4.0277     1.5726 0.1106
34    0.1894    4.2798     1.4620 0.0000
'''

# system -----
from __future__ import division
from numpy import log10, Inf, int32, float32
import numpy as np
import os
import glob
import copy
# local -----
from utilities import constants as const
from utilities import cosmology
from utilities import utility as ut

RESEARCH_DIRECTORY = ''

#===================================================================================================
# read in
#===================================================================================================
# Martin's TreePM ----------
class TreepmClass(ut.io.SayClass):
    '''
    Read [sub]halo catalog snapshots, return as list class.
    '''
    def __init__(self, sigma_8=0.8):
        self.treepm_directory = '/data1/arwetzel/'
        self.dimen_num = 3
        self.particle_num = {
            # connect simulation box length {Mpc/h comoving} to number of particles per dimension.
            50: 256,    # 512
            64: 800,
            100: 800,
            125: 1024,
            200: 1500,
            250: 2048,
            720: 1500
        }
        self.sigma_8 = sigma_8

    def read(self, catalog_kind='subhalo', box_length=250, zis=1, cat_in=None):
        '''
        Read snapshots in input range.

        Import catalog kind (subhalo, halo, both), simulation box length {Mpc/h comoving},
        snapshot index range, input [sub]halo catalog to appending snapshots to.
        '''
        if catalog_kind == 'both':
            sub = self.read('subhalo', box_length, zis)
            hal = self.read('halo', box_length, zis)
            return sub, hal
        elif catalog_kind == 'subhalo':
            cat = ut.array.ListClass()
            catz = ut.array.DictClass()
            catz['pos'] = []    # position (3D) of most bound particle {Mpc/h -> Mpc comoving}
            catz['vel'] = []    # velocity (3D) {Mpc/h/Gyr -> Mpc/Gyr comoving}
            #catz['m.bound'] = []    # mass of subhalo {M_sun}
            #catz['vel.circ.max'] = []    # maximum of circular velocity {km/s physical}
            catz['m.max'] = []    # maximum mass in history {M_sun}
            #catz['vel.circ.peak'] = []    # max of max of circular velocity {km/s physical}
            catz['ilk'] = []
            # 1 = central, 2 = virtual central, 0 = satellite, -1 = virtual satellite,
            # -2 = virtual satellite with no central, -3 = virtual satellite with no halo
            catz['par.i'] = []    # index of parent, at previous snapshot, with highest M_max
            #catz['par.n.i'] = []    # index of next parent to same child, at same snapshot
            catz['chi.i'] = []    # index of child, at next snapshot
            catz['m.frac.min'] = []    # minimum M_bound / M_max experienced
            #catz['m.max.rat.raw'] = []    # M_max ratio of two highest M_max parents (< 1)
            catz['cen.i'] = []    # index of central subhalo in same halo (can be self)
            catz['dist.cen'] = []    # distance from central {Mpc/h -> Mpc comoving}
            #catz['sat.i'] = []    # index of next highest M_max satellite in same halo
            catz['halo.i'] = []    # index of host halo
            catz['halo.m'] = []    # FoF mass of host halo {M_sun}
            #catz['inf.last.zi'] = []    # snapshot before fell into current halo
            #catz['inf.last.i'] = []    # index before fell into current halo
            #catz['inf.dif.zi'] = []    # snapshot when sat/central last was central/sat
            #catz['inf.dif.i'] = []    # index when sat/central last was central/sat
            catz['inf.first.zi'] = []    # snapshot before first fell into another halo
            catz['inf.first.i'] = []    # index before first fell into another halo
            # derived ----------
            catz['m.star'] = []    # stellar mass {M_sun}
            #catz['mag.r'] = []    # magnitude in r-band
            #catz['m.max.rat'] = []    # same as m.max.rat.raw, but incorporates disrupted subhalos
            #catz['m.star.rat'] = []    # M_star ratio of two highest M_star parents
            catz['ssfr'] = []
            #catz['dn4k'] = []
            #catz['g-r'] = []
        elif catalog_kind == 'halo':
            cat = ut.array.ListClass()
            catz = ut.array.DictClass()
            catz['pos'] = []    # 3D position of most bound particle {Mpc/h -> Mpc comoving}
            #catz['vel'] = []    # 3D velocity {Mpc/h/Gyr -> Mpc/Gyr comoving}
            catz['m.fof'] = []    # FoF mass {M_sun}
            catz['vel.circ.max'] = []    # maximum circular vel = sqrt(G * M(r) / r) {km/s physical}
            #catz['v.disp'] = []    # velocity dispersion {km/s} (DM, 1D, no hubble flow)
            catz['m.200c'] = []    # M_200c from unweighted fit of NFW M(< r) {M_sun}
            catz['c.200c'] = []    # concentration (200c) from unweighted fit of NFW M(< r)
            #catz['c.fof'] = []    # concentration derive from r_{2/3 mass} / r_{1/3 mass}
            catz['par.i'] = []    # index of parent, at previous snapshot, with max mass
            #catz['par.n.i'] = []    # index of next parent to same child, at same snapshot
            #catz['chi.i'] = []    # index of child, at next snapshot
            #catz['m.fof.rat'] = []    # FoF mass ratio of two highest mass parents (< 1)
            #catz['cen.i'] = []     # index of central subhalo
        else:
            raise ValueError('catalog kind = %s not valid' % catalog_kind)
        self.directory_sim = self.treepm_directory + 'lcdm%d/' % box_length
        self.directory_tree = self.directory_sim + 'tree/'
        zis = ut.array.arrayize(zis)
        # read auxilliary data
        catz.Cosmo = self.read_cosmology()
        catz.info = {
            'kind': catalog_kind,
            'source': 'L%d' % box_length,
            'box.length.no-hubble': float(box_length),
            'box.length': float(box_length) / catz.Cosmo['hubble'],
            'particle.num': self.particle_num[box_length],
            'particle.m': catz.Cosmo.particle_mass(box_length, self.particle_num[box_length])
        }
        if catalog_kind == 'subhalo':
            catz.info['m.kind'] = 'm.max'
        elif catalog_kind == 'halo':
            catz.info['m.kind'] = 'm.fof'
        if cat_in is not None:
            cat = cat_in
        else:
            cat.Cosmo = catz.Cosmo
            cat.info = catz.info
            cat.snap = self.read_snapshot_time()
        # sanity check on snapshot range
        if zis.max() >= cat.snap['z'].size:
            zis = zis[zis < cat.snap['z'].size]
        for u_all in xrange(zis.max() + 1):
            if u_all >= len(cat):
                cat.append({})
        for zi in zis:
            cat[zi] = copy.copy(catz)
            cat[zi].snaps = cat.snap
            cat[zi].snap = {}
            for k in cat.snap.dtype.names:
                cat[zi].snap[k] = cat.snap[k][zi]
            cat[zi].snap['i'] = zi
            self.read_snapshot(cat[zi], zi)
        return cat

    def read_snapshot(self, catz, zi):
        '''
        Read single snapshot, assign to catalog.

        Import catalog of [sub]halo at snapshot, snapshot index.
        '''
        if catz.info['kind'] == 'subhalo':
            file_name_base = 'subhalo_tree_%s.dat' % str(zi).zfill(2)
            props = [
                'pos', 'vel', 'm.bound', 'vel.circ.max', 'm.max', 'vel.circ.peak', 'ilk', 'par.i',
                'par.n.i', 'chi.i', 'm.frac.min', 'm.max.rat.raw', 'inf.last.zi', 'inf.last.i',
                'inf.first.zi', 'inf.first.i',
                'cen.i', 'dist.cen', 'sat.i', 'halo.i', 'halo.m'
            ]
            #'inf.dif.zi', 'inf.dif.i', 
        elif catz.info['kind'] == 'halo':
            file_name_base = 'halo_tree_%s.dat' % str(zi).zfill(2)
            props = [
                'pos', 'vel', 'm.fof', 'vel.circ.max', 'vel.disp', 'm.200c', 'c.200c', 'c.fof',
                'par.i', 'par.n.i', 'chi.i', 'm.fof.rat', 'cen.i'
            ]
            if catz.info['box.length.no-hubble'] == 720:
                props.remove('cen.i')
        file_name = self.directory_tree + file_name_base
        file_in = open(file_name, 'r')
        obj_num = int(np.fromfile(file_in, int32, 1))
        for prop in props:
            if len(prop) > 2 and (prop[-2:] == '.i' or prop[-3:] == '.zi' or prop == 'ilk'):
                dtype = int32
            else:
                dtype = float32
            self.read_property(file_in, catz, prop, dtype, obj_num)
            if prop == 'c.200c' and 'c.200c' in catz:
                catz['c.200c'][1:] = catz['c.200c'][1:].clip(1.5, 40)
        file_in.close()
        self.say('read %8d %s from %s' % (obj_num, catz.info['kind'], file_name_base))

    def read_property(self, file_in, cat, prop, dtype, obj_num):
        '''
        Read property from file, assign to [sub]halo catalog.

        Import input file, catalog at snapshot, property, data type, number of objects.
        '''
        if prop in ('pos', 'vel'):
            dimen_num = self.dimen_num
        else:
            dimen_num = 1
        temp = np.fromfile(file_in, dtype, obj_num * dimen_num)
        if prop in cat:
            if dimen_num > 1:
                cat[prop] = temp.reshape(obj_num, dimen_num)
            else:
                cat[prop] = temp
            if prop in ('pos', 'vel', 'dist.cen'):
                cat[prop] /= cat.Cosmo['hubble']

    def read_snapshot_time(self):
        '''
        Read time properties at snapshot, assign to simulation class/dictionary, return.
        '''
        file_name_base = 'snapshot.txt'
        file_name = self.directory_sim + file_name_base
        snaptime = np.loadtxt(file_name, comments='#', usecols=[1, 2, 3, 4, 5], dtype=[
            ('a', float32),
            ('z', float32),
            ('t', float32),
            ('t.wid', float32),
            ('t.hubble', float32)    # Hubble time = 1 / H(t) {Gyr}
        ])
        self.say('read ' + file_name)
        return snaptime

    def read_cosmology(self):
        '''
        Read cosmological parameters, save as class, return.
        '''
        Cosmo = cosmology.CosmologyClass(self.sigma_8)
        file_name_base = 'cosmology.txt'
        file_name = self.directory_sim + file_name_base
        file_in = open(file_name, 'r')
        for line in file_in:
            cin = np.array(line.split(), float32)
            if len(cin) == 7:
                Cosmo['omega_m'] = cin[0]
                Cosmo['omega_l'] = cin[1]
                Cosmo['w'] = cin[2]
                omega_b_0_h2 = cin[3]
                Cosmo['hubble'] = cin[4]
                Cosmo['n_s'] = cin[5]
                Cosmo['omega_b'] = omega_b_0_h2 / Cosmo['hubble'] ** 2
                break
            else:
                raise ValueError('%s not formatted correctly' % file_name_base)
        file_in.close()
        if (Cosmo['omega_m'] < 0 or Cosmo['omega_m'] > 0.5 or
            Cosmo['omega_l'] < 0.5 or Cosmo['omega_l'] > 1):
            self.say('! read strange cosmology in %s' % file_name_base)
        return Cosmo

    def pickle_first_infall(self, direction='read', sub=None, zis=np.arange(34), m_max_min=10.5):
        '''
        Read/write first infall times & subhalo indices at all snapshots in input range.

        Import pickle direction (read, write), subhalo catalog, properties for file name.
        '''
        inf_name = 'inf.first'
        file_name_short = 'subhalo_inf.first_m.max%.1f' % (m_max_min)
        file_name_base = (self.treepm_directory +
                          'lcdm%d/tree/' % sub.info['box.length.no-hubble'] + file_name_short)
        if direction == 'write':
            zis = np.arange(max(zis) + 1)
            siss = []
            inf_ziss = []
            inf_siss = []
            for zi in zis:
                subz = sub[zi]
                sis = ut.array.elements(subz[inf_name + '.zi'], [zi, Inf])
                siss.append(sis)
                inf_ziss.append(subz[inf_name + '.zi'][sis])
                inf_siss.append(subz[inf_name + '.i'][sis])
            ut.io.pickle_object(file_name_base, direction, [zis, siss, inf_ziss, inf_siss])
        elif direction == 'read':
            zis = ut.array.arrayize(zis)
            _zis_in, siss, inf_ziss, inf_siss = ut.io.pickle_object(file_name_base, direction)
            for zi in zis:
                subz = sub[zi]
                subz[inf_name + '.zi'] = np.zeros(subz['par.i'].size, int32) - 1
                subz[inf_name + '.zi'][siss[zi]] = inf_ziss[zi]
                subz[inf_name + '.i'] = ut.array.initialize_array(subz['par.i'].size)
                subz[inf_name + '.i'][siss[zi]] = inf_siss[zi]
        else:
            raise ValueError('not recognize i/o direction = %s' % direction)

Treepm = TreepmClass()


# ELVIS ----------
class ElvisClass(ut.io.SayClass):
    '''
    Read subhalo catalog from ELVIS suite.

    Host halo mass range: [12.0037, 12.4539]
    '''
    def __init__(self):
        self.simulation_directory = (RESEARCH_DIRECTORY +
                                     'project/current/sfr_satellite_milky-way/ELVIS/')
        self.catalog_directory = self.simulation_directory + 'ELVIS_halo_Catalogs/'
        self.catalog_directories = ['PairedCatalogs/', 'IsolatedCatalogs/', 'HiResCatalogs/']
        self.tree_directory = self.simulation_directory + 'ELVIS_Main_Branches/'
        self.tree_directories = ['PairedTrees/', 'IsolatedTrees/', 'HiResTrees/']
        self.Cosmo = cosmology.CosmologyClass(hubble=0.71, omega_m=0.266, omega_l=0.734,
                                              sigma_8=0.801, n_s=0.963, w=-1.0)

        self.dimen_num = 3

    def read_all(self, subz=None, sub=None, use_hires=False):
        '''
        Read catalogs of all subhalos at z = 0 & their main progenitor trees.

        Import catalog of subhalos at z = 0, catalog of main progenitor trees,
        whether to use high-resolution (versus normal-resolution) runs.
        '''
        if subz is None:
            subz = self.read_catalogs(use_hires=use_hires)
        if sub is None:
            sub = self.read_trees(use_hires=use_hires)

        for k in sub:
            if np.ndim(sub[k]) == 1 and k not in subz:
                subz[k] = sub[k]

        return subz

    def read_catalogs(self, pair_kind='', use_hires=False):
        '''
        Read catalogs of all subhalos at z = 0.

        Import host halo pair kind to read ('', 'single', 'pair'),
        whether to read high-resolution (versus normal-resolution) runs.
        '''
        subz = ut.array.DictClass()
        subz['id'] = []    # id of (sub)halo
        subz['pos'] = []    # position {Mpc}
        subz['vel'] = []    # velocity {km / s physical}
        subz['vel.circ.max'] = []    # maximum circular velocity {km / s physical}
        subz['vel.circ.peak'] = []    # peak of max circular velocity in history {km / s physical}
        subz['m.vir'] = []    # mass virial (centrals) or bound (satellites) {M_sun}
        subz['m.max'] = []
        subz['r.vir'] = []    # radius virial (centrals) or bound (satellites) {kpc}
        subz['r.vel.circ.max'] = []    # radius where circular velocity peaks {kpc}
        subz['m.max.a'] = []    # latest scale factor at which M_vir = M_max
        subz['m.max.z'] = []    # latest redshift at which M_vir = M_max
        subz['m.star'] = []    # stellar mass from modified SHAM in Garrison-Kimmel++2013 {M_sun}
        #subz['m.star.behroozi'] = []
        #subz['particle.num'] = []    # number of particles in subhalo
        subz['host.id'] = []    # id of host subhalo (can be satellite), [1, max], -1 if am central
        subz['cen.id'] = []    # id of central subhalo, [1, max]
        # derived -----
        subz['host.i'] = []    # index of host subhalo (can be satellite). -1 if am central
        subz['cen.i'] = []    # index of central subhalo
        subz['ilk'] = []    # 1 = central, 2 = primary host halo
                            # 0 = satellite in primary host halo, -1 = satellite in subhalo
                            # -2 = satellite in another halo, -3 = sat in subhalo in another halo
        subz['halo.name'] = []    # name of primary host halo
        subz['halo.is-pair'] = []    # whether host halo is Local Group-like pair
        subz['dist.cen'] = []    # distance to central subhalo {kpc comoving}
        subz['dist.vir'] = []    # distance to central subhalo / R_vir of host halo
        subz['halo.m'] = []    # viral mass of host halo {M_sun}

        catalog_directories = self.catalog_directories

        # determine directories to read
        if pair_kind:
            if pair_kind == 'pair':
                catalog_directories.remove('IsolatedCatalogs/')
                catalog_directories.remove('HiResCatalogs/')
                if use_hires:
                    raise ValueError('hi-res only for isolated host halos')
            elif pair_kind == 'single':
                catalog_directories.remove('PairedCatalogs/')
            else:
                raise ValueError('not recognize pair_kind = %s' % pair_kind)
        if not use_hires and 'HiResCatalogs/' in catalog_directories:
            catalog_directories.remove('HiResCatalogs/')

        # read files in relevant directories
        for directory in catalog_directories:
            file_names = glob.glob(self.catalog_directory + directory + '*.txt')
            for file_name in file_names:
                file_name_base = file_name.replace(self.catalog_directory + directory, '')
                if file_name_base in ('iHall.txt', 'iKauket.txt', 'iScylla.txt') and use_hires:
                    continue
                self.read_catalog(file_name_base, subz)

        subz.Cosmo = self.Cosmo
        subz.snap = {
            'a': 1.0, 'z': 0, 't': self.Cosmo.age(0),
            't.hubble': const.Gyr_per_sec / self.Cosmo.hubble_parameter(0)
        }
        subz.info = {
            'kind': 'subhalo', 'source': 'elvis',
            'box.length.no-hubble': 50, 'box.length': 50 / self.Cosmo['hubble'],
            'particle.num': 4096,
            'particle.m': self.Cosmo.particle_mass(50 / self.Cosmo['hubble'], 4096)
        }
        if pair_kind:
            subz.info['pair.kind'] = pair_kind

        # assign derived orbital properties
        self.assign_orbit(subz)

        self.say('read %6d %s from all catalogs' % (subz['id'].size, subz.info['kind']))

        return subz

    def read_catalog(self, file_name_base, subz_all=None, offset_num=0):
        '''
        Read catalog of subhalos at z = 0 for a single host halo [pair].

        Import file name base, subhalo dictionary to append to, subhalo index offset.
        '''
        if '_HiRes' in file_name_base:
            catalog_directory = self.catalog_directory + 'HiResCatalogs/'
        elif file_name_base[0] == 'i':
            catalog_directory = self.catalog_directory + 'IsolatedCatalogs/'
        elif '&' in file_name_base:
            catalog_directory = self.catalog_directory + 'PairedCatalogs/'

        if '.txt' not in file_name_base:
            file_name_base += '.txt'

        file_name = catalog_directory + file_name_base

        prop_in = np.loadtxt(file_name, comments='#', dtype=[
            ('id', int32),
            ('pos.x', float32), ('pos.y', float32), ('pos.z', float32),
            ('vel.x', float32), ('vel.y', float32), ('vel.z', float32),
            ('vel.circ.max', float32), ('vel.circ.peak', float32),
            ('m.vir', float32), ('m.max', float32),
            ('r.vir', float32), ('r.vel.circ.max', float32),
            ('m.max.a', float32), ('m.star', float32), ('m.star.behroozi', float32),
            ('particle.num', int32), ('host.id', int32), ('cen.id', int32),
        ])

        # transfer to dictionary
        subz = ut.array.DictClass()
        for k in prop_in.dtype.names:
            subz[k] = prop_in[k]
            if k[:2] == 'm.' and k[-2:] != '.a':
                subz[k][subz[k] > 0] = log10(subz[k])    # convert to log mass
            #elif k[:2] == 'r.':
            #    subz[k][subz[k] > 0] /= 1000    # convert radii to Mpc

        # assign phase-space coordinates to 3-D array
        for prop in ('pos', 'vel'):
            subz[prop] = np.transpose([subz[prop + '.x'], subz[prop + '.y'], subz[prop + '.z']])
            del(subz[prop + '.x'], subz[prop + '.y'], subz[prop + '.z'])

        # assign information on simulation & cosmology
        subz.Cosmo = self.Cosmo
        subz.snap = {
            'a': 1.0, 'z': 0, 't': self.Cosmo.age(0),
            't.hubble': const.Gyr_per_sec / self.Cosmo.hubble_parameter(0)
        }
        if 'HiRes' in file_name_base:
            particle_num = 8192
        else:
            particle_num = 4096
        subz.info = {
            'kind': 'subhalo', 'source': 'elvis',
            'box.length.no-hubble': 50, 'box.length': 50 / self.Cosmo['hubble'],
            'particle.num': particle_num,
            'particle.m': self.Cosmo.particle_mass(50 / self.Cosmo['hubble'], particle_num)
        }

        # assign subhalo type
        ut.catalog.assign_id_to_index(subz, 'id')
        # -99 is satellite without central in this catalog
        subz['ilk'] = np.zeros(subz['id'].size, subz['id'].dtype) - 99
        # 1 is central subhalo
        subz['ilk'][subz['cen.id'] == -1] = 1
        # 2 is primary host halo
        subz['ilk'][0] = 2
        if '&' in file_name_base:
            subz['ilk'][1] = 2
        sis_sat = ut.array.elements(subz['cen.id'], [1, Inf])
        cen_is = subz['id-to-index'][subz['cen.id'][sis_sat]]
        sis_sat = sis_sat[cen_is >= 0]    # make sure satellite has central in this catalog
        # -2 is normal satellite in non-primary host halo
        subz['ilk'][sis_sat] = -2
        # -3 is satellite of subhalo in non-primary host halo
        subz['ilk'][sis_sat[subz['host.id'][sis_sat] != subz['cen.id'][sis_sat]]] = -3
        # 0 is normal satellite in primary host halo
        subz['ilk'][sis_sat[subz['cen.id'][sis_sat] == subz['id'][0]]] = 0
        if '&' in file_name_base:
            subz['ilk'][sis_sat[subz['cen.id'][sis_sat] == subz['id'][1]]] = 0
        # -1 is satellite of subhalo in primary host halo
        sis_sat = ut.array.elements(subz['ilk'], 0)
        subz['ilk'][sis_sat[subz['host.id'][sis_sat] != subz['cen.id'][sis_sat]]] = -1

        # assign information on host halo
        subz['halo.name'] = np.r_[[file_name_base] * subz['id'].size]
        subz['halo.is-pair'] = np.zeros(subz['id'].size, int32)
        if '&' in file_name_base:
            subz['halo.is-pair'] += 1

        subz['halo.m'] = np.zeros(subz['id'].size, subz['m.vir'].dtype) - 1
        sis_cen = ut.array.elements(subz['ilk'], [1, Inf])
        subz['halo.m'][sis_cen] = subz['m.vir'][sis_cen]
        sis_sat = ut.array.elements(subz['ilk'], [-9, 0.1])
        cen_is = subz['id-to-index'][subz['cen.id'][sis_sat]]
        subz['halo.m'][sis_sat] = subz['m.vir'][cen_is]

        # assign distance to central
        subz['dist.cen'] = np.zeros(subz['id'].size, subz['pos'].dtype) - 1
        sis_cen = ut.array.elements(subz['ilk'], [1, Inf])
        subz['dist.cen'][sis_cen] = 0
        sis_sat = ut.array.elements(subz['ilk'], [-9, 0.1])
        cen_is = subz['id-to-index'][subz['cen.id'][sis_sat]]
        subz['dist.cen'][sis_sat] = ut.coord.distance('scalar', subz['pos'][cen_is],
                                                      subz['pos'][sis_sat], subz.info['box.length'])
        subz['dist.cen'][sis_sat] *= 1000    # convert to {kpc}
        subz['dist.vir'] = copy.copy(subz['dist.cen'])
        subz['dist.vir'][sis_sat] /= subz['r.vir'][cen_is]

        # assign formation information
        subz['m.max.z'] = 1 / subz['m.max.a'] - 1

        # assign indices in this list from ids. if joining to all catalogs, offset accordingly.
        if subz_all is not None:
            offset_num = len(subz_all['id'])
        else:
            offset_num = 0
        self.assign_indices(subz, offset_num)

        self.say('read %6d %s from %s' % (prop_in['id'].size, subz.info['kind'], file_name_base))

        if subz_all is not None:
            for k in subz_all:
                if k not in subz:
                    self.say('! warning: %s not in subhalo dictionary' % k)
                elif subz_all[k] == []:
                    subz_all[k] = subz[k]
                else:
                    subz_all[k] = np.concatenate((subz_all[k], subz[k]))
        else:
            return subz

    def read_trees(self, pair_kind='', use_hires=False):
        '''
        Read progenitor trees of subhalos for all host halos.

        Import host halo pair kind to read ('', 'single', 'pair'),
        whether to read high-resolution (versus normal-resolution) runs.
        '''
        sub = ut.array.DictClass()
        sub['id'] = []    # id of (sub)halo
        sub['pos'] = []    # position {Mpc comoving}
        sub['vel'] = []    # velocity {km / s physical}
        sub['vel.circ.max'] = []    # maximum circular velocity {km / s physical}
        sub['m.vir'] = []    # mass virial (centrals) or bound (satellites) {M_sun}
        sub['r.vir'] = []    # radius virial (centrals) or bound (satellites) {kpc comoving}
        sub['r.scale'] = []    # NFW scale radius {kpc -> Mpc comoving}
        #sub['m.star'] = []    # stellar mass from modified SHAM in Garrison-Kimmel++2013 {M_sun}
        sub['host.id'] = []    # id of host subhalo (can be satellite). -1 if am central
        sub['cen.id'] = []    # id of central subhalo
        sub['scale.factor'] = []    # expansion scale factor of snapshot
        # derived -----
        sub['redshift'] = []    # redshift of snapshot
        sub['time'] = []    # time of snapshot
        sub['host.i'] = []    # index of host subhalo (can be satellite). -1 if am central
        sub['cen.i'] = []    # index of central subhalo
        sub['ilk'] = []    # 1 = central, 2 = primary host halo
                            # 0 = satellite in primary host halo, -1 = satellite in subhalo
                            # -2 = satellite in another halo, -3 = sat in subhalo in another halo
        sub['halo.name'] = []    # name of primary host halo
        sub['halo.is-pair'] = []    # whether host halo is Local Group-like pair
        sub['vel.circ.peak'] = []    # peak of max circular velocity in history {km / s physical}
        sub['m.max'] = []    # maximum virial mass in history {M_sun}
        sub['m.max.zi'] = []
        sub['m.max.z'] = []
        sub['m.max.t'] = []
        sub['dist.cen'] = []    # distance to central subhalo {kpc comoving}
        sub['dist.cen.min'] = []
        sub['dist.cen.min.zi'] = []
        sub['dist.cen.min.z'] = []
        sub['dist.cen.min.t'] = []
        sub['dist.vir'] = []    # distance to central subhalo / R_vir of host halo
        sub['dist.vir.min'] = []
        sub['dist.vir.min.zi'] = []
        sub['dist.vir.min.z'] = []
        sub['dist.vir.min.t'] = []
        sub['inf.first.zi'] = []
        sub['inf.first.z'] = []
        sub['inf.first.t'] = []
        sub['inf.last.zi'] = []
        sub['inf.last.z'] = []
        sub['inf.last.t'] = []
        #sub['halo.m'] = []    # viral mass of host halo {M_sun}
        # orbit -----
        sub['vel.rad'] = []
        sub['vel.tan'] = []
        sub['vel.tot'] = []
        sub['momentum.ang'] = []
        sub['energy.kin'] = []

        tree_directories = self.tree_directories

        # determine directories to read
        if pair_kind:
            if pair_kind == 'pair':
                tree_directories.remove('IsolatedTrees/')
                tree_directories.remove('HiResTrees/')
                if use_hires:
                    raise ValueError('hi-res only for isolated host halos')
            elif pair_kind == 'single':
                tree_directories.remove('PairedTrees/')
            else:
                raise ValueError('not recognize pair_kind = %s' % pair_kind)
        if not use_hires and 'HiResTrees/' in tree_directories:
            tree_directories.remove('HiResTrees/')

        for directory in tree_directories:
            dir_name_bases = glob.glob(self.tree_directory + directory + '*')
            for dir_name_base in dir_name_bases:
                dir_name_base = dir_name_base.replace(self.tree_directory + directory, '')
                if dir_name_base in ('iHall', 'iKauket', 'iScylla') and use_hires:
                    continue
                self.read_tree(dir_name_base, sub)

        # self.assign_indices(sub)
        sub.Cosmo = self.Cosmo
        sub.info = {
            'kind': 'subhalo',
            'source': 'elvis',
            'box.length.no-hubble': 50,
            'box.length': 50 / self.Cosmo['hubble'],
            'particle.num': 4096,
            'particle.m': self.Cosmo.particle_mass(50 / self.Cosmo['hubble'], 4096)
        }
        if pair_kind:
            sub.info['pair.kind'] = pair_kind

        self.say('read %6d %s from all catalogs' % (sub['id'].size, sub.info['kind']))

        return sub

    def read_tree(self, halo_directory, sub_all=None, offset_num=0):
        '''
        Read subhalo progenitor tree for single host halo [pair].

        Import halo file name base, subhalo dictionary to append to, subhalo index offset.
        '''
        prop_read = {
            'id': ('ID', int32),    # id of (sub)halo
            'pos.x': ('X', float32),    # position {Mpc comoving}
            'pos.y': ('Y', float32),
            'pos.z': ('Z', float32),
            'vel.x': ('Vx', float32),    # velocity {km / s physical}
            'vel.y': ('Vy', float32),
            'vel.z': ('Vz', float32),
            'vel.circ.max': ('Vmax', float32),    # maximum circular velocity {km / s physical}
            'm.vir': ('Mvir', float32),    # mass virial (centrals), bound (satellites) {M_sun}
            'r.vir': ('Rvir', float32),    # radius virial (cen), bound (sat) {kpc -> Mpc comoving}
            'r.scale': ('Rs', float32),    # NFW scale radius {kpc -> Mpc comoving}
            'virtual': ('Phantom', int32),    # whether am interpolated subhalo
            'host.id': ('pID', int32),    # id of host subhalo (can be satellite). -1 if am central
            'cen.id': ('upID', int32),    # id of central subhalo
            'scale.factor': ('scale', float32),    # expansion scale factor
        }

        if '_HiRes' in halo_directory:
            tree_directory = self.tree_directory + 'HiResTrees/'
        elif halo_directory[0] == 'i':
            tree_directory = self.tree_directory + 'IsolatedTrees/'
        elif '&' in halo_directory:
            tree_directory = self.tree_directory + 'PairedTrees/'
        else:
            raise ValueError('not recognize halo_directory = %s' % halo_directory)

        # transfer to dictionary
        sub = ut.array.DictClass()
        for k in prop_read:
            dir_name = (tree_directory + ut.io.get_safe_path(halo_directory) + prop_read[k][0] +
                        '.txt')
            sub[k] = np.loadtxt(dir_name, comments='#', dtype=prop_read[k][1])
            if len(sub[k].shape) == 2 and sub[k].shape[1] == 150:
                # deal with iSonny, which is regular resolution but has 2x snapshots
                sub[k] = sub[k][:, ::2]
            if k in ('pos.x', 'pos.y', 'pos.z', 'vel.x', 'vel.y', 'vel.z', 'vel.circ.max',
                     'm.vir', 'r.vir', 'r.scale', 'scale.factor'):
                sub[k][sub[k] == 0] = -1
            if k[:2] == 'm.':
                sub[k][sub[k] > 0] = log10(sub[k][sub[k] > 0])    # convert to log mass
            #elif k[:2] == 'r.':
            #    sub[k][sub[k] > 0] /= 1000    # convert radii to Mpc

        # assign phase-space coordinates to 3-D array
        for prop in ('pos', 'vel'):
            sub[prop] = np.transpose([sub[prop + '.x'], sub[prop + '.y'], sub[prop + '.z']],
                                     (1, 2, 0))
            del(sub[prop + '.x'], sub[prop + '.y'], sub[prop + '.z'])

        # assign information on simulation and cosmology
        sub.Cosmo = self.Cosmo
        redshifts = 1 / sub['scale.factor'][0] - 1
        redshifts[redshifts < 0] = -1
        reals = redshifts >= 0
        times = np.zeros(redshifts.size, redshifts.dtype) - 1
        times[reals] = self.Cosmo.age(redshifts[reals])
        hubble_times = np.zeros(redshifts.size, redshifts.dtype) - 1
        hubble_times[reals] = const.Gyr_per_sec / self.Cosmo.hubble_parameter(redshifts[reals])
        sub.snap = {
            'a': sub['scale.factor'][0],
            'z': redshifts,
            't': times,
            't.hubble': const.Gyr_per_sec / self.Cosmo.hubble_parameter(redshifts)
        }
        if 'HiRes' in halo_directory:
            particle_num = 8192
        else:
            particle_num = 4096
        sub.info = {
            'kind': 'subhalo',
            'source': 'elvis',
            'box.length.no-hubble': 50,
            'box.length': 50 / self.Cosmo['hubble'],
            'particle.num': particle_num,
            'particle.m': self.Cosmo.particle_mass(50 / self.Cosmo['hubble'], particle_num)
        }

        # assign information on snapshot time
        reals = sub['scale.factor'] > 0
        sub['redshift'] = np.zeros(sub['scale.factor'].shape, sub['scale.factor'].dtype) - 1
        sub['redshift'][reals] = 1 / sub['scale.factor'][reals] - 1
        sub['time'] = np.zeros(sub['redshift'].shape, sub['redshift'].dtype) - 1
        sub['time'][reals] = self.Cosmo.age(sub['redshift'][reals])

        # assign subhalo type
        self.assign_id_to_index_tree(sub, 'id', 1)
        # -99 is satellite without central in this catalog
        sub['ilk'] = np.zeros(sub['id'].shape, sub['id'].dtype) - 99
        am_cen = sub['cen.id'] == -1
        # 1 is central subhalo
        sub['ilk'][am_cen] = 1
        # 2 is primary host halo
        sub['ilk'][0, am_cen[0]] = 2
        if '&' in halo_directory:
            sub['ilk'][1, am_cen[1]] = 2
        am_sat = sub['cen.id'] > 0
        # make sure satellite has central in this catalog
        am_sat *= sub['id-to-index'][sub['cen.id']] >= 0
        # -2 is normal satellite in non-primary host halo
        sub['ilk'][am_sat] = -2
        # -3 is satellite of subhalo in non-primary host halo
        sub['ilk'][am_sat * (sub['host.id'] != sub['cen.id'])] = -3
        # 0 is normal satellite in primary host halo
        sub['ilk'][am_sat * (sub['cen.id'] == sub['id'][0])] = 0
        if '&' in halo_directory:
            sub['ilk'][am_sat * (sub['cen.id'][am_sat] == sub['id'][1])] = 0
        # -1 is satellite of subhalo in primary host halo
        am_sat = sub['ilk'] == 0
        sub['ilk'][am_sat * (sub['host.id'] != sub['cen.id'])] = -1

        # assign information on host halo
        sub['halo.name'] = np.r_[[halo_directory] * sub['id'].shape[0]]
        sub['halo.is-pair'] = np.zeros(sub['id'].shape[0], int32)
        if '&' in halo_directory:
            sub['halo.is-pair'] += 1

        """
        sub['halo.m'] = np.zeros(sub['id'].shape, sub['m.vir'].dtype) - 1
        sis_cen = sub['ilk'] >= 1
        sub['halo.m'][sis_cen] = sub['m.vir'][sis_cen]
        sis_sat = (sub['ilk'] >= -9) * (sub['ilk'] <= 0)
        cen_is = sub['id-to-index'][sub['cen.id'][sis_sat]]
        sub['halo.m'][sis_sat] = sub['m.vir'][cen_is]
        """

        # assign distance to central
        self.assign_indices_tree(sub, offset_num=0)
        sub['dist.cen'] = np.zeros(sub['id'].shape, sub['pos'].dtype)
        sub['dist.cen'] += Inf
        #am_cen = sub['cen.id'] == -1
        #sub['dist.cen'][am_cen] = 0
        sub['dist.vir'] = copy.copy(sub['dist.cen'])
        for zi in xrange(sub['id'].shape[1]):
            sis_sat = ut.array.elements(sub['ilk'][:, zi], [-9, 0.1])
            cen_is = sub['cen.i'][sis_sat, zi]
            sub['dist.cen'][sis_sat, zi] = ut.coord.distance(
                'scalar', sub['pos'][cen_is, zi], sub['pos'][sis_sat, zi], sub.info['box.length'])
            sub['dist.cen'][sis_sat, zi] *= 1000    # convert to {kpc}
            sub['dist.vir'][sis_sat, zi] = sub['dist.cen'][sis_sat, zi] / sub['r.vir'][cen_is, zi]

        # assign derived orbital properties
        self.assign_orbit(sub)

        # assign history extrema information
        redshifts = sub['redshift'][0]
        times = sub['time'][0]
        props = [
            ['m.max', 'm.vir', np.max, np.argmax],
            ['vel.circ.peak', 'vel.circ.max', np.max, np.argmax],
            ['dist.cen.min', 'dist.cen', np.min, np.argmin],
            ['dist.vir.min', 'dist.vir', np.min, np.argmin]
        ]
        for prop in props:
            sub[prop[0]] = prop[2](sub[prop[1]], 1)
            zi = prop[3](sub[prop[1]], 1)
            sub[prop[0] + '.zi'] = zi
            sub[prop[0] + '.z'] = redshifts[zi]
            sub[prop[0] + '.t'] = times[0] - times[zi]

        # assign information on infall time
        for inf_kind in ('inf.first', 'inf.last'):
            for inf_prop, dtype in (('.zi', int32), ('.z', sub['redshift'].dtype),
                                    ('.t', sub['time'].dtype)):
                sub[inf_kind + inf_prop] = np.zeros(sub['redshift'].shape[0], dtype) - 1

        am_cen = sub['ilk'] >= 1
        am_sat = (sub['ilk'] > -9) * (sub['ilk'] <= 0)
        for si in xrange(sub['redshift'].shape[0]):
            cen_zis = np.where(am_cen[si])[0]
            sat_zis = np.where(am_sat[si])[0]
            if cen_zis.size and cen_zis[0] > 0:
                sub['inf.last.zi'][si] = cen_zis[0]
                sub['inf.last.z'][si] = redshifts[cen_zis[0]]
                sub['inf.last.t'][si] = times[0] - times[cen_zis[0]]
            if sat_zis.size:
                sub['inf.first.zi'][si] = sat_zis[-1] + 1
                sub['inf.first.z'][si] = redshifts[sat_zis[-1] + 1]
                sub['inf.first.t'][si] = times[0] - times[sat_zis[-1] + 1]

        # assign indices in this list from ids
        if sub_all is not None:
            if sub_all['id'] == []:
                offset_num = 0
            else:
                offset_num = len(sub_all['id'][:, 0])
        else:
            offset_num = 0
        self.assign_indices_tree(sub, offset_num)

        self.say('read %6d %s from %s' % (sub['id'].size, sub.info['kind'], halo_directory))

        # combine with full catalog, if input
        if sub_all is not None:
            for k in sub_all:
                if k not in sub:
                    self.say('! warning: %s not in subhalo dictionary' % k)
                elif sub_all[k] == []:
                    sub_all[k] = sub[k]
                else:
                    sub_all[k] = np.concatenate((sub_all[k], sub[k]))
        else:
            return sub

    def assign_id_to_index_tree(self, sub, id_name='id', id_min=1):
        '''
        Assign to tree catalog an array that points from id to index in array.
        Safely set null values to -length of array.

        Import catalog, id name, minimum id.
        '''
        # first, make sure no duplicate id in tree
        if (sub[id_name][sub[id_name] >= id_min].size !=
            np.unique(sub[id_name][sub[id_name] >= id_min]).size):
            raise ValueError('ids are not unique in single halo tree')
        sub[id_name + '-to-index'] = ut.array.initialize_array(sub[id_name].max() + 1)
        #sub[id_name + '-to-index'] = np.arange(sub[id_name].max() + 1)
        indexs = (np.zeros(sub[id_name].shape, sub[id_name].dtype) +
                  np.transpose([ut.array.arange_length(sub[id_name][:, 0])]))
        sub[id_name + '-to-index'][sub[id_name]] = indexs

    def assign_indices_tree(self, sub, offset_num=0):
        '''
        Assign indices from ids of central and host satellite subhalo in tree.

        Import catalog of subhalo tree, index offset (if read catalogs from multiple halos).
        '''
        self.assign_id_to_index_tree(sub, 'id', 1)
        indexs = (np.zeros(sub['id'].shape, sub['id'].dtype) +
                  np.transpose([ut.array.arange_length(sub['id'][:, 0])]))
        am_cen = sub['cen.id'] == -1
        am_sat = sub['cen.id'] > 0
        sub['cen.i'] = np.zeros(sub['id'].shape, sub['id'].dtype) - 10000000
        sub['cen.i'][am_cen] = indexs[am_cen] + offset_num
        sub['cen.i'][am_sat] = sub['id-to-index'][sub['cen.id'][am_sat]] + offset_num
        sub['host.i'] = np.zeros(sub['id'].shape, sub['id'].dtype) - 10000000
        sub['host.i'][am_sat] = sub['id-to-index'][sub['host.id'][am_sat]] + offset_num
        del(sub['id-to-index'])

    def assign_indices(self, subz, offset_num=0):
        '''
        Assign indices from ids of central and host satellite subhalo.

        Import catalog of subhalo at snapshot, index offset (if read catalogs from multiple halos).
        '''
        ut.catalog.assign_id_to_index(subz, 'id')
        sis_cen = ut.array.elements(subz['cen.id'], -1)
        sis_sat = ut.array.elements(subz['cen.id'], [1, Inf])
        subz['cen.i'] = np.zeros(subz['id'].size, subz['id'].dtype) - 10000000
        subz['cen.i'][sis_cen] = sis_cen + offset_num
        subz['cen.i'][sis_sat] = subz['id-to-index'][subz['cen.id'][sis_sat]] + offset_num
        subz['host.i'] = np.zeros(subz['id'].size, subz['id'].dtype) - 10000000
        subz['host.i'][sis_sat] = subz['id-to-index'][subz['host.id'][sis_sat]] + offset_num
        del(subz['id-to-index'])

    def get_catalog_snapshot(self, sub, zi):
        '''
        Get subhalo catalog at snapshot, to match Treepm format.

        Import subhalo progenitor tree catalog, snapshot index.
        '''
        subz = ut.array.DictClass()
        for k in sub:
            if np.ndim(sub[k]) > 1:
                subz[k] = sub[k][:, zi]
        subz.Cosmo = sub.Cosmo
        subz.info = sub.info
        subz.snap = {}
        subz.snap = {}
        for k in sub.snap:
            subz.snap[k] = sub.snap[k][zi]
        return subz

    def assign_orbit(self, sub):
        '''
        Assign orbital information to satellites of primary host halo[s].

        Import catalog of subhalo [at snapshot].
        '''
        from halo import halo_orbit

        if np.ndim(sub['id']) == 1:
            sis_sat = ut.array.elements(sub['ilk'], [-1, 0.1])
            orb = halo_orbit.get_orbit_dict(sub, sis_sat, sub['cen.i'][sis_sat])
            for k in orb:
                sub[k] = np.zeros(sub['id'].size, dtype=orb[k].dtype) - 1
                sub[k][sis_sat] = orb[k]
        else:
            for zi in ut.array.arange_length(sub['id'][0]):
                subz = self.get_catalog_snapshot(sub, zi)
                sis_sat = ut.array.elements(subz['ilk'], [-1, 0.1])
                orb = halo_orbit.get_orbit_dict(subz, sis_sat, subz['cen.i'][sis_sat])
                for k in orb:
                    if np.ndim(orb[k]) == 1:
                        # assign only scalar quantities
                        if zi == 0:
                            sub[k] = np.zeros(sub['id'].shape, dtype=orb[k].dtype) - 1
                        if k in sub:
                            sub[k][sis_sat, zi] = orb[k]

Elvis = ElvisClass()


# read/write pickle format ----------
class SubhaloPropClass(ut.io.SayClass):
    '''
    Read/write additional properties for subhalo catalog, mostly for satellite histories.
    '''
    def __init__(self):
        self.directory_prop = RESEARCH_DIRECTORY + 'project/current/subhalo_data/'

    def pickle_prop(self, direction='read', sub=None, zi_now=1, zi_max=15, prop='dist.cen.int',
                    zis_assign=[]):
        '''
        Read/write: dist.cen.int {Mpc comoving}, M_star (> 9.7 for galaxy mock), tid.vel, tid.ene.

        Import pickle direction (read, write), subhalo catalog,
        current & maximum shapshot for history, property to pickle.
        '''
        if not zis_assign:
            zis_assign = range(zi_now, zi_max + 1)
        if prop == 'dist.cen.int':
            prop_lim = [1e-10, Inf]
            m_kind = 'm.star'
            m_lim = [9.7, Inf]
            file_name_short = '%s_zi%d-%d_%s%.1f' % (prop, zi_now, zi_max, m_kind, m_lim[0])
        elif prop == 'm.star':
            prop_lim = [8.5, Inf]
            scat = 0.15
            file_name_short = '%s_zi%d-%d_%s%.1f_scat%.2f' % (prop, zi_now, zi_max, m_kind,
                                                              prop_lim[0], scat)
        elif prop == 'dist.neig-5' or prop[:3] == 'tid':
            if prop[:3] == 'tid':
                prop_lim = [1e-10, Inf]
                neig_dist_max = 0.29
            elif prop == 'dist.neig-5':
                prop_lim = [1e-10, 19.99]
                neig_dist_max = 3.57
            m_kind = 'm.star'
            m_min = 9.7
            neig_m_min = 8.5
            neig_num_max = 200
            file_name_short = '%s_zi%d-%d_%s%.1f_neig_%s%.1f_num%2d_dist%.1f' % (
                prop, zi_now, zi_max, m_kind, m_min, m_kind, neig_m_min, neig_num_max,
                neig_dist_max)
        else:
            raise ValueError('not recognize prop = %s' % prop)
        file_name_base = self.directory_prop + file_name_short
        zis = np.arange(zi_now, zi_max + 1)
        if direction == 'write':
            propss = []
            siss = []
            for zi in zis:
                sis = ut.array.elements(sub[zi][prop], prop_lim)
                propss.append(sub[zi][prop][sis])
                siss.append(sis)
                self.say('zi = %2d | %7d above prop min' % (zi, sis.size))
            ut.io.pickle_object(file_name_base, direction, [zis, siss, propss])
        elif direction == 'read':
            zis, siss, propss = ut.io.pickle_object(file_name_base, direction)
            snapshot_num = len(sub)
            for zi in xrange(snapshot_num):
                sub[zi][prop] = []
            for zii, zi in enumerate(zis):
                if zi < snapshot_num:
                    sub[zi][prop] = np.zeros(sub[zi]['m.max'].size, float32)
                    sub[zi][prop][1:] -= 1
                    sub[zi][prop][siss[zii]] = propss[zii]
                    self.say('snapshot = %d | read %d objects' % (zi, siss[zii].size))
                else:
                    self.say('subhalo goes back to snap = %d, only assign %s to there' %
                             (zi - 1, prop))
                    break
        else:
            raise ValueError('not recognize i/o direction = %s' % direction)

    # satellite history properties ----------
    def pickle_first_infall_znow(self, direction='read', subz=None, m_kind='m.max',
                                 m_lim=[10.8, Inf], dis_mf=0.007):
        '''
        Import pickle direction (read, write), catalog of subhalo at snapshot,
        properties for file name.
        Have m.star = [9.7, Inf] & m.max = [10.8, Inf] to zi.max = 34.
        '''
        inf_name = 'inf.first'
        file_name_short = '%s_zi%d_%s%.1f_mf%.3f' % (inf_name, subz.snap['i'], m_kind, m_lim[0],
                                                     dis_mf)
        file_name_base = self.directory_prop + file_name_short
        if direction == 'write':
            sis = ut.array.elements(subz[inf_name + '.zi'], [subz.snap['i'], Inf])
            ut.io.pickle_object(file_name_base, direction, [sis, subz[inf_name + '.zi'][sis],
                                                            subz[inf_name + '.i'][sis]])
        elif direction == 'read':
            sis, inf_zis, inf_sis = ut.io.pickle_object(file_name_base, direction)
            subz[inf_name + '.zi'] = np.zeros(subz[subz.keys()[0]].size, int32) - 1
            subz[inf_name + '.zi'][sis] = inf_zis
            subz[inf_name + '.i'] = ut.array.initialize_array(subz[subz.keys()[0]].size)
            subz[inf_name + '.i'][sis] = inf_sis
        else:
            raise ValueError('not recognize i/o direction = %s' % direction)

    def pickle_prop_extrema_sat(self, direction='read', subz=None, zi_max=15, m_kind='m.star',
                                m_lim=[9.7, Inf], neig_m_lim=[8.5, Inf]):
        '''
        Read/write extrema history data for satellites.

        Import pickle direction (read, write), catalog of subhalo at snapshot,
        properties for file name.
        '''
        file_name_short = 'sat_prop_extrema_zi%d-%d_%s%.1f_neig_%s%.1f' % (
            subz.snap['i'], zi_max, m_kind, m_lim[0], m_kind, neig_m_lim[0])
        file_name_base = self.directory_prop + file_name_short
        if direction == 'write':
            ut.io.pickle_object(file_name_base, direction, subz.sathistory)
        elif direction == 'read':
            subz.sathistory = {}
            subz.sathistory = ut.io.pickle_object(file_name_base, direction)
        else:
            raise ValueError('not recognize i/o direction = %s' % direction)

    def pickle_history_sat(self, props='some', sub=None, zi_now=1, zi_max=15):
        '''
        Read & assign satellite properties going back to zi_max.

        Import properties to assign history of (some, all), subhalo catalog,
        current & maximum snapshot for history.
        '''
        if props in ('some', 'all'):
            # self.pickle_first_infall_znow('read', sub[zi_now], 'm.star', [9.7, Inf])
            self.pickle_first_infall_znow('read', sub[zi_now], 'm.max', [10.8, Inf])
            self.pickle_prop_extrema_sat('read', sub[zi_now], zi_max, 'm.star', [9.7, Inf],
                                         [8.5, Inf])
            if props == 'all':
                self.pickle_prop('read', sub, zi_now, zi_max, 'dist.cen')
                self.pickle_prop('read', sub, zi_now, zi_max, 'dist.neig-5')
                self.pickle_prop('read', sub, zi_now, zi_max, 'tid.ene.m.bound')
                self.pickle_prop('read', sub, zi_now, zi_max, 'm.star')
        else:
            raise ValueError('not recognize props = %s' % props)

    def pickle_cen_ejected_znow(self, direction='read', eje=None, zi=1, m_kind='m.max',
                                m_lim=[10.8, Inf], dis_mf=0.007):
        '''
        Import pickle direction (read, write), catalog of ejected satellites at snapshot,
        properties for file name.
        Have m_max = [10.8, Inf] to zi_max = 34.
        '''
        file_name_short = 'cen.ejected_zi%d_%s%.1f_mf%.3f' % (zi, m_kind, m_lim[0], dis_mf)
        file_name_base = self.directory_prop + file_name_short
        if direction == 'write':
            ut.io.pickle_object(file_name_base, direction, eje)
        elif direction == 'read':
            eje = ut.io.pickle_object(file_name_base, direction)
            return eje
        else:
            raise ValueError('not recognize i/o direction = %s' % direction)

    # nearest halo properties ----------
    def pickle_halo_near(self, direction='read', subz=None, gm_kind='m.max', gm_lim=[10.8, Inf],
                         dis_mf=0.007, neig_hm_lim=[12, Inf], neig_num_max=100, neig_distmax=10):
        '''
        Pickle properties of nearest (minimum distance / R_200m(neig)) halo.

        Import pickle direction (read, write), catalog of subhalo at snapshot,
        galaxy mass kind & range, disruption mass fraction,
        neighbor halo mass range & maximum number & maximum distance.
        '''
        prop_base_name = 'nearest.halo'

        file_name_base = 'nearest.halo_%s%s_mf%.3f_neig_hm%s_num%d_dist%d' % (
            gm_kind, gm_lim[0], dis_mf, neig_hm_lim[0], neig_num_max, neig_distmax)
        file_name = self.directory_prop + file_name_base
        if direction == 'write':
            sis = ut.array.elements(subz[prop_base_name + '.cen.i'], [0, Inf])
            self.say('write %d nearest halos' % sis.size)
            propss = []
            prop_names = []
            for k in subz.keys():
                if prop_base_name in k:
                    prop_names.append(k)
                    propss.append(subz[k][sis])
            ut.io.pickle_object(file_name, direction, [sis, prop_names, propss])
        elif direction == 'read':
            sis, prop_names, propss = ut.io.pickle_object(file_name, direction)
            for pi in xrange(len(prop_names)):
                subz[prop_names[pi]] = np.zeros(subz[gm_kind].size, propss[pi].dtype) - 1
                subz[prop_names[pi]][sis] = propss[pi]
            self.say('read %d nearest halos' % sis.size)
        else:
            raise ValueError('not recognize i/o direction = %s' % direction)

SubhaloProp = SubhaloPropClass()


def fix(sub, zi_min, zi_max, prop):
    for zi in xrange(zi_min, zi_max + 1):
        sis = ut.array.elements(sub[zi][prop], [1e-10, Inf])
        sub[zi][prop][sis] += 3 * log10(sub.Cosmo['hubble'])


#===================================================================================================
# write
#===================================================================================================
def write_subhalo_mock(subz, m_kind='m.star', m_lim=[8.5, Inf], scat=0.15, dis_mf=0.007):
    '''
    Print subhalo catalog with neighbor counts for Jeremy.

    Import catalog of subhalo at snapshot, mass kind & range, m_gal - m_halo scatter,
    disrupttion mass fraction.
    '''
    from . import sham

    directory_name = RESEARCH_DIRECTORY + 'sdss/group_catalog/tinker/subhalo/'

    sham.assign(subz, m_kind, scat, dis_mf)
    ms = subz[m_kind] + 2 * log10(subz.Cosmo['hubble'])
    sis = ut.array.elements(ms, m_lim)
    poss = subz['pos'][sis]
    # convert to {km/s physical}
    vels = subz['vel'][sis] * ut.coord.velocity_unit(subz.snap['a'], subz.Cosmo['hubble'], 'km/s')

    file_name_base = 'subhalo_L%.0f_zi%d_%s%.1f_sc%s_mf%s.txt' % (
        subz.info['box.length'], subz.snap['i'], m_kind, m_lim[0], str(scat).split('.')[1],
        str(dis_mf).split('.')[1])
    file_name = directory_name + file_name_base
    file_out = open(file_name, 'w')
    print >> file_out, '# box length = %d Mpc' % subz.info['box.length']
    print >> file_out, '# snapshot = %d' % subz.snap['i']
    print >> file_out, '# redshift = %.3f' % subz.snap['z']
    print >> file_out, '# %s = [%.2f, %.2f]' % (m_kind, m_lim[0], ms.max())
    print >> file_out, '# scatter in log M_star - log M_max = %.2f dex' % scat
    print >> file_out, (
        '# position(x,y,z){Mpc comoving} velocity(x,y,z){km/s physical} logM_star{M_sun / h^2} ' +
        'logM_max{M_sun} id_subhalo')
    for sii, si in enumerate(sis):
        print >> file_out, (
            '%.4f %.4f %.4f %.2f %.2f %.2f %.3f %.3f %d') % (
            poss[sii, 0], poss[sii, 1], poss[sii, 2], vels[sii, 0], vels[sii, 1], vels[sii, 2],
            ms[si], subz['m.max'][si], si)
    file_out.close()


def write_subhalo_mock_alis(subz, halz, m_kind='m.max', m_lim=[11, Inf], dis_mf=0.007):
    '''
    .
    '''
    from . import sham

    file_name_base = 'catalog_subhalo_%s%.1f.txt' % (m_kind, m_lim[0])
    file_name = RESEARCH_DIRECTORY + 'sdss/group_catalog/tinker/galaxy-mock_share/' + file_name_base
    file_out = open(file_name, 'w')
    sis = ut.catalog.indices_subhalo(subz, m_kind, m_lim, [1, Inf], dis_mf=dis_mf)
    scats = [0, 0.15, 0.2]
    m_stars = []
    for scat in scats:
        sham.assign(subz, 'm.star', scat, dis_mf)
        m_stars.append(subz['m.star'][sis])
    print >> file_out, ('# masses in log{M/M_sun}, using h = %.2f' % subz.Cosmo['hubble'])
    print >> file_out, '# positions in {Mpc comoving}, velocities in {Mpc/Gyr comoving}'
    print >> file_out, '# number of subhalos = %d' % sis.size
    print >> file_out, '# log(M_max / M_sun) = [%.2f, %.2f]' % (m_lim[0], subz[m_kind][sis].max())
    print >> file_out, ('# id-of-subhalo id-of-central id-of-halo M_max am-sat M_halo C_200c ' +
                        'position(x,y,z) velocity(x,y,z) M_star(0) M_star(0.15) M_star(0.2)')
    for sii, si in enumerate(sis):
        if subz['ilk'][si] <= 0:
            am_sat = 1
        else:
            am_sat = 0
        hi = subz['halo.i'][si]
        print >> file_out, (
            '%d %d %d %.3f %d %.3f %.2f %.3f %.3f %.3f %.5f %.5f %.5f %.3f %.3f %.3f') % (
            si, subz['cen.i'][si], subz['halo.i'][si], subz[m_kind][si], am_sat,
            subz['halo.m'][si] + ut.catalog.MRAT_200M_B168, halz['c.200c'][hi],
            subz['pos'][si, 0], subz['pos'][si, 1], subz['pos'][si, 2],
            subz['vel'][si, 0], subz['vel'][si, 1], subz['vel'][si, 2],
            m_stars[0][sii], m_stars[1][sii], m_stars[2][sii]
        )
    file_out.close()


def write_subhalo_mgrow_cen(sub, zi_now=1, zi_max=22, m_kind='m.max', m_lim=[11, Inf],
                            dis_mf=0.007):
    '''
    Print mass growth information for Jeremy/Alis.

    Import subhalo catalog, snapshot range, mass kind & range, disrupt m fraction.
    '''
    from treepm import subhalo
    form_time_kind = 't'
    form_prop_kind = 'm.max'
    sis = ut.catalog.indices_subhalo(sub[zi_now], m_kind, m_lim, ilk='cen', dis_mf=dis_mf)
    form_times_50 = subhalo.Formation.get_times(sub, zi_now, zi_max, sis, form_time_kind,
                                                form_prop_kind, 0.5)
    form_times_85 = subhalo.Formation.get_times(sub, zi_now, zi_max, sis, form_time_kind,
                                                form_prop_kind, 0.85)
    zis = np.array([3, 5, 8, 10, 12, 14, 16, 18, 20, 22])
    file_name_base = 'subhalo_central_%s-grow_L%.0f_zi%d_%s%.1f.txt' % (
        m_kind, sub.info['box.length'], zi_now, m_kind, m_lim[0])
    file_out = open(file_name_base, 'w')
    print >> file_out, '# box length = %.1f Mpc' % sub.info['box.length']
    print >> file_out, '# snapshot = %d' % zi_now
    print >> file_out, '# redshift = %.2f' % sub.snap['z'][zi_now]
    print >> file_out, '# age = %.3f Gyr' % sub.snap['t'][zi_now]
    print >> file_out, '# %s range = [%.2f, %.2f]' % (m_kind, m_lim[0],
                                                      sub[zi_now][m_kind][sis].max())
    print >> file_out, (
        '# id_subhalo t(M=0.5) t(M=0.85) M(z=%.2f) M(z=%.2f) M(z=%.2f) M(z=%.2f) ' +
        'M(z=%.2f) M(z=%.2f) M(z=%.2f) M(z=%.2f) M(z=%.2f) M(z=%.2f) M(z=%.2f)') % (
        sub.snap['z'][zi_now], sub.snap['z'][zis[0]], sub.snap['z'][zis[1]],
        sub.snap['z'][zis[2]], sub.snap['z'][zis[3]], sub.snap['z'][zis[4]],
        sub.snap['z'][zis[5]], sub.snap['z'][zis[6]], sub.snap['z'][zis[7]],
        sub.snap['z'][zis[8]], sub.snap['z'][zis[9]])
    for sii, si in enumerate(sis):
        ms = np.zeros(zis.size, float32)
        par_zi, par_si = zi_now + 1, sub[zi_now]['par.i'][si]
        while 0 < par_zi <= zis.max() and par_si > 0:
            for zii in xrange(zis.size):
                if par_zi == zis[zii]:
                    ms[zii] = sub[par_zi]['m.max'][par_si]
            par_zi, par_si = par_zi + 1, sub[par_zi]['par.i'][par_si]
        print >> file_out, (
            '%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f') % (
            si, form_times_50[sii], form_times_85[sii], sub[zi_now]['m.max'][si],
            ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6], ms[7], ms[8], ms[9])
    file_out.close()


def write_pos_binary(catz, cis, file_name_base, prop=None):
    '''
    Write positions [in units of box length] to binary file.

    Import catalog of [sub]halo at snapshot, indices, file name.
    '''
    file_name_base += '.dat'
    file_out = open(file_name_base, 'wb')
    np.array(len(cis), int32).tofile(file_out)
    np.array(catz['pos'][cis][:, 0] / catz.info['box.length'], float32).tofile(file_out)
    np.array(catz['pos'][cis][:, 1] / catz.info['box.length'], float32).tofile(file_out)
    np.array(catz['pos'][cis][:, 2] / catz.info['box.length'], float32).tofile(file_out)
    if prop:
        np.array(catz[prop][cis], float32).tofile(file_out)
    file_out.close()
    print '# printed %d objects in %s' % (cis.size, file_name_base)
