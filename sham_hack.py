'''
Assign stellar mass/magnitude to subhalos via abundance matching.

Masses in log {M_sun}, luminosities in log {L_sun / h^2}, distances in {Mpc comoving}.
'''

# system -----
#from __future__ import division
import numpy as np
from numpy import log10, Inf
from scipy import integrate, interpolate, ndimage
# local -----
#from visualize import plot_sm
from utilities import utility as ut


def assign(sub, m_kind='m.star', scat=0, dis_mf=0.007, source='', sham_prop='m.max', zis=None):
    '''
    Assign Mag_r or M_star via abundance matching.

    Import catalog of subhalo [at snapshot], mass kind (mag.r, m.star),
    1-sigma mass scatter at fixed sham prop [dex], disruption mass fraction (for both cens & sats),
    mass source, property to abundance match against, [snapshot index[s]].
    '''
    if isinstance(sub, list):
        if zis is None:
            raise ValueError('subhalo catalog is a tree list, but no input snapshot index[s]')
    elif isinstance(sub, dict):
        if zis is not None:
            raise ValueError('input snapshot index[s], but input catalog of subhalo at snapshot')
        sub = [sub]
        zis = [0]
    subz = sub[zis[0]]
    vol = subz.info['box.length'] ** 3
    print 'Box Length', subz.info['box.length']
    print 'Box Hubble', subz.Cosmo['hubble']
    zis = ut.array.arrayize(zis)
    if m_kind == 'm.star':
        if not source:
            source = 'li-drory-march'
        redshift = subz.snap['z']
        if redshift < 0.1:
            redshift = 0.1
        MF = SMFClass(source, redshift, scat, subz.Cosmo['hubble'])
    elif m_kind == 'mag.r':
        if source == 'cool_ages':
            redshift = subz.snap['z']
            if redshift < 0.1:
                redshift = 0.1
            MF = LFClass(source, scat, subz.Cosmo['hubble'], redshift)
        else:
            if not source:
                source = 'blanton'
            MF = LFClass(source, scat, subz.Cosmo['hubble'])
    else:
        raise ValueError('not recognize m_kind = %s' % m_kind)
    for zi in zis:
        subz = sub[zi]
        subz[m_kind] = np.zeros(subz[sham_prop].size, np.float32)
        if m_kind == 'm.star':
            z = subz.snap['z']
            if z < 0.1:
                z = 0.1
            MF.initialize_redshift(z)
        elif m_kind == 'mag.r':
            if source == 'cool_ages':
                z = subz.snap['z']
                if z < 0.1:
                    z = 0.1
                MF.initialize_redshift(z)
        # maximum number of objects in volume to assign given SMF/LF threshold
        num_max = int(round(MF.numden(MF.mmin) * vol))
        sis = ut.array.elements(subz[sham_prop], [0.001, Inf])
        if dis_mf:
            sis = ut.array.elements(subz['m.frac.min'], [dis_mf, Inf], sis)
        siis_sort = np.argsort(subz[sham_prop][sis]).astype(sis.dtype)[::-1][:num_max]
        num_sums = ut.array.arange_length(num_max) + 1
        if scat:
            if m_kind == 'm.star': 
                scats = np.random.normal(np.zeros(num_max), MF.scat).astype(np.float32)
            elif m_kind == 'mag.r': 
                scats = np.random.normal(np.zeros(num_max), 2.5 * MF.scat).astype(np.float32)
            #print MF.m_scat(num_sums / vol) + scats
            subz[m_kind][sis[siis_sort]] = MF.m_scat(num_sums / vol) + scats
        else:
            subz[m_kind][sis[siis_sort]] = MF.m(num_sums / vol)


class SMFClass:
    '''
    Relate number density [dnumden / dlog(M_star/M_sun)] <-> stellar mass [log10(M_star/M_sun)]
    using fits to observed stellar mass functions.
    All SMFs assume input Hubble constant.
    '''
    def __init__(self, source='li-march', redshift=0.1, scat=0, hubble=0.7):
        '''
        Import SMF source, redshift, log scatter in M_star at fixed Msub.
        '''
        self.source = source
        self.scat = scat
        self.hubble = hubble
        if source == 'li':
            '''
            Li & White 2009. z = 0.1 from SDSS. Chabrier IMF. Complete to 1e8 M_sun/h^2.
            '''
            self.redshifts = np.array([0.1])
            self.mchars = np.array([10.525]) - 2 * log10(hubble)    # {M_sun}
            self.amplitudes = np.array([0.0083]) * hubble ** 3    # {Mpc ^ -3 / log(M/M_sun)}
            self.slopes = np.array([-1.155])
            self.initialize_redshift(redshift)
        elif source == 'baldry':
            '''
            Baldry et al 2008. z = 0.1 from SDSS. diet Salpeter IMF = 0.7 Salpeter.
            Complete to 1e8 M_sun.
            '''
            h_them = 0.7    # their assumed hubble constant
            self.redshifts = np.array([0.1])
            # covert to Chabrier
            self.mchars = (np.array([10.525]) + 2 * log10(h_them / hubble) + log10(1 / 1.6 / 0.7))
            self.amplitudes = np.array([0.00426]) * (hubble / h_them) ** 3
            self.amplitudes2 = np.array([0.00058]) * (hubble / h_them) ** 3
            self.slopes = np.array([-0.46])
            self.slopes2 = np.array([-1.58])
            self.initialize_redshift(redshift)
        elif source == 'cole-march':
            '''
            Marchesini et al 2009. 1.3 < z < 4.0. Kroupa IMF.
            z = 0.1 from Cole et al 2001 (2dF), converting their Salpeter to Kroupa.
            *** In order to use out to z ~ 4, made evolution flat from z = 3.5 to 4.
            '''
            self.redshifts = np.array([0.1, 1.6, 2.5, 3.56, 4.03])
            self.mchars = np.array([10.65, 10.60, 10.65, 11.07, 11.07]) - 2 * log10(hubble)
            # converted to {Mpc ^ -3 dex ^ -1}
            self.amplitudes = np.array([90.00, 29.65, 11.52, 1.55, 1.55]) * 1e-4 * hubble ** 3
            self.slopes = np.array([-1.18, -1.00, -1.01, -1.39, -1.39])
            self.make_splines()
            self.initialize_redshift(redshift)
        elif source == 'li-march':
            '''
            Marchesini et al 2009, using Li & White at z = 0.1.
            '''
            self.redshifts = np.array([0.1, 1.6, 2.5, 3.56, 4.03])
            self.mchars = np.array([10.525, 10.60, 10.65, 11.07, 11.07]) - 2 * log10(hubble)
            self.amplitudes = (np.array([0.0083, 0.002965, 0.00115, 0.000155, 0.000155]) *
                               hubble ** 3)
            self.slopes = np.array([-1.155, -1.00, -1.01, -1.39, -1.39])
            self.make_splines()
            self.initialize_redshift(redshift)
        elif source == 'muzzin': 
            ''' Muzzin et al. (2013)
            '''
            self.redshifts = np.array([0.35, 0.75, 1.25, 1.75]) 
            self.mchars = np.array([11.22, 11., 10.87, 10.81]) #- 2 * log10(hubble)
            self.amplitudes = (np.array([12.16e-4, 16.25e-4, 13.91e-4, 10.13e-4]))# * hubble ** 3)
            self.slopes = np.array([-1.29, -1.17, -1.02, -0.86])
            self.make_splines()
            self.initialize_redshift(redshift)

        elif source == 'li-march-extreme': 
            '''
            More extreme version of Marchesini et al 2009, using Li & White at z = 0.1.
            '''
            self.redshifts = np.array([0.1, 1.6, 2.5, 3.56, 4.03])
            self.mchars = np.array([10.525, 10.60, 10.65, 11.07, 11.07]) - 2 * log10(hubble)
            self.amplitudes = (np.array([0.0083, 0.00001, 0.00001, 0.00001, 0.000001]) *
                               hubble ** 3)
            self.slopes = np.array([-1.155, -1.00, -1.01, -1.39, -1.39])
            self.make_splines()
            self.initialize_redshift(redshift)
        elif source == 'constant-li': 
            '''
            Li & White at all redshifts 
            '''
            self.redshifts = np.arange(0.1, 4.03, 0.1) 
            self.mchars = np.repeat(10.525, len(self.redshifts)) - 2 * log10(hubble)
            self.amplitudes = (np.repeat(0.0083, len(self.redshifts))* hubble ** 3)
            self.slopes = np.repeat(-1.155, len(self.redshifts))
            self.make_splines()
            self.initialize_redshift(redshift)

        elif source == 'fontana':
            '''
            Fontana et al 2006. 0.4 < z < 4 from GOODS-MUSIC. Salpeter IMF.
            z = 0.1 from Cole et al 2001.
            '''
            h_them = 0.7    # their assumed hubble constant
            self.redshifts = np.array([0.1, 4.0])    # store redshift range of validity
            self.amplitude0 = 0.0035 * (hubble / h_them) ** 3    # to {Mpc ^ -3 / log10(M/M_sun)}
            self.amplitude1 = -2.2
            self.slope0 = -1.18
            self.slope1 = -0.082
            self.mchar0 = 11.16    # log10(M/M_sun)
            self.mchar1 = 0.17    # log10(M/M_sun)
            self.mchar2 = -0.07    # log10(M/M_sun)
            # convert to my hubble & Chabrier IMF
            self.mchar0 += 2 * log10(h_them / hubble) - log10(1.6)
            self.initialize_redshift(redshift)
        elif source == 'li-drory-march':
            '''
            Drory et al 2009. 0.3 < z < 1.0 from COSMOS.
            Chabrier IMF limited to 0.1 - 100 M_sun.
            Complete to (8.0, 8.6, 8.9, 9.1) M_sun/h^2 at z = (0.3, 0.5, 0.7, 0.9).
            Anchor to Li & White at z = 0.1, Marchesini et al at higher redshift.
            See Ilbert et al 2010 for alternate COSMOS version.
            '''
            h_them = 0.72    # their assumed hubble constant
            self.redshifts = np.array([0.3, 0.5, 0.7, 0.9])
            self.mchars = np.array([10.90, 10.91, 10.95, 10.92]) + 2 * log10(h_them / hubble)
            # convert to [Mpc ^ -3 dex^-1]
            self.amplitudes = (np.array([0.00289, 0.00174, 0.00216, 0.00294]) *
                               (hubble / h_them) ** 3)
            self.slopes = np.array([-1.06, -1.05, -0.93, -0.91])
            self.mchars2 = np.array([9.63, 9.70, 9.75, 9.85]) + 2 * log10(h_them / hubble)
            self.amplitudes2 = (np.array([0.00180, 0.00143, 0.00289, 0.00212]) *
                                (hubble / h_them) ** 3)
            self.slopes2 = np.array([-1.73, -1.76, -1.65, -1.65])
            # add li & white
            self.redshifts = np.append(0.1, self.redshifts)
            self.mchars = np.append(10.525 - 2 * log10(hubble), self.mchars)
            self.amplitudes = np.append(0.0083 * hubble ** 3, self.amplitudes)
            self.slopes = np.append(-1.155, self.slopes)
            self.mchars2 = np.append(self.mchars2[0], self.mchars2)
            self.amplitudes2 = np.append(0, self.amplitudes2)
            self.slopes2 = np.append(self.slopes2[0], self.slopes2)
            # add marchesini et al
            h_them = 0.7    # their assumed hubble constant
            self.redshifts = np.append(self.redshifts, [1.6, 2.5, 3.56, 4.03])
            self.mchars = np.append(self.mchars,
                np.array([10.60, 10.65, 11.07, 11.07]) - 2 * log10(hubble))
            self.amplitudes = np.append(self.amplitudes,
                                        np.array([0.002965, 0.00115, 0.000155, 0.000155]) *
                                        hubble ** 3)
            self.slopes = np.append(self.slopes, [-1.00, -1.01, -1.39, -1.39])
            self.mchars2 = np.append(self.mchars2, np.zeros(4) + self.mchars2[0])
            self.amplitudes2 = np.append(self.amplitudes2, np.zeros(4))
            self.slopes2 = np.append(self.slopes2, np.zeros(4) + self.slopes2[0])
            self.make_splines()
            self.initialize_redshift(redshift)
        elif source == 'li-drory-march_sameslope':
            '''
            Apply low-mass slope from Drory et al 2009 to Li & White, Marchesini et al.
            '''
            self.redshifts = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.6, 2.5, 3.56, 4.03])
            self.mchars = np.array([10.525, 10.61, 10.62, 10.66, 10.63, 10.60, 10.65, 11.07,
                                    11.07] - 2 * log10(hubble))
            self.amplitudes = np.array([0.0083, 0.00774, 0.00466, 0.00579, 0.00787, 0.00297,
                                        0.00115, 0.000155, 0.000155]) * hubble ** 3
            self.slopes = np.array([-1.155, -1.06, -1.05, -0.93, -0.91, -1.00, -1.01, -1.39, -1.39])
            self.mchars2 = (np.array([9.35, 9.34, 9.41, 9.46, 9.56, 9.41, 9.46, 9.83, 9.83]) -
                            2 * log10(hubble))
            self.amplitudes2 = np.array([0.00269, 0.00482, 0.00383, 0.00774, 0.00568, 0.000962,
                                         0.000375, 0.0000503, 0.0000503]) * hubble ** 3
            self.slopes2 = np.array([-1.70, -1.73, -1.76, -1.65, -1.65, -1.72, -1.74, -2.39, -2.39])
            self.make_splines()
            self.initialize_redshift(redshift)
        elif source == 'perez':
            '''
            Perez-Gonzalez et al 2008. 0.1 < z < 4.0 from Spitzer, Hubble, Chandra.
            Salpeter IMF.
            Complete to (8, 9.5, 10, 11) M_star at z = (0, 1, 2, 3).
            '''
            h_them = 0.7    # their assumed hubble constant
            self.redshifts = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.15, 1.45, 1.8, 2.25, 2.75, 3.25,
                                       3.75])
            self.mchars = np.array([11.16, 11.20, 11.26, 11.25, 11.27, 11.31, 11.34, 11.40, 11.46,
                                    11.34, 11.33, 11.36]) + 2 * log10(h_them / hubble)
            # convert to Chabrier IMF
            self.mchars -= log10(1.6)
            # convert to [Mpc ^ -3 dex ^ -1]
            self.amplitudes = (10 ** np.array([-2.47, -2.65, -2.76, -2.82, -2.91, -3.06, -3.27,
                                              - 3.49, -3.69, -3.64, -3.74, -3.94]) *
                               (hubble / h_them) ** 3)
            self.slopes = np.array([-1.18, -1.19, -1.22, -1.26, -1.23, -1.26, -1.29, -1.27, -1.26,
                                    - 1.20, -1.14, -1.23])
            self.make_splines()
            self.initialize_redshift(redshift)
        else:
            raise ValueError('not recognize source = %s' % source)

    def make_splines(self):
        '''
        Make spline fits to SMF fit parameters v redshift.
        Use 1st order spline (k) to avoid ringing.
        '''
        self.mchar_z_spl = interpolate.splrep(self.redshifts, self.mchars, k=1)
        self.slope_z_spl = interpolate.splrep(self.redshifts, self.slopes, k=1)
        self.amplitude_z_spl = interpolate.splrep(self.redshifts, self.amplitudes, k=1)
        if self.source in ('li-drory-march', 'li-drory-march_sameslope'):
            self.mchar2_z_spl = interpolate.splrep(self.redshifts, self.mchars2, k=1)
            self.slope2_z_spl = interpolate.splrep(self.redshifts, self.slopes2, k=1)
            self.amplitude2_z_spl = interpolate.splrep(self.redshifts, self.amplitudes2, k=1)

    def initialize_redshift(self, redshift=0.1):
        '''
        Make spline to get mass from number density.

        Import redshift.
        Find SMF fit parameters at redshift, correcting amplitude by * log(10) & slope
        by + 1 to make dndm call faster.
        '''
        if redshift < self.redshifts.min() - 1e-5 or redshift > self.redshifts.max() + 1e-5:
            raise ValueError('z = %.2f out of range for %s' % (redshift, self.source))
        self.redshift = redshift
        if self.source in ('li'):
            self.m_char = self.mchars[0]
            self.amplitude = self.amplitudes[0] * np.log(10)
            self.slope = self.slopes[0] + 1
        elif self.source in ('baldry'):
            self.m_char = self.mchars[0]
            self.mchar2 = self.mchars[0]
            self.amplitude = self.amplitudes[0] * np.log(10)
            self.amplitude2 = self.amplitudes2[0] * np.log(10)
            self.slope = self.slopes[0] + 1
            self.slope2 = self.slopes2[0] + 1
        elif self.source in ('cole-march', 'li-march', 'perez', 'constant-li', 'li-march-extreme', 'muzzin'):
            self.m_char = interpolate.splev(redshift, self.mchar_z_spl)
            self.amplitude = interpolate.splev(redshift, self.amplitude_z_spl) * np.log(10)
            self.slope = interpolate.splev(redshift, self.slope_z_spl) + 1
        elif self.source == 'fontana':
            self.m_char = self.mchar0 + self.mchar1 * redshift + self.mchar2 * redshift ** 2
            self.amplitude = (self.amplitude0 * (1 + redshift) ** self.amplitude1) * np.log(10)
            self.slope = (self.slope0 + self.slope1 * redshift) + 1
        elif self.source in ('li-drory-march', 'li-drory-march_sameslope'):
            self.m_char = interpolate.splev(redshift, self.mchar_z_spl)
            self.amplitude = interpolate.splev(redshift, self.amplitude_z_spl) * np.log(10)
            self.slope = interpolate.splev(redshift, self.slope_z_spl) + 1
            self.mchar2 = interpolate.splev(redshift, self.mchar2_z_spl)
            self.amplitude2 = interpolate.splev(redshift, self.amplitude2_z_spl) * np.log(10)
            self.slope2 = interpolate.splev(redshift, self.slope2_z_spl) + 1
        self.make_numden_m_spline(self.redshift, self.scat)

    def dndm(self, m_star):
        '''
        Compute d(num-den) / d(log m) = ln(10) * amplitude * (10^(m_star - m_char)) ** (1 + slope) *
        exp(-10^(m_star - m_char)).

        Import stellar mass.
        '''
        m_rats = 10 ** (m_star - self.m_char)
        if 'drory' in self.source or self.source == 'baldry':
            dm2s = 10 ** (m_star - self.mchar2)
            return (self.amplitude * m_rats ** self.slope * np.exp(-m_rats) +
                    self.amplitude2 * dm2s ** self.slope2 * np.exp(-dm2s))
        else:
            return self.amplitude * m_rats ** self.slope * np.exp(-m_rats)

    def numden(self, m_min, m_max=14):
        '''
        Compute number density within range.

        Import stellar mass range.
        '''
        return integrate.quad(self.dndm, m_min, m_max)[0]

    def make_numden_m_spline(self, redshift=0.1, scat=0):
        '''
        Make splines to relate d(num-den) / d[log]m & num-den(> m) to m.

        Import redshift (if want to change), mass scatter [dex].
        '''
        iter_num = 30

        if redshift != self.redshift:
            self.initialize_redshift(redshift)
        if scat != self.scat:
            self.scat = scat
        dm = 0.01
        dm_scat_lo = 3 * scat    # extend fit for deconvolute b.c.'s
        dm_scat_hi = 0.5 * scat    # extend fit for deconvolute b.c.'s
        self.mmin = 7.3
        self.mmax = 12.3
        m_stars = np.arange(self.mmin - dm_scat_lo, self.mmax + dm_scat_hi, dm, np.float32)
        numdens = np.zeros(m_stars.size)
        dndms = np.zeros(m_stars.size)
        for mi in xrange(m_stars.size):
            # make sure numdens are monotonically decreasing even if = -infinity
            numdens[mi] = self.numden(m_stars[mi]) + 1e-9 * (1 - mi * 0.001)
            dndms[mi] = self.dndm(m_stars[mi]) + 1e-9 * (1 - mi * 0.001)
        # make no scatter splines
        self.log_numden_m_spl = interpolate.splrep(m_stars, log10(numdens))
        self.m_log_numden_spl = interpolate.splrep(log10(numdens)[::-1], m_stars[::-1])
        # at high z, smf not monotonically decreasing, so spline not work on below
        # self.m_log_dndm_spl = interpolate.splrep(log10(dndms)[::-1], m_stars[::-1])
        # make scatter splines
        if scat:
            # deconvolve osbserved smf assuming scatter to find unscattered one
            dndms_scat = ut.math.deconvolute(dndms, scat, dm, iter_num)
            # chop off lower boundaries, unreliable
            m_stars = m_stars[dm_scat_lo / dm:]
            dndms_scat = dndms_scat[dm_scat_lo / dm:]
            # find spline to integrate over
            self.dndm_m_scat_spl = interpolate.splrep(m_stars, dndms_scat)
            numdens_scat = np.zeros(m_stars.size)
            for mi in xrange(m_stars.size):
                numdens_scat[mi] = interpolate.splint(m_stars[mi], m_stars.max(),
                                                      self.dndm_m_scat_spl)
                numdens_scat[mi] += 1e-9 * (1 - mi * 0.001)
            self.log_numden_m_scat_spl = interpolate.splrep(m_stars, log10(numdens_scat))
            self.m_log_numden_scat_spl = interpolate.splrep(log10(numdens_scat)[::-1],
                                                            m_stars[::-1])

    def m(self, num_den):
        '''
        Get mass at threshold.

        Import threshold number density.
        '''
        return interpolate.splev(log10(num_den), self.m_log_numden_spl).astype(np.float32)

    def m_scat(self, num_den):
        '''
        Get mass at threshold, using de-scattered source.

        Import threshold number density.
        '''
        return interpolate.splev(log10(num_den), self.m_log_numden_scat_spl).astype(np.float32)

    def m_dndm(self, dn_dm):
        '''
        Get mass at d(num-den)/d[log]m.

        Import d(num-den) / d[log]m.
        '''
        return interpolate.splev(log10(dn_dm), self.m_log_dndm_spl)

    def dndm_scat(self, m):
        '''
        Get d(num-den) / d[log]m at m, using de-scattered source.

        Import mass.
        '''
        return interpolate.splev(m, self.dndm_m_scat_spl)

    def numden_scat(self, m):
        '''
        Get num-den(>[log]m) at m, using de-scattered source.

        Import mass.
        '''
        return 10 ** (interpolate.splev(m, self.log_numden_m_scat_spl))


class LFClass(SMFClass):
    '''
    Relate number density [Mpc ^ -3] <-> magnitude/luminosity using spline fit to luminosity
    functions.

    Import spline querying functions from SMFClass.
    '''
    def __init__(self, source='blanton', scat=0, hubble=0.7, redshift=0.1):
        '''
        Import source, log-normal scatter.
        '''
        self.source = source
        self.scat = scat
        self.hubble = hubble
        if source == 'norberg':
            # Norberg et al 2002: 2dF r-band at z ~ 0.1.
            self.m_char = -19.66
            self.amplitude = 1.61e-2 * hubble ** 3    # Mpc ^ -3
            self.slope = -1.21
        elif source == 'blanton':
            # Blanton et al 03: SDSS r-band z ~ 0.1.
            self.m_char = -20.44
            self.amplitude = 1.49e-2 * hubble ** 3    # Mpc ^ -3
            self.slope = -1.05
        elif source == 'sheldon':
            # Sheldon et al 07: SDSS i-band z = 0.25. Valid for Mag < -19.08 (0.19L*).
            self.m_char = -20.9    # Hansen et al 09 catalog has -20.8
            self.amplitude = 1.02e-2 * hubble ** 3    # Mpc ^ -3
            self.slope = -1.21
        elif source == 'cool_ages': 
            # Cool et al 2012: AGES.
            self.redshifts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.65])
            self.mchars = np.array([-20.58, -20.81, -20.81, -20.99, -21.29, -21.38]) 
            self.amplitudes = (np.array([1.59e-2, 1.52e-2, 1.24e-2, 1.44e-2, 1.08e-2, 1.05e-2]) * hubble ** 3)    # Mpc ^ -3
            self.slopes = np.repeat(-1.05, len(self.redshifts)) 
            self.make_splines()
            self.initialize_redshift(redshift)
        else:
            raise ValueError('not recognize source = %s in LFClass' % source)

        if source != 'cool_ages': 
            self.make_numden_m_spline(scat, redshift=None)

    def dndm(self, mag):
        '''
        Get d(num-den) / d(mag).

        Import (positive) magnitude.
        '''
        mag *= -1.
        return (np.log(10) / 2.5 * self.amplitude *
                10 ** ((self.slope + 1) / 2.5 * (self.m_char - mag)) *
                np.exp(-10 ** ((self.m_char - mag) / 2.5)))

    def numden(self, m_min, m_max=25):
        '''
        Get number density within range.

        Import (positive) magnitude range.
        '''
        return integrate.quad(self.dndm, m_min, m_max)[0]
    
    def initialize_redshift(self, redshift=0.1):
        '''
        Make spline to get mass from number density.

        Import redshift.
        Find SMF fit parameters at redshift, correcting amplitude by * log(10) & slope
        by + 1 to make dndm call faster.
        '''
        if redshift < self.redshifts.min() - 1e-5:# or redshift > self.redshifts.max() + 1e-5:
            raise ValueError('z = %.2f out of range for %s' % (redshift, self.source))
        self.redshift = redshift
        self.m_char = interpolate.splev(redshift, self.mchar_z_spl, ext=0)
        self.amplitude = interpolate.splev(redshift, self.amplitude_z_spl, ext=0) 
        self.slope = interpolate.splev(redshift, self.slope_z_spl, ext=0)
        self.make_numden_m_spline(scat = self.scat, redshift = self.redshift)

    def make_numden_m_spline(self, scat=0, redshift=0.1):
        '''
        Make splines to relate d(num-den)/d(mag) & num-den(> mag) to mag.

        Import scatter [dex].
        '''
        try: 
            if redshift != self.redshift:
                self.initialize_redshift(redshift)
        except AttributeError:
            pass 
        if scat != self.scat:
            self.scat = scat    # convert scatter in log(lum) to scatter in magnitude
        mag_scat = 2.5 * self.scat
        deconvol_iter_num = 20
        dmag = 0.01
        dmag_scat_lo = 2 * mag_scat    # extend fit for b.c.'s of deconvolute
        dmag_scat_hi = 1 * mag_scat
        self.mmin = 17.0
        self.mmax = 23.3
        mags = np.arange(self.mmin - dmag_scat_lo, self.mmax + dmag_scat_hi, dmag, np.float32)
        numdens = np.zeros(mags.size)
        dndms = np.zeros(mags.size)
        for mi in xrange(len(mags)):
            numdens[mi] = np.abs(self.numden(mags[mi]))
            dndms[mi] = self.dndm(mags[mi])
        #print 'numden ', numdens[:10]
        #print mags[:10]
        # make no scatter splines
        self.log_numden_m_spl = interpolate.splrep(mags, log10(numdens))
        self.dndm_m_spl = interpolate.splrep(mags, dndms)
        self.m_log_numden_spl = interpolate.splrep(log10(numdens)[::-1], mags[::-1])
        # make scatter splines
        if self.scat:
            # deconvolve observed lf assuming scatter to find unscattered one
            dndms_scat = ut.math.deconvolute(dndms, mag_scat, dmag, deconvol_iter_num)
            # chop off boundaries, unreliable
            mags = mags[dmag_scat_lo / dmag:-dmag_scat_hi / dmag]
            dndms_scat = dndms_scat[dmag_scat_lo / dmag:-dmag_scat_hi / dmag]
            # find spline to integrate over
            self.dndm_m_scat_spl = interpolate.splrep(mags, dndms_scat)
            numdens_scat = np.zeros(mags.size)
            for mi in xrange(mags.size):
                numdens_scat[mi] = np.abs(interpolate.splint(mags[mi], mags.max(), self.dndm_m_scat_spl))
                numdens_scat[mi] += 1e-9 * (1 - mi * 0.001)
            self.log_numden_m_scat_spl = interpolate.splrep(mags, log10(numdens_scat))
            self.m_log_numden_scat_spl = interpolate.splrep(log10(numdens_scat)[::-1], mags[::-1])
    






#===================================================================================================
# test/plot
#===================================================================================================
def test_sham(sub, zi, m_kind, m_min, m_max, scat=0.2, mfracmin=0, m_wid=0.1, source='',
              sham_kind='m.max'):
    '''
    Plot mass functions.

    Import subhalo catalog, snapshot index,
    mass kind (m.star, mag.r) & range & scatter at fixed m_max,
    disruption mass fraction, bin size, GMF source, subhalo property to assign against.
    '''
    m_wid_scat = 3 * scat
    m_bins = np.arange(m_min - m_wid_scat, m_max + m_wid_scat, m_wid, np.float32) + 0.5 * m_wid
    if m_kind == 'm.star':
        if not source:
            source = 'li-march'
        Sf = SMFClass(source, sub.snap['z'][zi], scat, sub.Cosmo['hubble'])
    elif m_kind == 'mag.r':
        if not source:
            source = 'blanton'
        Sf = LFClass(source, scat, sub.Cosmo['hubble'])
    # analytic gmf, no scatter
    dndm_anal = Sf.dndm(m_bins)
    if scat:
        # convolve above gmf with scatter, then deconvolve, to see if can recover
        dndm_anal_conv = ndimage.filters.gaussian_filter1d(dndm_anal, Sf.scat / m_wid)
        dndm_anal_decon = ut.math.deconvolute(dndm_anal_conv, Sf.scat, m_wid, 30)
        # mean (underlying) relation
        dndm_anal_pre = Sf.dndm_scat(m_bins)
        # observed gmf after convolution (no random noise)
        dndm_anal_recov = ndimage.filters.gaussian_filter1d(dndm_anal_pre, Sf.scat / m_wid)
        # cut out extremes, unreliable
        cutoff = int(round(m_wid_scat / m_wid))
        if cutoff > 0:
            m_bins = m_bins[cutoff:-cutoff]
            dndm_anal = dndm_anal[cutoff:-cutoff]
            dndm_anal_conv = dndm_anal_conv[cutoff:-cutoff]
            dndm_anal_pre = dndm_anal_pre[cutoff:-cutoff]
            dndm_anal_decon = dndm_anal_decon[cutoff:-cutoff]
            dndm_anal_recov = dndm_anal_recov[cutoff:-cutoff]
    m_bins -= 0.5 * m_wid
    # assign mass to subhalo, with or without scatter (random noise at high mass end)
    assign(sub, zi, m_kind, scat, mfracmin, source, sham_kind)
    ims = ut.bin.idigitize(sub[zi][m_kind], m_bins)
    gal_nums = np.zeros(m_bins.size)
    for mi in xrange(m_bins.size):
        gal_nums[mi] = ims[ims == mi].size
    print 'bin count min %d' % np.min(gal_nums)
    dndm_sham = gal_nums / sub.info['box.length'] ** 3 / m_wid
    print 'assign ratio ave %.3f' % np.mean(abs(dndm_sham / dndm_anal))
    if scat:
        print 'recov ratio ave %.3f' % np.mean(abs(dndm_anal_recov / dndm_anal))

    # plot ----------
    Plot = plot_sm.PlotClass()
    Plot.set_axis('lin', 'lin', [m_min, m_max], log10(dndm_anal))
    Plot.make_window()
    Plot.draw('c', m_bins, log10(dndm_anal))
    Plot.draw('c', m_bins, log10(dndm_sham), ct='red')
    if scat:
        Plot.draw('c', m_bins, log10(dndm_anal_pre), ct='green')
        Plot.draw('c', m_bins, log10(dndm_anal_recov), ct='blue')


def plot_source_compare(sources=['li-march', 'perez'], redshifts=0.1, m_lim=[8.0, 11.7], m_wid=0.1,
                        plot_kind='value'):
    '''
    Plot each source at each redshift.

    Import mass functions, redshifts, plotting mass range & bin width, plot kind (value, ratio).
    '''
    sources = ut.array.arrayize(sources)
    redshifts = ut.array.arrayize(redshifts)
    Mbin = ut.bin.BinClass(m_lim, m_wid)
    log_dn_dlogms = []
    for src_i in xrange(sources.size):
        log_dn_dlogms_so = []
        for zi in xrange(redshifts.size):
            Smf = SMFClass(sources[src_i], redshifts[zi], scat=0, hubble=0.7)
            log_dn_dlogms_so.append(log10(Smf.dndm(Mbin.mids)))
        log_dn_dlogms.append(log_dn_dlogms_so)

    # plot ----------
    Plot = plot_sm.PlotClass()
    if plot_kind == 'ratio':
        ys = 10 ** (log_dn_dlogms - log_dn_dlogms[0][0])
        Plot.axis.space_y = 'lin'
    elif plot_kind == 'value':
        ys = log_dn_dlogms
        Plot.axis.space_y = 'log'
    Plot.set_axis('log', '', Mbin.mids, ys, tick_lab_kind='log')
    Plot.set_axis_label('m.star', 'dn/dlog(M_{star}) [h^{3}Mpc^{-3}]')
    Plot.make_window()
    Plot.set_label(pos_y=0.4)
    for src_i in xrange(sources.size):
        for zi in xrange(redshifts.size):
            Plot.draw('c', Mbin.mids, log_dn_dlogms[src_i][zi], ct=src_i, lt=zi)
            Plot.make_label(sources[src_i] + ' z=%.1f' % redshifts[zi])
    # add in cosmos at z = 0.35
    '''
    cosmos = [[8.8,    0.015216 , 1.250341e-03],
                        [9,         0.01257,    1.210321e-03],
                        [9.2,         0.01009,    1.047921e-03],
                        [9.4,        0.007941,    8.908445e-04],
                        [9.6,        0.006871,    7.681928e-04],
                        [9.8,        0.005688,    6.825634e-04],
                        [10,        0.005491,    6.136567e-04],
                        [10.2,        0.004989,    6.004422e-04],
                        [10.4,         0.00478,    5.917784e-04],
                        [10.6,         0.00423,    5.851342e-04],
                        [10.8,        0.003651,    4.919025e-04],
                        [11,        0.002253,    3.562664e-04],
                        [11.2,        0.001117,    2.006811e-04],
                        [11.4,     0.0004182,    8.486049e-05],
                        [11.6,     8.365e-05,    2.802892e-05],
                        [11.8,     1.195e-05,    8.770029e-06]]
    '''
    # cosmos at z = 0.87
    cosmos = [[9.8, 0.005377, 3.735001e-04],
                        [10, 0.004206, 3.443666e-04],
                        [10.2, 0.003292, 3.235465e-04],
                        [10.4, 0.003253, 3.318173e-04],
                        [10.6, 0.002985, 3.198681e-04],
                        [10.8, 0.002994, 2.735925e-04],
                        [11, 0.002218, 1.922526e-04],
                        [11.2, 0.001202, 1.067172e-04],
                        [11.4, 0.0005681, 3.983348e-05],
                        [11.6, 0.0001837, 1.195015e-05],
                        [11.8, 4.214e-05, 3.200856e-06],
                        [12, 1.686e-06, 7.463160e-07]]
    cosmos = np.array(cosmos)
    cosmos = cosmos.transpose()
    cosmos[1] = log10(cosmos[1] * 0.72 ** -3)
    # Plot.draw('pp', cosmos[0], cosmos[1], pt=123)
