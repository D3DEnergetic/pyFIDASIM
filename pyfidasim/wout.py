from pathlib import Path
import numpy as np
from scipy.io import netcdf


class readWout:
    """
    A class to read wout netCDF files from VMEC

    ...

    Attributes
    ----------
    path : str
        path to the directory in which the wout woutfile is located
    name : str, optional
        name of the wout woutfile being read (default is 'wout_HSX_main_opt0.nc')
    diffAmps : bool, optional
        read differential flux coordinates and include in fourier amplitudes
        (default is False)
    curvAmps : bool, optional
        read B field differentials necessary to perform B field curvature
        calculations and include in fourier amplitudes (default is False)
    iotaPro : bool, optional
        read iota profile from wout woutfile (default is False)

    Methods
    -------
    transForm_3D(u_dom, v_dom, ampKeys)
        Performs 3D Fourier transform on specified keys

    transForm_2D_sSec(rEff, u_dom, v_dom, ampKeys)
        Performs 2D Fourier transform along one flux surface on specified keys

    transForm_2D_uSec(u, v_dom, ampKeys)
        Performs 2D Fourier transform along a poloidal cross section on
        specified keys

    transForm_2D_vSec(u_dom, v, ampKeys)
        Performs 2D Fourier transform along a toroidal cross section on
        specified keys

    transForm_1D(rEff, u, v, ampKeys)
        Performs 1D Fourier transform at a particular flux coordinate on
        specified keys

    boozerSpectra(ax, ampKey, nSpec=10, excFirst=True)
        Plots highest amplitude Fourier modes on specified keys

    Raises
    ------
    IOError
        wout woutfile specified does not exist
    """

    def __init__(self, path='', name='wout_HSX_main_opt0.nc',
                 diffAmps=False, curvAmps=False, iotaPro=False):

        fullpath = (Path(path) / name).resolve()
        if not fullpath.exists():
            raise FileNotFoundError(f'{fullpath.as_posix()} does not exist')

        f = netcdf.netcdf_file(
            fullpath.as_posix(),
            mode='r',
            version=4,
            mmap=False)

        if iotaPro:
            self.iota = f.variables['iotaf'].data

        self.nfp = f.variables['nfp'].data
        self.mpol = f.variables['mpol'].data
        self.ntor = f.variables['ntor'].data

        self.ns = f.variables['ns'].data
        self.s_dom = np.linspace(0, 1, self.ns)

        self.xm = f.variables['xm'].data
        self.xn = f.variables['xn'].data
        self.md = len(self.xm)

        self.xm_nyq = f.variables['xm_nyq'].data
        self.xn_nyq = f.variables['xn_nyq'].data
        self.md_nyq = len(self.xm_nyq)

        self.ds = 1. / (self.ns - 1)

        self.fourierAmps = {'R': f.variables['rmnc'].data,
                            'Z': f.variables['zmns'].data,
                            'Lambda': f.variables['lmns'].data,
                            'Bu': f.variables['bsubumnc'].data,
                            'Bv': f.variables['bsubvmnc'].data,
                            'Bs': f.variables['bsubsmns'].data,
                            'Bmod': f.variables['bmnc'].data,
                            'Jacobian': f.variables['gmnc'].data
                            }

        if self.md == self.md_nyq:
            self.nyq_limit = False

            self.cosine_keys = ['R', 'Jacobian', 'Bu', 'Bv', 'Bmod']
            self.sine_keys = ['Z', 'Lambda', 'Bs']

        else:
            self.nyq_limit = True

            self.cosine_keys = ['R', 'Jacobian']
            self.sine_keys = ['Z', 'Lambda']

            self.cosine_nyq_keys = ['Bu', 'Bv', 'Bmod']
            self.sine_nyq_keys = ['Bs']

        if diffAmps:
            self.fourierAmps['dR_ds'] = np.gradient(
                f.variables['rmnc'].data, self.ds, axis=0)
            self.fourierAmps['dR_du'] = -f.variables['rmnc'].data * self.xm
            self.fourierAmps['dR_dv'] = f.variables['rmnc'].data * self.xn

            self.fourierAmps['dZ_ds'] = np.gradient(
                f.variables['zmns'].data, self.ds, axis=0)
            self.fourierAmps['dZ_du'] = f.variables['zmns'].data * self.xm
            self.fourierAmps['dZ_dv'] = -f.variables['zmns'].data * self.xn

            self.cosine_keys.extend(['dR_ds', 'dZ_du', 'dZ_dv'])
            self.sine_keys.extend(['dR_du', 'dR_dv', 'dZ_ds'])

        if curvAmps:
            self.fourierAmps['dBs_du'] = f.variables['bsubsmns'].data * self.xm
            self.fourierAmps['dBs_dv'] = - \
                f.variables['bsubsmns'].data * self.xn

            self.fourierAmps['dBu_ds'] = np.gradient(
                f.variables['bsubumnc'].data, self.ds, axis=0)
            self.fourierAmps['dBu_dv'] = f.variables['bsubumnc'].data * self.xn

            self.fourierAmps['dBv_ds'] = np.gradient(
                f.variables['bsubvmnc'].data, self.ds, axis=0)
            self.fourierAmps['dBv_du'] = f.variables['bsubvmnc'].data * self.xm

            self.fourierAmps['dBmod_ds'] = np.gradient(
                f.variables['bmnc'].data, self.ds, axis=0)
            self.fourierAmps['dBmod_du'] = -f.variables['bmnc'].data * self.xm
            self.fourierAmps['dBmod_dv'] = f.variables['bmnc'].data * self.xn

            if self.nyq_limit:
                self.cosine_nyq_keys.extend(
                    ['dBs_du', 'dBs_dv', 'dBu_ds', 'dBv_ds', 'dBmod_ds'])
                self.sine_nyq_keys.extend(
                    ['dBu_dv', 'dBv_du', 'dBmod_du', 'dBmod_dv'])

            else:
                self.cosine_keys.extend(
                    ['dBs_du', 'dBs_dv', 'dBu_ds', 'dBv_ds', 'dBmod_ds'])
                self.sine_keys.extend(
                    ['dBu_dv', 'dBv_du', 'dBmod_du', 'dBmod_dv'])

        f.close()

    def transForm_3D(self, u_dom, v_dom, ampKeys):
        """ Performs 3D Fourier transform on specified keys

        Parameters
        ----------
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.u_num = u_dom.shape[0]
        self.v_num = v_dom.shape[0]

        self.u_dom = u_dom
        self.v_dom = v_dom

        pol, tor = np.meshgrid(self.u_dom, self.v_dom)

        pol_xm = np.dot(
            self.xm.reshape(
                self.md, 1), pol.reshape(
                1, self.v_num * self.u_num))
        tor_xn = np.dot(
            self.xn.reshape(
                self.md, 1), tor.reshape(
                1, self.v_num * self.u_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(
                        ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = np.dot(
                        self.xm_nyq.reshape(
                            self.md_nyq, 1), pol.reshape(
                            1, self.v_num * self.u_num))
                    tor_nyq_xn = np.dot(
                        self.xn_nyq.reshape(
                            self.md_nyq, 1), tor.reshape(
                            1, self.v_num * self.u_num))

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], cos_mu_nv).reshape(
                    self.ns, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], sin_mu_nv).reshape(
                    self.ns, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], cos_nyq_mu_nv).reshape(
                    self.ns, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], sin_nyq_mu_nv).reshape(
                    self.ns, self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_2D_sSec(self, rEff, u_dom, v_dom, ampKeys):
        """ Performs 2D Fourier transform along one flux surface on specified keys

        Parameters
        ----------
        rEff : float
            effective radius of flux surface on which Fourier tronsfrom will
            be performed
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = 1
        self.u_num = u_dom.shape[0]
        self.v_num = v_dom.shape[0]

        self.u_dom = u_dom
        self.v_dom = v_dom

        self.s_idx = np.argmin(np.abs(self.s_dom - rEff))

        pol, tor = np.meshgrid(self.u_dom, self.v_dom)

        pol_xm = np.dot(
            self.xm.reshape(
                self.md, 1), pol.reshape(
                1, self.v_num * self.u_num))
        tor_xn = np.dot(
            self.xn.reshape(
                self.md, 1), tor.reshape(
                1, self.v_num * self.u_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(
                        ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = np.dot(
                        self.xm_nyq.reshape(
                            self.md_nyq, 1), pol.reshape(
                            1, self.v_num * self.u_num))
                    tor_nyq_xn = np.dot(
                        self.xn_nyq.reshape(
                            self.md_nyq, 1), tor.reshape(
                            1, self.v_num * self.u_num))

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][self.s_idx], cos_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][self.s_idx], sin_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][self.s_idx], cos_nyq_mu_nv).reshape(
                    self.ns, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][self.s_idx], sin_nyq_mu_nv).reshape(
                    self.ns, self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_2D_uSec(self, u, v_dom, ampKeys):
        """ Performs 2D Fourier transform along a poloidal cross section on
        specified keys

        Parameters
        ----------
        u : float
            poloidal coordinate at which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = self.ns
        self.u_num = 1
        self.v_num = v_dom.shape[0]

        self.u_dom = u
        self.v_dom = v_dom

        pol_xm = self.xm.reshape(self.md, 1) * u
        tor_xn = np.dot(
            self.xn.reshape(
                self.md, 1), self.v_dom.reshape(
                1, self.v_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(
                        ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = self.xm_nyq.reshape(self.md_nyq, 1) * u
                    tor_nyq_xn = np.dot(
                        self.xn_nyq.reshape(
                            self.md_nyq, 1), self.v_dom.reshape(
                            1, self.v_num))

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], cos_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], sin_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], cos_nyq_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], sin_nyq_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_2D_vSec(self, u_dom, v, ampKeys):
        """ Performs 2D Fourier transform along a toroidal cross section on
        specified keys

        Parameters
        ----------
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v : float
            toroidal coordinate at which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = self.ns
        self.u_num = u_dom.shape[0]
        self.v_num = 1

        self.u_dom = u_dom
        self.v_dom = v

        pol_xm = np.dot(
            self.xm.reshape(
                self.md, 1), self.u_dom.reshape(
                1, self.u_num))
        tor_xn = self.xn.reshape(self.md, 1) * v

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(
                        ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = np.dot(
                        self.xm_nyq.reshape(
                            self.md_nyq, 1), self.u_dom.reshape(
                            1, self.u_num))
                    tor_nyq_xn = self.xn_nyq.reshape(self.md_nyq, 1) * v

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], cos_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], sin_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], cos_nyq_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key], sin_nyq_mu_nv).reshape(
                    self.s_num, self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_1D(self, rEff, u, v, ampKeys):
        """ Performs 1D Fourier transform at specified flux coordinates on
        specified keys

        Parameters
        ----------
        rEff : float
            effective radius at which to perform Fourier tronsfrom
        u : float
            poloidal coordinate at which to perform Fourier transform
        v : float
            toroidal coordinate at which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        s_idx = np.argmin(np.abs(self.s_dom - rEff))

        pol_xm = self.xm.reshape(self.md, 1) * u
        tor_xn = self.xn.reshape(self.md, 1) * v

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(
                        ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = self.xm_nyq.reshape(self.md_nyq, 1) * u
                    tor_nyq_xn = self.xn_nyq.reshape(self.md_nyq, 1) * v

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key][s_idx], cos_mu_nv)[0]

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key][s_idx], sin_mu_nv)[0]

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key][s_idx], cos_nyq_mu_nv)[0]

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(
                    self.fourierAmps[key][s_idx], sin_nyq_mu_nv)[0]

            else:
                raise NameError('key = {} : is not available'.format(key))

    def boozerSpectra(self, ax, ampKey, nSpec=10, excFirst=True):
        """ Plots the most prominant modes according to the Fourier amplitude
        specified.  The first mode is excluded by default.

        Parameters
        ----------
        ax : object
            axis on which to plot the spectrum
        ampKey : str
            key specifying Fourier amplitude to be plotted
        nSpec : int
            number of modes to plot (default is 10)
        excFirst : boolean
            defaults to exclude the first mode (default is True)
        """
        specData = {}
        specOrder = []
        for i in range(0, self.md):
            key = '[{0}, {1}]'.format(int(self.xn[i]), int(self.xm[i]))
            spec = self.fourierAmps[ampKey][0::, i]
            specMax = np.max(np.abs(spec))

            specData[key] = spec
            specOrder.append([specMax, key])

        if excFirst:
            io = 1
        else:
            io = 0

        specOrder = sorted(specOrder, key=lambda x: x[0], reverse=True)
        for i in range(nSpec):
            key = specOrder[io + i][1]
            spec = specData[key]

            ax.plot(self.s_dom[1::], spec[1::], label=key)
