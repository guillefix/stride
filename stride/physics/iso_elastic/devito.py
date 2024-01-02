
import os
import glob
import shutil
import tempfile
import warnings
import numpy as np
import scipy.signal

import mosaic
from mosaic.utils import camel_case, at_exit
from mosaic.comms.compression import maybe_compress, decompress

from stride.utils import fft
from stride.problem import StructuredData
# from devito import KroneckerDelta
from devito import Grid, Function, Eq, Operator, TensorFunction
from sympy import KroneckerDelta

from ..common.devito import GridDevito, OperatorDevito, config_devito, devito
from ..problem_type import ProblemTypeBase
from .. import boundaries
# from .examples.seismic.source import WaveletSource, RickerSource, GaborSource, TimeAxis
from devito import SpaceDimension, Constant, VectorTimeFunction, TensorTimeFunction, TimeFunction
from ..common.source import WaveletSource, RickerSource, GaborSource, TimeAxis
from ..boundaries import boundaries_registry


__all__ = ['IsoElasticDevito']


warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


@mosaic.tessera
class IsoElasticDevito(ProblemTypeBase):
    space_order = 8
    time_order = 2
    # time_order = 1

    @property
    def subdomains(self):
        return self._cached_subdomains

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.undersampling_factor = 4

        self.boundary_type = 'sponge_boundary_1'
        self.interpolation_type = 'linear'

        dev_grid = kwargs.pop('dev_grid', None)
        self.dev_grid = dev_grid or GridDevito(self.space_order, self.time_order, **kwargs)
        # import pdb; pdb.set_trace()

        self.state_operator = OperatorDevito(self.space_order, self.time_order,
                                                name='elastic_iso_state',
                                                grid=self.dev_grid,
                                                **kwargs)

        self._cached_subdomains = None



    async def before_forward(self, wavelets, vp, vs, rho=None, alpha=None, lam=None, mu=None, **kwargs):

        platform = kwargs.get('platform', 'cpu')
        is_nvidia = platform is not None and 'nvidia' in platform
        is_nvc = platform is not None and (is_nvidia or 'nvc' in platform)
        diff_source = kwargs.pop('diff_source', False)
        save_compression = kwargs.get('save_compression',
                                      'bitcomp' if self.space.dim > 2 else None)
        save_compression = save_compression if (is_nvidia or is_nvc) and devito.pro_available else None
        problem = kwargs.get('problem')
        shot = problem.shot

        num_sources = shot.num_points_sources
        num_receivers = shot.num_points_receivers
        rec_tau = self.dev_grid.sparse_time_function('rec_tau', num=num_receivers,
                                                    coordinates=shot.receiver_coordinates,
                                                    interpolation_type=self.interpolation_type)

        self.src = self.dev_grid.sparse_time_function('src', num=num_sources,
                                                    coordinates=shot.source_coordinates,
                                                    interpolation_type=self.interpolation_type)

        space_order = None
        layers = devito.HostDevice if is_nvidia else devito.NoLayers
        p_saved = self.dev_grid.undersampled_time_function('p_saved',
                                                            bounds=kwargs.pop('save_bounds', None),
                                                            factor=self.undersampling_factor,
                                                            space_order=space_order,
                                                            layers=layers,
                                                            compression=save_compression)

        # if self._needs_grad(wavelets, rho, alpha):
        #     p_saved_expr = p
        # else:
        # p_saved_expr = self._forward_save(p)
        # nani is the above for?

        # t0, tn = 0., 0.00005
        # self.dt = dt = 0.00000002
        self.dt = dt = self.time.step
        print(f"dt: {dt}")
        print(f"End time: {self.time.stop}")

        # Absorbing boundaries
        self.boundary = boundaries_registry[self.boundary_type](self.dev_grid)
        _, _, _ = self.boundary.apply(vel, vp.extended_data)

        # Now we create the velocity and pressure fields

        # self.v = v = VectorTimeFunction(name='v', grid=self.dev_grid.devito_grid, space_order=self.space_order, time_order=self.time_order)
        # self.tau = tau = TensorTimeFunction(name='t', grid=self.dev_grid.devito_grid, space_order=self.space_order, time_order=self.time_order)
        self.v = v = self.dev_grid.vector_time_function('v', coefficients='standard')
        self.tau = tau = self.dev_grid.tensor_time_function('tau', coefficients='standard')
        # self.tau = tau = TensorTimeFunction(name='t', grid=self.dev_grid.devito_grid, space_order=self.space_order, time_order=1, save=self.time.num)

        # self.p_saved = p_saved = TimeFunction(name='p_saved', grid=self.dev_grid.devito_grid, space_order=self.space_order, time_order=self.time_order, save=self.time.num)

        density = self.dev_grid.with_halo(rho.extended_data)
        vp = self.dev_grid.with_halo(vp.extended_data)
        vs = self.dev_grid.with_halo(vs.extended_data)
        vp_fun = vp
        vs_fun = vs

        # Prepare the subdomains
        abox, full, interior, boundary = self._subdomains()

        update_saved = [devito.Eq(p_saved, tau[0], subdomain=abox)]
        # update_saved = []
        devicecreate = (self.dev_grid.vars.tau, self.dev_grid.vars.p_saved,)


        # src_scale = 2 * self.time.step**2 * vp_fun / max(*self.space.spacing)
        src_scale = 1000

        # The source injection term
        src_xx = self.src.inject(field=tau.forward[0,0], expr=self.src * src_scale)
        src_zz = self.src.inject(field=tau.forward[1,1], expr=self.src * src_scale)

        # populate sources with data
        wavelets = wavelets.data

        window = scipy.signal.get_window(('tukey', 0.001), self.time.num, False)
        window = window.reshape((self.time.num, 1))

        self.dev_grid.vars.src.data[:] = wavelets.T * window

        if self.interpolation_type == 'linear':
            self.dev_grid.vars.src.coordinates.data[:] = shot.source_coordinates
            self.dev_grid.vars.rec_tau.coordinates.data[:] = shot.receiver_coordinates

        # Thorbecke's parameter notation
        cp2 = vp*vp
        cs2 = vs*vs
        byn_fun = self.dev_grid.function('byn_fun')
        lam_fun = self.dev_grid.function('lam_fun')
        mu_fun = self.dev_grid.function('mu_fun')
        self.dev_grid.vars.byn_fun.data_with_halo[:] = 1/density

        mu = cs2*density
        l = (cp2*density - 2*mu)
        self.dev_grid.vars.lam_fun.data_with_halo[:] = l
        self.dev_grid.vars.mu_fun.data_with_halo[:] = mu

        # First order elastic wave equation
        pde_v = v.dt - byn_fun * devito.div(tau)
        pde_tau = tau.dt - lam_fun * devito.diag(devito.div(v.forward)) - mu_fun * (devito.grad(v.forward) + devito.grad(v.forward).transpose(inner=False))
        # Time update
        u_v = Eq(v.forward, devito.solve(pde_v, v.forward))
        u_t = Eq(tau.forward, devito.solve(pde_tau, tau.forward))


        # self.op = Operator([u_v] + [u_t]  + src_xx + src_zz + update_saved)
        # Compile the operator
        kwargs['devito_config'] = kwargs.get('devito_config', {})
        kwargs['devito_config']['devicecreate'] = devicecreate

        self.state_operator.set_operator([u_v] + [u_t]  + src_xx + src_zz + update_saved,
                                            **kwargs)
        self.state_operator.compile()


        return

    async def run_forward(self, wavelets, vp, vs, rho=None, alpha=None, lam=None, mu=None, **kwargs):
        devito_args = kwargs.get('devito_args', {})
        # self.op(dt=self.dt, **devito_args)
        self.state_operator.run(dt=self.time.step,
                                **devito_args)

        return

    async def after_forward(self, wavelets, vp, vs, rho=None, alpha=None, lam=None, mu=None, **kwargs):
        # self._wavefield = self.tau[0, 0].data
        # self._wavefield = self.p_saved.data
        # self._wavefield = self.p_saved.data[::4, :,:]
        self._wavefield = self.dev_grid.vars.p_saved.data
        problem = kwargs.pop('problem')
        shot = problem.shot
        traces_data = np.asarray(self.dev_grid.vars.rec_tau.data, dtype=np.float32).T
        # traces_data *= self._max_wavelet / self._src_scale
        traces = shot.observed.alike(name='modelled', data=traces_data)

        return traces
    
    @property
    def wavefield(self):
        if self._wavefield is None:
            return None

        wavefield_data = np.asarray(self._wavefield.data, dtype=np.float32)
        wavefield = StructuredData(name='p',
                                   data=wavefield_data,
                                   shape=wavefield_data.shape)

        return wavefield

    def _subdomains(self, *args, **kwargs):
        problem = kwargs.get('problem')

        full = self.dev_grid.full
        interior = self.dev_grid.interior
        boundary = self.dev_grid.pml
        self._cached_subdomains = (full, full, interior, boundary)

        return full, full, interior, boundary
