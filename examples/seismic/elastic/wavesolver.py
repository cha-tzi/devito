from devito import Function, TimeFunction, memoized_meth, warning
from examples.seismic import PointSource, Receiver
from examples.seismic.elastic.operators import ForwardOperator


class ElasticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    :param model: Physical model with domain parameters
    :param source: Sparse point symbol providing the injected wave
    :param receiver: Sparse point symbol describing an array of receivers
    :param time_order: Order of the time-stepping scheme (default: 2, choices: 2,4)
                       time_order=4 will not implement a 4th order FD discretization
                       of the time-derivative as it is unstable. It implements instead
                       a 4th order accurate wave-equation with only second order
                       time derivative. Full derivation and explanation of the 4th order
                       in time can be found at:
                       http://www.hl107.math.msstate.edu/pdfs/rein/HighANM_final.pdf
    :param space_order: Order of the spatial stencil discretisation (default: 4)

    Note: space_order must always be greater than time_order
    Note2: This is an experimental staggered grid elastic modeling kernel. Only 2D supported
    """
    def __init__(self, model, source, receiver, space_order=4, **kwargs):
        self.model = model
        self.source = source
        self.receiver = receiver

        self.space_order = space_order
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        # Cache compiler options
        self._kwargs = kwargs
        warning("This is an experimental staggered grid elastic modeling kernel." +
                "Only 2D supported")

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, source=self.source,
                               receiver=self.receiver,
                               space_order=self.space_order, **self._kwargs)


    def forward(self, src=None, vp=None, vs=None, rho=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        :param src: Symbol with time series data for the injected source term
        :param rec: Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        :param save: Option to store the entire (unrolled) wavefield

        :returns: Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        if src is None:
            src = self.source
        # Create a new receiver object to store the result
        rec1 = Receiver(name='rec1', grid=self.model.grid,
                       time_range=self.receiver.time_range,
                       coordinates=self.receiver.coordinates.data)
        rec2 = Receiver(name='rec2', grid=self.model.grid,
                       time_range=self.receiver.time_range,
                       coordinates=self.receiver.coordinates.data)

        # Create all the fields vx, vz, tau_xx, tau_zz, tau_xz
        vx = TimeFunction(name='vx', grid=self.model.grid, staggered=(0, 1, 0),
                          save=source.nt if save else None,
                          time_order=2, space_order=self.space_order)
        vz = TimeFunction(name='vz', grid=self.model.grid, staggered=(0, 0, 1),
                          save=source.nt if save else None,
                          time_order=2, space_order=self.space_order)
        txx = TimeFunction(name='txx', grid=self.model.grid,
                          save=source.nt if save else None,
                          time_order=2, space_order=self.space_order)
        tzz = TimeFunction(name='tzz', grid=self.model.grid,
                          save=source.nt if save else None,
                          time_order=2, space_order=self.space_order)
        txz = TimeFunction(name='txz', grid=self.model.grid, staggered=(0, 1, 1),
                          save=source.nt if save else None,
                          time_order=2, space_order=self.space_order)
        # Pick m from model unless explicitly provided
        if vp is None:
            vp = vp or self.model.vp
        if vs is None:
            vs = vs or self.model.vs
        if rho is None:
            rho = rho or self.model.rho
        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec1=rec1, vx=vx, vz=vz, txx=txx,
                                          tzz=tzz, txz=txz, vp=vp, vs=vs, rho=rho, rec2=rec2,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec1, rec2, vx, vz, txx, tzz, txz, summary
