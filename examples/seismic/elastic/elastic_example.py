import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from devito import Constant
from examples.seismic.elastic import ElasticWaveSolver
from examples.seismic import ModelElastic, RickerSource, Receiver, TimeAxis, Model
from examples.seismic.acoustic import AcousticWaveSolver

def elastic_setup(shape=(50, 50), spacing=(15.0, 15.0),
                   tn=500., space_order=4, nbpml=10,
                   **kwargs):

    # shape = (201, 201)
    shape = (1601, 401)
    spacing = (7.5, 7.5)
    origin = (0., 0.)
    #
    vp = np.fromfile("/data/mlouboutin3/devito_data/Simple2D/vp_marmousi_bi", dtype=np.float32, sep="")
    vp = np.reshape(vp, shape)
    # Cut the model to make it slightly cheaper
    vp = vp[301:-300, :]
    shape = vp.shape
    # vp = 1.500 # * np.ones(shape)
    vs = .5*vp
    rho = vp - np.min(vp) + 1.0
    nrec = shape[0]
    model = ModelElastic(origin, spacing, shape, space_order, vp, vs, rho)
    model2 = Model(origin, spacing, shape, space_order, vp)
    # Derive timestepping from model spacing
    dt = model.critical_dt
    t0 = 0.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Derive timestepping from model spacing
    dt2 = model2.critical_dt
    t0 = 0.0
    time_range2 = TimeAxis(start=t0, stop=tn, step=dt2)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.015, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, lust below surface)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]


    # Define source geometry (center of domain, just below surface)
    src2 = RickerSource(name='src2', grid=model2.grid, f0=0.015, time_range=time_range2)
    src2.coordinates.data[0, :] = np.array(model2.domain_size) * .5
    src2.coordinates.data[0, -1] = model2.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, lust below surface)
    rec2 = Receiver(name='rec2', grid=model2.grid, time_range=time_range2, npoint=nrec)
    rec2.coordinates.data[:, 0] = np.linspace(0., model2.domain_size[0], num=nrec)
    rec2.coordinates.data[:, 1:] = src2.coordinates.data[0, 1:]


    # Create solver object to provide relevant operators
    solver = ElasticWaveSolver(model, source=src, receiver=rec,
                                space_order=space_order, **kwargs)
    solver2 = AcousticWaveSolver(model2, source=src2, receiver=rec2,
                                 space_order=space_order, **kwargs)
    return solver, solver2


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbpml=40, full_run=False,
        autotune=False, **kwargs):

    solver, solver2 = elastic_setup(shape=shape, spacing=spacing, nbpml=nbpml, tn=tn,
                                    space_order=space_order, **kwargs)
    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    rec1, rec2, vx, vz, txx, tzz, txz, summary = solver.forward()
    rec3, u, _ = solver2.forward()
    import matplotlib.pyplot as plt
    plt.figure();plt.imshow(rec1.data[:], cmap="seismic", aspect=.2, vmin=-.01, vmax=.01);
    plt.figure();plt.imshow(rec2.data[:], cmap="seismic", aspect=.2, vmin=-.01, vmax=.01);
    # rec4 = rec3.resample(dt=rec1._time_range.step)
    plt.figure();plt.imshow(rec4.data[:], cmap="seismic", aspect=.2, vmin=-.1, vmax=.1);
    plt.show()
    from IPython import embed; embed();

if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-dse", "-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DSE) mode")
    args = parser.parse_args()

    # 3D preset parameters
    shape = (150, 150)
    spacing = (15.0, 15.0)
    tn = 3500.0

    run(shape=shape, spacing=spacing, nbpml=args.nbpml, tn=tn,
        space_order=args.space_order,
        autotune=args.autotune, dse=args.dse, dle=args.dle, full_run=args.full)
