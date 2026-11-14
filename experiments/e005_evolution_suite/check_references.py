"""Cross-check distributed fields with the frozen initial conditions or independent integrators."""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat

HERE = Path(__file__).parent
DATA = HERE / "data"


def allen_reference(nx, substeps, times):
    # Strang splitting. Both the diffusion flow and the pointwise reaction flow are exact.
    x = -1 + 2 * np.arange(nx) / nx
    k = np.pi * np.fft.fftfreq(nx, 1/nx)
    u = x*x*np.cos(np.pi*x)
    frames = [u]
    for dt in np.diff(times)/substeps:
        half = np.exp(-0.5e-4*dt*k*k)
        decay = np.exp(-10*dt)
        for _ in range(substeps):
            u = np.fft.ifft(half*np.fft.fft(u)).real
            u = u / np.sqrt(decay+(1-decay)*u*u)
            u = np.fft.ifft(half*np.fft.fft(u)).real
        frames.append(u)
    return np.stack(frames, axis=1)


def cgle_crosscheck(reference):
    # Independent method of lines on 32 Fourier nodes, adaptive DOP853 time stepping.
    x = 2*np.pi*np.arange(32)/32
    k = np.fft.fftfreq(32, 1/32)
    initial = .6 + .25*np.exp(1j*x) + .1*np.exp(-2j*x)
    def rhs(t, u):
        lap=np.fft.ifft(-k*k*np.fft.fft(u))
        return u+(1+1j)*lap-(1+.5j)*abs(u)**2*u
    sol=solve_ivp(rhs,(0.,2.),initial,t_eval=reference["t"],atol=1e-12,rtol=1e-10,method="DOP853")
    assert sol.success
    return relative(sol.y.T,reference["fine"][:,::4])


def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


if __name__=="__main__":
    mat=loadmat(DATA/"AC.mat")
    x=mat["x"].ravel(); times=mat["tt"].ravel()
    np.testing.assert_allclose(x,-1+2*np.arange(512)/512)
    ref=mat["uu"].real
    low=allen_reference(512,2,times)
    high=allen_reference(1024,4,times)[::2]
    with h5py.File(DATA/"fhn_first.h5") as f:
        g=f["0000"]; assert list(g["data"].shape)==[101,128,128,2]
        assert all(np.all(np.diff(g["grid/"+axis][:])>0) for axis in "xyt")
        assert "seed: 0" in g.attrs["config"] and "train_0000.h5" in g.attrs["config"]
    result=dict(allen_raw_sha256=hashlib.sha256((DATA/"AC.mat").read_bytes()).hexdigest(),
                allen_initial_max_abs=float(np.max(abs(ref[:,0]-x*x*np.cos(np.pi*x)))),
                allen_relative_512_to_file=relative(low,ref),
                allen_relative_1024_to_file=relative(high,ref),
                allen_relative_coarse_to_fine=relative(low,high),
                cgle_relative_independent_DOP853_32_to_reference_128=cgle_crosscheck(np.load(DATA/"cgle.npz")),
                fhn_archived_group="0000", fhn_trajectory_shape=[101,128,128,2])
    (HERE/"acceptance").mkdir(exist_ok=True)
    (HERE/"acceptance"/"references.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result,indent=2))
