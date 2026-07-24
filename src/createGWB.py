"""
MODIFIED FROM:
https://github.com/vallis/libstempo/blob/master/libstempo/toasim.py#L748
"""

# General Python imports
import os, sys
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

# NANOGRAV imports
from pta_replicator.white_noise import add_measurement_noise
from pta_replicator.white_noise import add_jitter
from pta_replicator.red_noise import add_red_noise, add_gwb
from pta_replicator.simulate import load_pulsar, load_from_directories, simulate_pulsar
from pta_replicator.simulate import make_ideal

import pint
from pint.models import get_model_and_toas
from pint.residuals import Residuals
from pint.simulation import zero_residuals
pint.logging.setup(sink=sys.stderr, level="WARNING", usecolors=True)

# Load environment from .env file
load_dotenv()

# Random number seed
NP_SEED = int(os.getenv("NP_SEED"))
if NP_SEED is not None:
    np.random.seed(NP_SEED)

# 15-year dataset
DATA_15YR_ROOT = Path(os.getenv("DATA_15YR_ROOT"))
DATA_15YR_PAR = DATA_15YR_ROOT / "narrowband/par"
DATA_15YR_TIM = DATA_15YR_ROOT / "narrowband/tim"

# Test par and tim files
parfile = DATA_15YR_PAR / "B1937+21_PINT_20220306.nb.par"
timfile = DATA_15YR_TIM / "B1937+21_PINT_20220306.nb.tim"

# Zero residuals
model, toas = get_model_and_toas(parfile, timfile)
zero_residuals(toas, model)
toas.print_summary()

# Get MJDs
mjds = toas.get_mjds()
print(toas)
print(mjds)
stop

# PARAMS TO RECOVER
# Amplitude of red noise in GW units 
# [-18, -13]
Amp = -15.
# Red noise power law spectral index 
# [1, 7]
gam = 13./3.

# pulsar object for single pulsar
psr = []
# Add red noise with no spatial correlations
noCorr=False
# Produce spectrum with turnover at frequency f0
turnover=False
# coefficients of spherical harmonic decomposition of GW power
clm = [np.sqrt(4.0 * np.pi)]
# maximum multipole of GW power decomposition
lmax = int(len(clm) - 1)
# Frequency of spectrum turnover
f0=1e-9
# Spectral index of power spectram for f << f0
beta=1
# Fudge factor for flatness of spectrum turnover
power=1
# User-supplied characteristic strain spectrum
# (first column is freqs, second is spectrum)
userSpec=None
# Number of points used in interpolation
npts=600
# Lowest frequency is 1/(howml * T)
howml=10

# number of pulsars
Npulsars = len(psr)

# gw start and end times for entire data set
start = np.min([p.toas().min() * 86400 for p in psr]) - 86400
stop = np.max([p.toas().max() * 86400 for p in psr]) + 86400

# duration of the signal
dur = stop - start

# get maximum number of points
if npts is None:
    # default to cadence of 2 weeks
    npts = dur / (86400 * 14)

# make a vector of evenly sampled data points
ut = np.linspace(start, stop, npts)

# time resolution in days
dt = dur / npts

# compute the overlap reduction function
if noCorr:
    ORF = np.diag(np.ones(Npulsars) * 2)
else:
    psrlocs = np.zeros((Npulsars, 2))

    for ii in range(Npulsars):
        if "RAJ" and "DECJ" in psr[ii].pars():
            psrlocs[ii] = np.double(psr[ii]["RAJ"].val), np.double(psr[ii]["DECJ"].val)
        elif "ELONG" and "ELAT" in psr[ii].pars():
            fac = 180.0 / np.pi
            # check for B name
            if "B" in psr[ii].name:
                epoch = "1950"
            else:
                epoch = "2000"
            coords = ephem.Equatorial(
                ephem.Ecliptic(str(psr[ii]["ELONG"].val * fac), str(psr[ii]["ELAT"].val * fac)), epoch=epoch
            )
            psrlocs[ii] = float(repr(coords.ra)), float(repr(coords.dec))

    psrlocs[:, 1] = np.pi / 2.0 - psrlocs[:, 1]
    anisbasis = np.array(anis.CorrBasis(psrlocs, lmax))
    ORF = sum(clm[kk] * anisbasis[kk] for kk in range(len(anisbasis)))
    ORF *= 2.0

# Define frequencies spanning from DC to Nyquist.
# This is a vector spanning these frequencies in increments of 1/(dur*howml).
f = np.arange(0, 1 / (2 * dt), 1 / (dur * howml))
f[0] = f[1]  # avoid divide by 0 warning
Nf = len(f)

# Use Cholesky transform to take 'square root' of ORF
M = np.linalg.cholesky(ORF)

# Create random frequency series from zero mean, unit variance, Gaussian distributions
w = np.zeros((Npulsars, Nf), complex)
for ll in range(Npulsars):
    w[ll, :] = np.random.randn(Nf) + 1j * np.random.randn(Nf)

# strain amplitude
if userSpec is None:
    f1yr = 1 / 3.16e7
    alpha = -0.5 * (gam - 3)
    hcf = Amp * (f / f1yr) ** (alpha)
    if turnover:
        si = alpha - beta
        hcf /= (1 + (f / f0) ** (power * si)) ** (1 / power)

elif userSpec is not None:
    freqs = userSpec[:, 0]
    if len(userSpec[:, 0]) != len(freqs):
        raise ValueError("Number of supplied spectral points does not match number of frequencies!")
    else:
        fspec_in = interp.interp1d(np.log10(freqs), np.log10(userSpec[:, 1]), kind="linear")
        fspec_ex = extrap1d(fspec_in)
        hcf = 10.0 ** fspec_ex(np.log10(f))

C = 1 / 96 / np.pi**2 * hcf**2 / f**3 * dur * howml

# inject residuals in the frequency domain
Res_f = np.dot(M, w)
for ll in range(Npulsars):
    Res_f[ll] = Res_f[ll] * C ** (0.5)  # rescale by frequency dependent factor
    Res_f[ll, 0] = 0  # set DC bin to zero to avoid infinities
    Res_f[ll, -1] = 0  # set Nyquist bin to zero also

# Now fill in bins after Nyquist (for fft data packing) and take inverse FT
Res_f2 = np.zeros((Npulsars, 2 * Nf - 2), complex)
Res_t = np.zeros((Npulsars, 2 * Nf - 2))
Res_f2[:, 0:Nf] = Res_f[:, 0:Nf]
Res_f2[:, Nf : (2 * Nf - 2)] = np.conj(Res_f[:, (Nf - 2) : 0 : -1])
Res_t = np.real(np.fft.ifft(Res_f2) / dt)

# shorten data and interpolate onto TOAs
Res = np.zeros((Npulsars, npts))
res_gw = []
for ll in range(Npulsars):
    Res[ll, :] = Res_t[ll, 10 : (npts + 10)]
    f = interp.interp1d(ut, Res[ll, :], kind="linear")
    res_gw.append(f(psr[ll].toas() * 86400))

# return res_gw
ct = 0
for p in psr:
    p.stoas[:] += res_gw[ct] / 86400.0
    ct += 1