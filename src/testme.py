import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# from enterprise.pulsar import Pulsar as ePsr
import pint.toa as toa
from pint.residuals import Residuals
from pint.models import get_model_and_toas
from pint.simulation import zero_residuals
import pint.logging
pint.logging.setup(level="WARNING")

load_dotenv()

N_PSR = 1

DATA_ROOT = Path(os.getenv("DATA_ROOT"))
psr_names = pd.read_csv(str(DATA_ROOT / "pulsars_15yr.txt"), header=None)
psr_names.columns = ["name"]

DATA_15YR_ROOT = Path(os.getenv("DATA_15YR_ROOT"))
DATA_15YR_PAR = DATA_15YR_ROOT / "narrowband/par"
DATA_15YR_TIM = DATA_15YR_ROOT / "narrowband/tim"

PLOT_PATH = Path(os.getenv("PLOT_PATH"))

all_pars = []
all_tims = []

for pname in psr_names["name"]:
    par_path = [p.name for p in DATA_15YR_PAR.glob(f"{pname}_PINT*.par")]
    tim_path = [p.name for p in DATA_15YR_TIM.glob(f"{pname}_PINT*.tim")]
    all_pars.append(par_path[0])
    all_tims.append(tim_path[0])

for pi, pname in enumerate(psr_names["name"]):
    if pi < N_PSR:
        # Get par and tim
        test_tim = DATA_15YR_TIM / all_tims[pi]
        test_par = DATA_15YR_PAR / all_pars[pi]

        # Enterprise pulsar
        # psr = ePsr(test_par, test_tim)
        # print("MADE ENTERPRISE PULSAR")

        ## ZERO RESIDUALS
        m, t_all = get_model_and_toas(test_par, test_tim)
        xt = t_all.get_mjds()
        rs_og = Residuals(t_all, m).phase_resids

        ## Zero Residuals
        zero_residuals(t_all, m)
        t_all.print_summary()
        rs_new = Residuals(t_all, m).phase_resids

        ## PLOT
        fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw={'wspace': 0})

        # Original
        cax = ax[0]
        cax.scatter(xt, rs_og, color="k", marker=".")
        cax.set_xlabel("MJD")
        cax.set_ylabel("Residual (Phase)")
        cax.set_title("Original Residuals")

        # Zeroed
        cax = ax[1]
        cax.scatter(xt, rs_new, color="k", marker=".")
        cax.set_xlabel("MJD")
        cax.set_title("Residuals After Zeroing")

        # Save figure
        plt.suptitle(f"{pname} Residuals")
        plt.savefig(PLOT_PATH / f"{pname}_zero_res.png")
        # plt.show()