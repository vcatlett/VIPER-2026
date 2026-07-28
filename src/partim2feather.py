import sys, json
from tqdm import tqdm
from glob import glob
import pandas as pd
from pta_replicator.simulate import load_pulsar, load_from_directories
import psutil
import pint
pint.logging.setup(sink=sys.stderr, level="WARNING", usecolors=True)

N_PSR = 2
PSR_NAMES = "/home/catlettv/novus/sandbox/repos/VIPER/VIPER-2026/data/pulsars_15yr.txt"
DATA_DIR = "/home/catlettv/novus/sandbox/repos/VIPER/VIPER-2026/results/A_15__gamma_13_3/"
NOISE_DICT = "/home/catlettv/novus/sandbox/repos/VIPER/VIPER-2026/data/NG15/ng15_dict.json"
# NOISE_DICT = "/home/catlettv/novus/sandbox/repos/VIPER/VIPER-2026/data/NG15/ng15_sim_dict.json"
PAR_DIR = DATA_DIR + "par/"
TIM_DIR = DATA_DIR + "tim/"
FEATHER_DIR = DATA_DIR + "feather/"

with open(NOISE_DICT, 'r') as fp:
    noise_params = json.load(fp)
# change number strings to floats:
for value in noise_params.values():
    value = float(value)

psr_names = pd.read_csv(PSR_NAMES)
# print("Loading pulsar objects...")
# psrs = load_from_directories(PAR_DIR, TIM_DIR, num_psrs=N_PSR)

print("Saving to feather files...")
for pi, p in tqdm(enumerate(psr_names["psr_name"])):
    if p == "J1909-3744":
        pass
    elif pi <= 53:
        pass
    else:
        print(pi)
        parfile = glob(f"{PAR_DIR}/{p}*.par")[0]
        timfile = glob(f"{TIM_DIR}/{p}*.tim")[0]
        psr = load_pulsar(parfile, timfile)
        savename = FEATHER_DIR + f"{p}.feather"
        ePsr = psr.to_enterprise()
        ePsr.to_feather(savename, noisedict=noise_params)
        # Get system-wide memory details
        system_ram = psutil.virtual_memory()
        # Convert bytes to Gigabytes (GB) for easier reading
        total_gb = system_ram.total / (1024 ** 3)
        used_gb = system_ram.used / (1024 ** 3)
        print(f"RAM: {system_ram.percent:.2f}% ({used_gb:.2f}/{total_gb:.2f})")
        del psr, ePsr, parfile, timfile