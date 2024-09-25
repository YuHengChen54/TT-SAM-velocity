import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from read_tsmip import cut_traces

sta_path = "../data"  
waveform_path = "../data/0918_M6.8_1319_1330/ascii"
traces = pd.read_csv("0918_M6.8_1319_1330/traces_catalog.csv")
catalog = pd.read_csv("0918_M6.8_1319_1330/event_catalog.csv")

station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv") 
traces.loc[traces.index, "p_pick_sec"] = pd.to_timedelta(
    traces["p_pick_sec"], unit="sec"
)
traces.loc[traces.index, "p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S.%f"
)

##################################### Reintegral displacement data #####################################
# refer reintegral velocity waveform


##################################### Pd #####################################
# Use np.maximum.accumulate to find the Maximum of displacement


##################################### CAV #####################################
# Use np.add.accumulate to calculate the cumulative velocity


##################################### TP #####################################
# First calculate TAUc by the paper Development of an Earthquake Early Warning System Using Real-Time Strong Motion Signals
# Use np.linalg.norm to calculate the vector of velocity & displacement
# integral displacemant & velocity (add from t = 0~T) --> r
# TAUc = (2*pi)/sqrt(r)
# TP = TAUc * Pd