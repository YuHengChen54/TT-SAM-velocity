import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from read_tsmip import cut_traces

start_year = 1999
end_year = 2019
sta_path = "../data"
waveform_path = "D:/TT-SAM/waveform"
catalog = pd.read_csv(
    f"../data/{start_year}_{end_year}_final_catalog.csv"
)
traces = pd.read_csv(
    f"../data/{start_year}_{end_year}_final_traces_Vs30.csv"
)

station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv") 
traces.loc[traces.index, "p_pick_sec"] = pd.to_timedelta(
    traces["p_pick_sec"], unit="sec"
)
traces.loc[traces.index, "p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S.%f"
)

# into hdf5
output = f"../data/TSMIP_{start_year}_{end_year}_Vs30_integral.hdf5"
error_event = {"EQ_ID": [], "reason": []}

with h5py.File(output, "r+") as file:
    data = file["data"]
    meta = file["metadata"]
    for eq_id in tqdm(catalog["EQ_ID"]):
        try:
            _, vel_info = cut_traces(traces, eq_id, waveform_path, waveform_type="vel", vel_lowpass_freq=0.33)
            
            event = data[f"{eq_id}"]
            # event.create_dataset("vel_lowfreq_traces", data=vel_info["traces"], dtype=np.float64)
            event["pgv"][...] = vel_info["pgv"]
        except Exception as reason:
            print(f"EQ_ID:{eq_id}, {reason}")
            error_event["EQ_ID"].append(eq_id)
            error_event["reason"].append(reason)
            continue

