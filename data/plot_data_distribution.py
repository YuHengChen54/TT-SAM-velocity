import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

data_path = "TSMIP_1999_2019_Vs30_integral.hdf5"
mag_threshold = 0

init_event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")


event_metadata = init_event_metadata[
init_event_metadata["magnitude"] >= mag_threshold
]

mag_oversample = False
bias_to_close_station = False
event_data = pd.DataFrame()
trace_data = pd.DataFrame()


#============================ 計算 Magnitude Oversample ============================#

if mag_oversample:
    for eq_id in event_metadata["EQ_ID"]:
        filter_id = event_metadata["EQ_ID"] == eq_id
        if event_metadata[filter_id]["magnitude"].values[0] > 4:
            repeat_time = int(event_metadata.query(f"EQ_ID == {eq_id}")["magnitude"].values[0])
            for i in range(repeat_time):
                event_data = pd.concat([event_data, event_metadata[filter_id]], join="outer")
        else:
            event_data = pd.concat([event_data, event_metadata[filter_id]], join="outer")
else:
    event_data = event_metadata

#============================ 過濾pgv大小 ============================#

# event_filter = []
# for eq_id in event_data["EQ_ID"]:
#     filter_id = trace_metadata["EQ_ID"] == eq_id
#     filter_pgv = trace_metadata[filter_id]["pgv"] >= np.log10(0.057)
#     if len(trace_metadata[filter_id][filter_pgv]) < 1:
#         event_filter.append(False)
#     else:
#         event_filter.append(True)
#         trace_data = pd.concat([trace_data, trace_metadata[filter_id]], join="outer")
# event_data = event_data[event_filter]

#============================ 計算 Bias to Close Station ============================#

if bias_to_close_station:
    for eq_id in tqdm(event_data["EQ_ID"]):
        filter_id = trace_metadata["EQ_ID"] == eq_id
        if len(trace_metadata[filter_id]) > 25:
            trace_data = pd.concat([trace_data, trace_metadata[filter_id]], join="outer")
            times = int(np.ceil(len(trace_metadata[filter_id])/25))
            for i in range(times):
                trace_data = pd.concat([trace_data, trace_metadata[filter_id].sort_values(["p_pick_sec"], ascending=True)[0:25]], join="outer")
        else:
            trace_data = pd.concat([trace_data, trace_metadata[filter_id]], join="outer")
else:
    for eq_id in tqdm(event_data["EQ_ID"]):
        filter_id = trace_metadata["EQ_ID"] == eq_id
        trace_data = pd.concat([trace_data, trace_metadata[filter_id]], join="outer")


fig, ax = plt.subplots(figsize=(10,7))
ax.hist([event_data.query("year==2016")["magnitude"], event_data.query("year!=2016")["magnitude"]], 
                bins=30, 
                edgecolor="black", 
                stacked=True, 
                label=['test', 'train'])
ax.legend(loc="upper right")
ax.set_title("Origin Data", fontsize=20)
ax.set_xlabel("Magnitude", fontsize=20)
ax.set_ylabel("Number of Events", fontsize=20)
ax.legend()
ax.set_yscale("log")


fig, ax1 = plt.subplots(figsize=(10,7))
ax1.hist([trace_data.query("year==2016")["pgv"], trace_data.query("year!=2016")["pgv"]], 
                bins=30, 
                edgecolor="black", 
                stacked=True, 
                label=['test', 'train'])
ax1.legend(loc="upper right")
ax1.set_title("Origin Data", fontsize=20)
ax1.set_xlabel("PGV log(m/s)", fontsize=20)
ax1.set_ylabel("Number of Traces", fontsize=20)
ax1.set_yscale("log")