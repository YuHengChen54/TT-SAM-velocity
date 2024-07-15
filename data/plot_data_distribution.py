import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = "TSMIP_1999_2019_Vs30_integral.hdf5"
mag_threshold = 0

init_event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")


event_metadata = init_event_metadata[
init_event_metadata["magnitude"] >= mag_threshold
]

event_filter = []
trace_data = pd.DataFrame()

for eq_id in event_metadata["EQ_ID"]:
    filter_id = trace_metadata["EQ_ID"] == eq_id
    filter_pgv = trace_metadata[filter_id]["pgv"] >= np.log10(0.057)
    if len(trace_metadata[filter_id][filter_pgv]) < 1:
        event_filter.append(False)
    else:
        event_filter.append(True)
        trace_data = pd.concat([trace_data, trace_metadata[filter_id]], join="outer")

event_metadata = event_metadata[event_filter]
magnitude_count = event_metadata["magnitude"].value_counts().sort_index(ascending=True)


fig, ax = plt.subplots(figsize=(10,7))
ax.hist([event_metadata.query("year==2016")["magnitude"], event_metadata.query("year!=2016")["magnitude"]], 
                bins=30, 
                edgecolor="black", 
                stacked=True, 
                label=['test', 'train'])
ax.legend()
ax.set_yscale("log")


fig, ax1 = plt.subplots(figsize=(10,7))
ax1.hist([trace_data.query("year==2016")["pgv"], trace_data.query("year!=2016")["pgv"]], 
                bins=30, 
                edgecolor="black", 
                stacked=True, 
                label=['test', 'train'])
ax1.set_yscale("log")