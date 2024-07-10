import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("..")
from data.multiple_sta_dataset import multiple_station_dataset

trace = pd.read_csv("../data/1999_2019_final_traces_Vs30.csv")
catalog = pd.read_csv("../data/1999_2019_final_catalog.csv")

fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(
    [trace.query("year==2016")["pgv"],trace.query("year!=2016")["pgv"]],
    bins=25,
    edgecolor="black",
    stacked=True,
    label=["test","train"],
)
ax.legend(loc='best')
ax.set_yscale("log")
label = ["1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pga_threshold = np.log10(
    [0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0,10])
pgv_threshold = np.log10(
    [0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4, 3])
ax.vlines(pgv_threshold, 0, 35000, linestyles="dotted", color="k")
for i in range(len(pgv_threshold)-1):
    ax.text((pgv_threshold[i] + pgv_threshold[i + 1]) / 2, 15000, label[i])
ax.set_ylabel("number of trace")
ax.set_xlabel("log(PGV (m/s))")
ax.set_title("TSMIP data PGV distribution")
# fig.savefig("../data/pgv distribution.png",dpi=300)

fig, ax = plt.subplots(figsize=(7, 7))
ax.hist(
    [catalog.query("year>=2009")["magnitude"],catalog.query("year<2009")["magnitude"]],
    bins=25,
    edgecolor="black",
    stacked=True,
    label=["origin","increased"],
)
ax.legend(loc='best')
ax.set_yscale("log")
ax.set_ylabel("number of event")
ax.set_xlabel("magnitude")
ax.set_title("TSMIP data magnitude distribution")
# fig.savefig("./events_traces_catalog/magnitude distribution.png",dpi=300)

full_data = multiple_station_dataset(
    "../data/TSMIP_1999_2019_Vs30_integral.hdf5",
    mode="train",
    mask_waveform_sec=3,
    weight_label=False,
    oversample=1.5,
    oversample_mag=4,
    test_year=2016,
    mask_waveform_random=True,
    mag_threshold=0,
    label_key="pgv",
    input_type="vel",
    data_length_sec=15,
    station_blind=True,
    bias_to_closer_station=True,
)