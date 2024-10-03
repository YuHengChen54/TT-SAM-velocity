import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from read_tsmip import cut_traces
import matplotlib.pyplot as plt


start_year = 1999
end_year = 2019
sta_path = "../data"
waveform_path = "../data/waveform"
catalog = pd.read_csv(f"../data/{start_year}_{end_year}_final_catalog.csv")
traces = pd.read_csv(f"../data/{start_year}_{end_year}_final_traces_Vs30.csv")
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
traces.loc[traces.index, "p_pick_sec"] = pd.to_timedelta(
    traces["p_pick_sec"], unit="sec"
)
traces.loc[traces.index, "p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S.%f"
)

output = f"../data/TSMIP_{start_year}_{end_year}_Vs30_integral.hdf5"
error_event = {"EQ_ID": [], "reason": []}


def plot_waveform(
    waveform_type_name,
    title,
    event_ML,
    y_lim=None,
    x_lim_sec=None,
    y_scale=None,
    ylabel=None,
    xlabel=None,
):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=350)
    y_min = 10000
    y_max = -10000
    for i in range(len(event_ML)):
        waveform = data[f"{event_ML[i][0]}"][f"{waveform_type_name}"][0][:, 0]
        TP = np.array(waveform)
        ax.plot(TP, label=f"ML {event_ML[i][1]}")
        minimum = min(waveform)
        maximum = max(waveform)
        if minimum < y_min:
            y_min = minimum
        if maximum > y_max:
            y_max = maximum * 1.1

    if x_lim_sec:
        ticks = np.arange(0, 6001, 200)  # 設置刻度，從 0 到 6000，每隔 200 個數據點
        tick_labels = np.arange(-5, 26, 1)  # 每個 200 個數據點對應 1 秒
        # 修改 X 軸的刻度和標籤
        ax.set_xticks(ticks, tick_labels)
        ax.set_xlim(1000, (x_lim_sec + 5) * 200)
    else:
        ticks = np.arange(0, 6001, 1000)  # 設置刻度，從 0 到 6000，每隔 200 個數據點
        tick_labels = np.arange(-5, 26, 5)  # 每個 200 個數據點對應 1 秒
        # 修改 X 軸的刻度和標籤
        ax.set_xticks(ticks, tick_labels)
        ax.set_xlim(0, 6000)

    if y_lim == "auto":
        ax.set_ylim(y_min, y_max)
    elif y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])

    if y_scale:
        ax.set_yscale(f"{y_scale}")

    ax.set_title(f"{title}", fontsize=22)
    ax.set_ylabel(f"{ylabel}", fontsize=18)
    ax.set_xlabel(f"{xlabel}", fontsize=18)
    ax.legend(loc="upper left")
    fig.show()


##################################### Reintegral displacement data #####################################
# refer reintegrate velocity waveform

with h5py.File(output, "r+") as file:
    data = file["data"]
    meta = file["metadata"]
    for eq_id in tqdm(catalog["EQ_ID"]):
        try:
            _, dis_info = cut_traces(
                traces, eq_id, waveform_path, waveform_type="dis", vel_lowpass_freq=None
            )

            event = data[f"{eq_id}"]
            event.create_dataset(
                "dis_traces_reintegrate", data=dis_info["traces"], dtype=np.float64
            )
        except Exception as reason:
            print(f"EQ_ID:{eq_id}, {reason}")
            error_event["EQ_ID"].append(eq_id)
            error_event["reason"].append(reason)
            continue

##################################### Pd #####################################
# Use np.maximum.accumulate to find the Maximum of displacement
with h5py.File(output, "r+") as file:
    data = file["data"]
    meta = file["metadata"]
    for eq_id in tqdm(catalog["EQ_ID"]):
        try:
            event = data[f"{eq_id}"]
            dis_new = np.abs(np.array(event["dis_traces_reintegrate"]))
            Pd_list = []
            for i in range(dis_new.shape[0]):
                Pd = np.maximum.accumulate(dis_new[i][:, 0])
                Pd_list.append(Pd)
            Pd_list = np.array(Pd_list)
            Pd_list = np.expand_dims(Pd_list, axis=2)

            event.create_dataset("Peak_dis", data=Pd_list, dtype=np.float64)
        except Exception as reason:
            print(f"EQ_ID:{eq_id}, {reason}")
            error_event["EQ_ID"].append(eq_id)
            error_event["reason"].append(reason)
            continue

##################################### CAV #####################################
# Use np.add.accumulate to calculate the cumulative velocity
with h5py.File(output, "r+") as file:
    data = file["data"]
    meta = file["metadata"]
    for eq_id in tqdm(catalog["EQ_ID"]):
        try:
            event = data[f"{eq_id}"]
            vel_wave = np.array(event["vel_traces"])
            CAV_list = []
            for i in range(vel_wave.shape[0]):
                CAV = np.add.accumulate(np.abs(vel_wave[i][:, 0]))
                CAV_list.append(CAV)
            CAV_list = np.array(CAV_list)
            CAV_list = np.expand_dims(CAV_list, axis=2)

            event.create_dataset("Cumulative_abs_vel", data=CAV_list, dtype=np.float64)
        except Exception as reason:
            print(f"EQ_ID:{eq_id}, {reason}")
            error_event["EQ_ID"].append(eq_id)
            error_event["reason"].append(reason)
            continue

##################################### TP #####################################
# First calculate TAUc by the paper Development of an Earthquake Early Warning System Using Real-Time Strong Motion Signals
# Use np.linalg.norm to calculate the vector of velocity & displacement
# integral displacemant & velocity (add from t = 0~T) --> r (Use np.add.accumulate)
# TAUc = (2*pi)/sqrt(r)
# TP = TAUc * Pd
with h5py.File(output, "r+") as file:
    data = file["data"]
    meta = file["metadata"]
    for eq_id in tqdm(catalog["EQ_ID"]):
        try:
            event = data[f"{eq_id}"]
            vel_wave = np.array(event["vel_traces"])
            dis_wave = np.array(event["dis_traces_reintegrate"])
            Pd = np.array(event["Peak_dis"])
            vel_all = (np.linalg.norm(vel_wave, axis=2)) ** 2
            dis_all = (np.linalg.norm(dis_wave, axis=2)) ** 2
            vel_cum = np.add.accumulate(vel_all, axis=1)
            dis_cum = np.add.accumulate(dis_all, axis=1)
            r = vel_cum / dis_cum
            TAUc = (2 * np.pi) / np.sqrt(r)
            TAUc = np.expand_dims(TAUc, axis=2)
            TP = TAUc * Pd

            event.create_dataset("TP", data=TP, dtype=np.float64)
        except Exception as reason:
            print(f"EQ_ID:{eq_id}, {reason}")
            error_event["EQ_ID"].append(eq_id)
            error_event["reason"].append(reason)
            continue

##################################### plot #####################################
with h5py.File(output, "r+") as file:
    data = file["data"]
    meta = file["metadata"]
    event_ML = [
        ["5932", 7.3],
        ["24784", 6.6],
        ["25900", 6.15],
        ["28548", 5],
        ["25285", 4.45],
        ["25480", 3.66],
    ]

    plot_waveform(
        waveform_type_name="TP",
        title="TAUc * Pd (TP)",
        event_ML=event_ML,
        y_lim="auto",
        x_lim_sec=10,
        y_scale="log",
        ylabel="TP (s*m)",
        xlabel="Seconds after P-arrival (s)",
    )
