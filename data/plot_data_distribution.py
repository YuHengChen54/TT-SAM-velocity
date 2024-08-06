import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm import tqdm
import random

data_path = "TSMIP_1999_2019_Vs30_integral.hdf5"
mag_threshold = 0

init_event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")


event_metadata = init_event_metadata[init_event_metadata["magnitude"] >= mag_threshold]

mag_oversample = True
bias_to_close_station = True
event_data = pd.DataFrame()
trace_data = pd.DataFrame()
trace_full_data = pd.DataFrame()

# ============================ 計算 Magnitude Oversample ============================#

if mag_oversample:
    for eq_id in event_metadata["EQ_ID"]:
        filter_id = event_metadata["EQ_ID"] == eq_id
        event_data = pd.concat([event_data, event_metadata[filter_id]], join="outer")
        if event_metadata[filter_id]["magnitude"].values[0] > 4:
            repeat_time = int(
                1.25
                ** (
                    event_metadata.query(f"EQ_ID == {eq_id}")["magnitude"].values[0] - 1
                )
                - 1
            )
            if repeat_time >= 1:
                for i in range(repeat_time):
                    event_data = pd.concat(
                        [event_data, event_metadata[filter_id]], join="outer"
                    )
else:
    event_data = event_metadata

# ============================ 過濾pgv大小 ============================#

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

# ============================ 計算 Bias to Close Station ============================#

if bias_to_close_station:
    for eq_id in tqdm(event_data["EQ_ID"]):
        filter_id = trace_metadata["EQ_ID"] == eq_id

        if len(trace_metadata[filter_id]) > 0:    #測站總數少於50就不納入資料
            if len(trace_metadata[filter_id]) > 25:
                time = int(np.ceil(len(trace_metadata[filter_id]) / 25))
                # 隨機刪除遠場資料
                numbers = np.arange(time)
                # 将前 70% 和后 30% 分开
                split_point = int(np.floor(0.6 * time))
                head_numbers = numbers[:split_point]
                tail_numbers = numbers[split_point:]
                delete_count = int(np.floor(time * 0.4 / 2))
                # 从后 30% 中随机选择要删除的数字
                if time >= 5:
                    to_delete = np.random.choice(
                        tail_numbers, size=delete_count, replace=False
                    )
                    # 保留未被删除的数字
                    remaining_tail_numbers = np.setdiff1d(tail_numbers, to_delete)

                    # 合并保留的前 70% 和后 30% 中未被删除的数字
                    remaining_numbers = np.concatenate(
                        (head_numbers, remaining_tail_numbers)
                    )
                else:
                    remaining_numbers = np.concatenate((head_numbers, tail_numbers))

                for i in remaining_numbers:
                    trace_data = pd.concat(
                        [
                            trace_data,
                            trace_metadata[filter_id].sort_values(
                                ["p_arrival_abs_time"], ascending=True
                            )[i * 25 : i * 25 + 25],
                        ],
                        join="outer",
                    )
                trace_full_data = pd.concat(
                    [
                        trace_full_data,
                        trace_metadata[filter_id].sort_values(
                            ["p_arrival_abs_time"], ascending=True
                        ),
                    ],
                    join="outer",
                )
            else:
                trace_data = pd.concat(
                    [
                        trace_data,
                        trace_metadata[filter_id].sort_values(
                            ["p_arrival_abs_time"], ascending=True
                        ),
                    ],
                    join="outer",
                )
                trace_full_data = pd.concat(
                    [
                        trace_full_data,
                        trace_metadata[filter_id].sort_values(
                            ["p_arrival_abs_time"], ascending=True
                        ),
                    ],
                    join="outer",
                )
else:
    for eq_id in tqdm(event_data["EQ_ID"]):
        filter_id = trace_metadata["EQ_ID"] == eq_id
        trace_data = pd.concat(
            [
                trace_data,
                trace_metadata[filter_id].sort_values(
                    ["p_arrival_abs_time"], ascending=True
                ),
            ],
            join="outer",
        )
        trace_full_data = pd.concat(
                    [
                        trace_full_data,
                        trace_metadata[filter_id].sort_values(
                            ["p_arrival_abs_time"], ascending=True
                        ),
                    ],
                    join="outer",
                )


fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(
    [
        event_data.query("year==2016")["magnitude"],
        event_data.query("year!=2016")["magnitude"],
    ],
    bins=30,
    edgecolor="black",
    stacked=True,
    label=["test", "train"],
)
ax.legend(loc="upper right")
ax.set_title("Magnitude Oversample Data", fontsize=20)
ax.set_xlabel("Magnitude", fontsize=20)
ax.set_ylabel("Number of Events", fontsize=20)
ax.legend()
ax.set_yscale("log")

pgv_threshold = np.log10([0, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4, 3])
label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
fig, ax1 = plt.subplots(figsize=(10, 7))
ax1.hist(
    [trace_data.query("year==2016")["pgv"], trace_data.query("year!=2016")["pgv"]],
    bins=30,
    edgecolor="black",
    stacked=True,
    label=["test", "train"],
)
for i in range(len(pgv_threshold) - 1):
    ax1.text((pgv_threshold[i] + pgv_threshold[i + 1]) / 2, 35000, label[i])
ax1.legend(loc="upper left")
ax1.vlines(pgv_threshold, 0, 40000, linestyles="dotted", color="k")
ax1.set_title("Delete station less than 25", fontsize=20)
# ax1.set_title("Mag + Bias Data", fontsize=20)
ax1.set_xlabel("PGV log(m/s)", fontsize=20)
ax1.set_ylabel("Number of Traces", fontsize=20)
ax1.set_yscale("log")
ax1.set_ylim(0, 100000)
ax1.set_xlim(-4, 0.5)

# 看PGV隨測站距離的分布
# eq_id = "25480"
# pgv_thresholds = np.log10([0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4, 3])
# labels = label = ["0", "1", "2", "3", "4", "5.0", "5.5", "6.0", "6.5", "7"]
# pgv = trace_metadata.query(f"EQ_ID == {eq_id}").sort_values(["p_arrival_abs_time"])["pgv"]
# for i in pgv.index:
#     for j, threshold in enumerate(pgv_thresholds):
#         if pgv[i] < threshold:
#             pgv[i] = labels[j]
#             break
# pgv = pd.DataFrame(pgv)
# pgv.reset_index(inplace=True)

# fig, ax = plt.subplots(figsize=(10, 7), dpi=400)
# ax.plot(pgv.index, pgv["pgv"])
# ax.set_title(f"EQ_ID : {eq_id}", fontsize=20)
# ax.set_xlabel("staions (sort by distance)", fontsize=20)
# ax.set_ylabel("intensity (5+ -> 5.5)", fontsize=20)
