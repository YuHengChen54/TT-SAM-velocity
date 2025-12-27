import numpy as np
import pandas as pd
import os
import re
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from analysis import Precision_Recall_Factory

model_list_pa = [1, 2, 3]
model_list_pv = [5, 31, 33]
model_list_pd = [7, 8, 9]
model_list_cvaa = [10, 11, 12]
model_list_cvav = [34, 14, 35]
model_list_cvad = [16, 17, 18]
model_list_CAV = [37, 38, 21]
model_list_Ia = [22, 23, 24]
model_list_IV2 = [25, 26, 27]
model_list_TP = [41, 29, 39]

score_pa = {}
score_pv = {}
score_pd = {}
score_cvaa = {}
score_cvav = {}
score_cvad = {}
score_CAV = {}
score_Ia = {}
score_IV2 = {}
score_TP = {}

model_type_list = [model_list_pa, model_list_pv, model_list_pd,
                model_list_cvaa, model_list_cvav, model_list_cvad,
                model_list_CAV, model_list_Ia, model_list_IV2, model_list_TP]

final_score_list = [score_pa, score_pv, score_pd,
                score_cvaa, score_cvav, score_cvad,
                score_CAV, score_Ia, score_IV2, score_TP]

time_after_p_arrival = 13
file_root_path = "../predict_features_validation"
for model_type, final_score in zip(model_type_list, final_score_list):
    r2_score_pga_list = []
    r2_score_pgv_list = []
    precision_pga_list = []
    recall_pga_list = []
    f1_score_pga_list = []
    precision_pgv_list = []
    recall_pgv_list = []
    f1_score_pgv_list = []
    for model_num in model_type:
        file_path = os.path.join(file_root_path, f"model_{model_num}")
        file_list = os.listdir(file_path)
        csv_files = [f for f in file_list if f.endswith('.csv')]
        # 篩選出檔名包含 " {time_after_p_arrival} " 的檔案
        file = [f for f in csv_files if f" {time_after_p_arrival} " in f][0]


        data = pd.read_csv(os.path.join(file_path, file))
        predict_pgv = data["predict"].values
        answer_pgv = data["answer"].values
        # Calculate r2 score for PGA
        r2_score_pgv = metrics.r2_score(answer_pgv, predict_pgv)

        predict_logic_pgv = np.where(predict_pgv > np.log10(0.057), 1, 0)
        answer_logic_pgv = np.where(answer_pgv > np.log10(0.057), 1, 0)
        precision_pgv, recall_pgv, f1_score_pgv = Precision_Recall_Factory.calculate_precision_recall_f1(answer_logic_pgv, predict_logic_pgv)

        r2_score_pgv_list.append(r2_score_pgv)
        precision_pgv_list.append(precision_pgv)
        recall_pgv_list.append(recall_pgv)
        f1_score_pgv_list.append(f1_score_pgv)

    final_score["R2 Score"] = r2_score_pgv_list
    final_score["Precision"] = precision_pgv_list
    final_score["Recall"] = recall_pgv_list
    final_score["F1 Score"] = f1_score_pgv_list

# # # =========== find highest r2 ==============
# highest_avg_r2_pga = 0.0
# highest_avg_r2_pgv = 0.0
# highest_avg_r2_pga_model = None
# highest_avg_r2_pgv_model = None
# highest_r2_pga = 0.0
# highest_r2_pgv = 0.0
# highest_r2_pga_model = None
# highest_r2_pgv_model = None

final_score_list_name = ["pa", "pv", "pd",
                        "cvaa", "cvav", "cvad",
                        "CAV", "Ia", "IV2", "TP"]

# for final_score, final_score_name in zip(final_score_list[0:12], final_score_list_name[0:12]):
#     avg_r2_pga = np.mean(final_score["R2 Score PGA"])
#     avg_r2_pgv = np.mean(final_score["R2 Score PGV"])
#     max_r2_pga = np.max(final_score["R2 Score PGA"]) if len(final_score["R2 Score PGA"]) > 0 else 0
#     max_r2_pgv = np.max(final_score["R2 Score PGV"]) if len(final_score["R2 Score PGV"]) > 0 else 0
#     if avg_r2_pga > highest_avg_r2_pga:
#         highest_avg_r2_pga = avg_r2_pga
#         highest_avg_r2_pga_model = final_score_name
#     if avg_r2_pgv > highest_avg_r2_pgv:
#         highest_avg_r2_pgv = avg_r2_pgv
#         highest_avg_r2_pgv_model = final_score_name
#     if max_r2_pga > highest_r2_pga:
#         highest_r2_pga = max_r2_pga
#         highest_r2_pga_model = final_score_name
#     if max_r2_pgv > highest_r2_pgv:
#         highest_r2_pgv = max_r2_pgv
#         highest_r2_pgv_model = final_score_name

#%% =========== plot ===========
plot_list = final_score_list
plot_list_name = final_score_list_name

# plot_list = final_score_list[::3][0:4]
# plot_list_name = final_score_list_name[::3][0:4]

pgv_score_type = ["R2 Score", "Precision", "Recall"]
colors = ["lightskyblue", "deepskyblue", "dodgerblue"]

x = np.arange(len(plot_list_name))
bar_width = 0.2
fig, ax = plt.subplots(figsize=(9, 5), dpi=450)
for idx, (score_type, color) in enumerate(zip(pgv_score_type, colors)):
    avg_r2_scores = []
    for final_score, final_score_name in zip(plot_list, plot_list_name):
        avg_r2_score = np.mean(final_score[score_type])
        avg_r2_scores.append(avg_r2_score)
    ax.bar(x + idx * bar_width, avg_r2_scores, label=score_type, color=color, width=bar_width)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(plot_list_name)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model Type and Learning Rate")
    ax.set_ylabel("Score")
    ax.set_title("Average Performance Scores")
    ax.legend()


fig, ax1 = plt.subplots(figsize=(9, 5), dpi=450)
for idx, (score_type, color) in enumerate(zip(pgv_score_type, colors)):
    max_r2_scores = []
    for final_score, final_score_name in zip(plot_list, plot_list_name):
        max_r2_score = np.max(final_score[score_type]) if len(final_score[score_type]) > 0 else 0
        max_r2_scores.append(max_r2_score)
    ax1.bar(x + idx * bar_width, max_r2_scores, label=score_type, color=color, width=bar_width)
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels(plot_list_name)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Model Type and Learning Rate")
    ax1.set_ylabel("Score")
    ax1.set_title("Maximum Performance Scores")
    ax1.legend()