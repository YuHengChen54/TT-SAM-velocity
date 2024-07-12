import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from analysis import Precision_Recall_Factory

model_num = 13
path = f"../predict/model_{model_num}_analysis"
output_path = f"{path}/model_{model_num}_analysis"
if not os.path.isdir(output_path):
    os.mkdir(output_path)

label = "pgv"
unit = "m/s"

#形成 warning threshold array 其中包含對應的4~5級標準
target_value = np.log10(0.15)

# 生成一個包含目標值的數組
score_curve_threshold = np.linspace(np.log10(0.007), np.log10(0.3), 100)

# 檢查最接近的值
closest_value = min(score_curve_threshold, key=lambda x: abs(x - target_value))

# 調整num參數以確保包含目標值
if closest_value != target_value:
    num_adjusted = 100 + int(np.ceil(abs(target_value - closest_value) / np.diff(score_curve_threshold[:2])))
    score_curve_threshold = np.linspace(np.log10(0.007), np.log10(0.3), num_adjusted)


intensity_score_dict = {"second": [], "intensity_score": []}
f1_curve_fig, f1_curve_ax = plt.subplots()
precision_curve_fig, precision_curve_ax = plt.subplots()
recall_curve_fig, recall_curve_ax = plt.subplots()
for mask_after_sec in [3, 5, 7, 10, 15]:
    data = pd.read_csv(f"{path}/{mask_after_sec} sec model{model_num} with all info_vel.csv")

    predict_label = data["predict"]
    real_label = data["answer"]
    # calculate intensity score
    intensity = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    data["predicted_intensity"] = predict_label.apply(Precision_Recall_Factory.pgv_to_intensity)
    data["answer_intensity"] = real_label.apply(Precision_Recall_Factory.pgv_to_intensity)
    intensity_score = (
        (data["predicted_intensity"] == data["answer_intensity"]).sum()
    ) / len(data)
    intensity_score_dict["second"].append(mask_after_sec)
    intensity_score_dict["intensity_score"].append(intensity_score)
    intensity_table = pd.DataFrame(intensity_score_dict)

    # intensity_table.to_csv(
    #     f"{output_path}/intensity table.csv",
    #     index=False,
    # )
    # plot intensity score confusion matrix
    intensity_confusion_matrix = confusion_matrix(
        data["answer_intensity"], data["predicted_intensity"], labels=intensity
    )
    fig,ax=Precision_Recall_Factory.plot_intensity_confusion_matrix(intensity_confusion_matrix,intensity_score,mask_after_sec,output_path=f"../predict/model_{model_num}_analysis")

    performance_score = {
        f"{label}_threshold ({unit})": [],
        "confusion matrix": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "F1": [],
    }
    for label_threshold in score_curve_threshold:
        predict_logic = np.where(predict_label > label_threshold, 1, 0)
        real_logic = np.where(real_label > label_threshold, 1, 0)
        matrix = confusion_matrix(real_logic, predict_logic, labels=[1, 0])
        accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)  # (TP+TN)/all
        precision = matrix[0][0] / np.sum(matrix, axis=0)[0]  # TP/(TP+FP)
        recall = matrix[0][0] / np.sum(matrix, axis=1)[0]  # TP/(TP+FP)
        F1_score = 2 / ((1 / precision) + (1 / recall))
        performance_score[f"{label}_threshold ({unit})"].append(
            np.round((10**label_threshold), 3)
        )  # m/s^2 / 9.8 = g
        performance_score["confusion matrix"].append(matrix)
        performance_score["accuracy"].append(accuracy)
        performance_score["precision"].append(precision)
        performance_score["recall"].append(recall)
        performance_score["F1"].append(F1_score)

    f1_curve_fig, f1_curve_ax = Precision_Recall_Factory.plot_score_curve(
        performance_score,
        f1_curve_fig,
        f1_curve_ax,
        "F1",
        score_curve_threshold,
        mask_after_sec,
        output_path=f"../predict/model_{model_num}_analysis",
    )
    precision_curve_fig, precision_curve_ax = Precision_Recall_Factory.plot_score_curve(
        performance_score,
        precision_curve_fig,
        precision_curve_ax,
        "precision",
        score_curve_threshold,
        mask_after_sec,
        output_path=f"../predict/model_{model_num}_analysis",
    )
    recall_curve_fig, recall_curve_ax = Precision_Recall_Factory.plot_score_curve(
        performance_score,
        recall_curve_fig,
        recall_curve_ax,
        "recall",
        score_curve_threshold,
        mask_after_sec,
        output_path=f"../predict/model_{model_num}_analysis",
    )

    predict_table = pd.DataFrame(performance_score)
    # predict_table.to_csv(
    #     f"{output_path}/{mask_after_sec} sec confusion matrix table.csv",
    #     index=False,
    # )
