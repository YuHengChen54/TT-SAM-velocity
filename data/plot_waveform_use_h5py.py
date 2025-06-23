%matplotlib inline
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter

with h5py.File("TSMIP_1999_2019_Vs30_integral.hdf5", "r") as f:
    station = np.array(f["data"]["24784"]["station_name"])
    waveform = np.array(f["data"]["24784"]["vel_traces"])
    waveform_low = np.array(f["data"]["24784"]["vel_lowfreq_traces"])


fig = plt.figure(figsize=(8, 6), dpi=600)

# 1) 用外層 GridSpec 分成 3 段：上半 2 張、空白、下半 2 張
#    height_ratios=[1, 0.3, 1] → 中間那段空白高度是 0.3 倍
outer = GridSpec(
    nrows=3, ncols=1,
    height_ratios=[1, 0.3, 1],
    hspace=0  # 全局不加額外間隔
)

# 2) 上半組再細分 2 行 (row 0 用 top_group)
top_group = outer[0].subgridspec(nrows=2, ncols=1, hspace=0)
ax1 = fig.add_subplot(top_group[0])
ax2 = fig.add_subplot(top_group[1])

# 3) 下半組再細分 2 行 (row 2 用 bot_group)
bot_group = outer[2].subgridspec(nrows=2, ncols=1, hspace=0)
ax3 = fig.add_subplot(bot_group[0])
ax4 = fig.add_subplot(bot_group[1])

# 4) 畫圖示範
ax1.plot(waveform[6, :, 0], color="darkblue", linewidth=1, label=f"{str(station[6])[2:-1]}")
ax2.plot(waveform[9, :, 0], color="darkblue", linewidth=1, label=f"{str(station[9])[2:-1]}")
ax3.plot(waveform_low[6, :, 0], color="darkblue", linewidth=1, label=f"{str(station[6])[2:-1]}")
ax4.plot(waveform_low[9, :, 0], color="darkblue", linewidth=1, label=f"{str(station[9])[2:-1]}")

for ax in (ax1, ax3):
    # 1) 不顯示 label
    ax.xaxis.set_major_formatter(NullFormatter())
    # 2) 確保刻度出現在底部
    ax.xaxis.set_ticks_position('bottom')
    # 3) 刻度往內畫、長度 6pt
    ax.tick_params(
        axis='x',
        which='major',
        direction='in',   # in→從底部往上畫
        length=6,
        labelbottom=False  # 再保險一次不顯示文字
    )
# 让 ax2, ax4 的 x-ticks 也朝图内（向上）画
for ax in (ax2, ax4):
    ax.tick_params(
        axis='x',
        which='major',
        direction='in',    # 刻度线朝内/向上
        length=6,          # 长度 6pt
        labelbottom=True   # 保留文字标签（如已有的话）
    )


for axes in [ax1, ax2, ax3, ax4]:
    axes.set_xlim(0, 6000)
    axes.legend(loc="upper left")

# 1) 先設定想要出現刻度的位置
ticks = [0*200, 5*200, 10*200, 15*200, 20*200, 25*200, 30*200]
ax2.set_xticks(ticks)
ax4.set_xticks(ticks)

# 2) 指定每個位置對應的文字（可以是任何字串）
labels = ["0", "5", "10", "15", "20", "25", "30"]
ax2.set_xticklabels(labels, rotation=0, fontsize=10)
ax4.set_xticklabels(labels, rotation=0, fontsize=10)

ax4.set_xlabel("Time (s)", fontsize=12)
ax1.set_title("Meinong velocity waveform", fontsize=16)
ax3.set_title("Low frequency waveform", fontsize=16)


plt.tight_layout()
plt.show()
# plt.savefig("Vel vs lowfreq waveform (Meinong)", dpi=600)
plt.close()