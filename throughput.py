import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1) 数据定义（与原先一致）
# ---------------------------

# Llama-2-13b/512+64/PC-Low
x1a = [32, 64, 128, 256]
y1a = [
    (64 * 32) / 128,
    (64 * 64) / 214,
    (64 * 128) / 440,
    (64 * 256) / 1240,
]
x2a = [32, 64, 128, 256]
y2a = [
    (64 * 32) / 177,
    (64 * 64) / 290,
    (64 * 128) / 487,
    (64 * 256) / 862,
]
x3a = [32, 64, 128, 256]
y3a = [
    (64 * 32) / 120,
    (64 * 64) / 197,
    (64 * 128) / 379,
    (64 * 256) / 745,
]

# Llama-2-13b/1024+64/PC-Low
x1b = [16, 32, 64, 128]
y1b = [
    (64 * 16) / 131,
    (64 * 32) / 208,
    (64 * 64) / 439,
    (64 * 128) / 828,
]
x2b = [16, 32, 64, 128]
y2b = [
    (64 * 16) / 187,
    (64 * 32) / 298,
    (64 * 64) / 475,
    (64 * 128) / 866,
]
x3b = [16, 32, 64, 128]
y3b = [
    (64 * 16) / 127,
    (64 * 32) / 176,
    (64 * 64) / 313,
    (64 * 128) / 666,
]

# Llama-2-13b/512+64/PC-High
x1c = [16, 32, 64, 128]
x1c = [16, 32, 64, 128, 256, 512]
y1c = [
    (64 * 16) / 60,
    (64 * 32) / 99,
    (64 * 64) / 164,
    (64 * 128) / 356,
    (64 * 256) / 784,
    (64 * 512) / 1541,
]

x2c = [16, 32, 64, 128, 256, 512]
y2c = [
    (64 * 16) / 55,
    (64 * 32) / 86,
    (64 * 64) / 157,
    (64 * 128) / 309,
    (64 * 256) / 614,
    (64 * 512) / 1241,
]

x3c = [16, 32, 64, 128, 256, 512]
y3c = [
    (64 * 16) / 38,
    (64 * 32) / 64,
    (64 * 64) / 115,
    (64 * 128) / 230,
    (64 * 256) / 460,
    (64 * 512) / 952,
]

# Llama-2-13b/1024+64/PC-High
x1d = [16, 32, 64, 128, 256]
y1d = [
    (64 * 16) / 131,
    (64 * 32) / 208,
    (64 * 64) / 439,
    (64 * 128) / 488,
    (64 * 256) / 1400,
]
x2d = [16, 32, 64, 128, 256]
y2d = [
    (64 * 16) / 87,
    (64 * 32) / 137,
    (64 * 64) / 248,
    (64 * 128) / 360,
    (64 * 256) / 659,
]
x3d = [16, 32, 64, 128, 256]
y3d = [
    (64 * 16) / 67,
    (64 * 32) / 105,
    (64 * 64) / 178,
    (64 * 128) / 260,
    (64 * 256) / 478,
]

# Opt-13b/512+64/PC-Low
x1e = [16, 32, 64, 128, 256]
y1e = [
    (64 * 16) / 163,
    (64 * 32) / 274,
    (64 * 64) / 288,
    (64 * 128) / 510,
    (64 * 256) / 1917,
]
x2e = [16, 32, 64, 128, 256]
y2e = [
    (64 * 16) / 145,
    (64 * 32) / 209,
    (64 * 64) / 313,
    (64 * 128) / 522,
    (64 * 256) / 841,
]
x3e = [16, 32, 64, 128, 256]
y3e = [
    (64 * 16) / 124,
    (64 * 32) / 164,
    (64 * 64) / 250,
    (64 * 128) / 413,
    (64 * 256) / 742,
]

# Opt-13b/1024+64/PC-Low
x1f = [16, 32, 64, 128]
y1f = [(64 * 16) / 204, (64 * 32) / 304, (64 * 64) / 503, (64 * 128) / 1500]
x2f = [16, 32, 64, 128]
y2f = [
    (64 * 16) / 244,
    (64 * 32) / 349,
    (64 * 64) / 513,
    (64 * 128) / 860,
]
x3f = [16, 32, 64, 128]
y3f = [
    (64 * 16) / 179,
    (64 * 32) / 202,
    (64 * 64) / 320,
    (64 * 128) / 714,
]

# Opt-13b/512+64/PC-High
x1g = [16, 32, 64, 128, 256]
y1g = [(64 * 16) / 95, (64 * 32) / 145, (64 * 64) / 227, (64 * 128) / 376, (64 * 256) / 699]
x2g = [16, 32, 64, 128, 256]
y2g = [(64 * 16) / 57, (64 * 32) / 96, (64 * 64) / 169, (64 * 128) / 311, (64 * 256) / 595]
x3g = [16, 32, 64, 128, 256]
y3g = [(64 * 16) / 45, (64 * 32) / 63, (64 * 64) / 119, (64 * 128) / 239, (64 * 256) / 448]

# Opt-13b/1024+64/PC-High
x1h = [16, 32, 64, 128, 256]
y1h = [
    (64 * 16) / 138,
    (64 * 32) / 267,
    (64 * 64) / 461,
    (64 * 128) / 1027,
    (64 * 256) / 1822,
]
x2h = [16, 32, 64, 128, 256]
y2h = [
    (64 * 16) / 96,
    (64 * 32) / 161,
    (64 * 64) / 310,
    (64 * 128) / 602,
    (64 * 256) / 1313,
]
x3h = [16, 32, 64, 128, 256]
y3h = [
    (64 * 16) / 67,
    (64 * 32) / 121,
    (64 * 64) / 219,
    (64 * 128) / 420,
    (64 * 256) / 881,
]


# ---------------------------
# 2) 辅助函数：标注最大 throughput
# ---------------------------
def add_max_annotation(ax, x, y, color, extra_str=None, x_offset=-2, y_offset=10):
    """在折线图上标注 y 最大值对应的点。"""
    if not y:  # 防止空列表
        return
    max_y_idx = np.argmax(y)
    max_y = y[max_y_idx]
    # 如果要在注释里拼接额外说明，可以用 f-string
    if extra_str:
        note_text = f"{max_y:.1f}({extra_str})"
    else:
        note_text = f"{max_y:.1f}"

    ax.annotate(
        note_text,
        xy=(x[max_y_idx], max_y),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=color,
    )


def add_top_annotation(ax, x, y, color, extra_str=None, line_idx=0):
    """
    在当前子图顶部(按line_idx顺序排)写出该条曲线的最大值信息。
    - line_idx: 第几条线，用来在竖直方向做微调，防止重叠。
    """
    if not y:
        return
    max_idx = np.argmax(y)
    max_val = y[max_idx]
    # 如果要在注释里拼接额外说明，可以用 f-string
    if extra_str:
        note_text = f"{max_val:.1f}({extra_str})"
    else:
        note_text = f"{max_val:.1f}"

    # 这里的 0.95 - line_idx*0.07 表示从上往下，每条线分行写
    y_pos = 1.25 - line_idx * 0.07

    ax.text(
        0.5,  # x=0.5 => 子图水平居中
        y_pos,  # 在子图顶部附近
        note_text,
        transform=ax.transAxes,  # 使用子图自己的[0,1]坐标系
        ha="center",  # 水平居中
        va="top",  # 文字对齐方式
        color=color,
        fontsize=9,  # 可自己调节字号
    )


# ---------------------------
# 3) 创建 2×4 大图
# ---------------------------
fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 可根据需要调节尺寸

# 方便索引
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[0, 2]
ax4 = axes[0, 3]
ax5 = axes[1, 0]
ax6 = axes[1, 1]
ax7 = axes[1, 2]
ax8 = axes[1, 3]

# 颜色
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # (蓝, 橙, 绿)
title_size = 10
# ---------------------------
# 第一行：Llama-2-13b 四种情形
# ---------------------------

# ----- 子图1: Llama-2-13b/512+64/PC-Low -----
line1a = ax1.plot(x1a, y1a, marker="o", label="Flexgen (CPU compute)", color=colors[0])[0]
line2a = ax1.plot(x2a, y2a, marker="s", label="Flexgen (No CPU compute)", color=colors[1])[0]
line3a = ax1.plot(x3a, y3a, marker="^", label="DynaGen", color=colors[2])[0]

# add_max_annotation(ax1, x1a, y1a, colors[0], "2*32,20,1,1,100")  # b*gbs,wg,prefetch_w,prefetch_c,cpu_del%
# add_max_annotation(ax1, x2a, y2a, colors[1], "2*128,0,1,1,0")
# add_max_annotation(ax1, x3a, y3a, colors[2], "8*32,20,2,2,50")

add_top_annotation(ax1, x1a, y1a, color=colors[0], extra_str="2*32,20,1,1,100", line_idx=2)
add_top_annotation(ax1, x2a, y2a, color=colors[1], extra_str="2*128,0,1,1,0", line_idx=1)
add_top_annotation(ax1, x3a, y3a, color=colors[2], extra_str="8*32,20,2,2,50", line_idx=0)

ax1.set_xlabel("Requests (#)")
ax1.set_ylabel("Generation Throughput (token/s)")
ax1.set_title("(a) Llama-2-13b/512+64/PC-Low", y=-0.4, fontsize=title_size)
# ax1.set_ylim(ax1.get_ylim()[0], max(y3a) * 1.1)
ax1.grid(True)

# ----- 子图2: Llama-2-13b/1024+64/PC-Low -----
line1b = ax2.plot(x1b, y1b, marker="o", color=colors[0])[0]
line2b = ax2.plot(x2b, y2b, marker="s", color=colors[1])[0]
line3b = ax2.plot(x3b, y3b, marker="^", color=colors[2])[0]

# add_max_annotation(ax2, x1b, y1b, colors[0], "1*32,20,1,1,100")
# add_max_annotation(ax2, x2b, y2b, colors[1], "8*12,20,1,1,0")
# add_max_annotation(ax2, x3b, y3b, colors[2], "2*16,20,12,12,50")

add_top_annotation(ax2, x1b, y1b, color=colors[0], extra_str="1*32,20,1,1,100", line_idx=2)
add_top_annotation(ax2, x2b, y2b, color=colors[1], extra_str="8*12,20,1,1,0", line_idx=1)
add_top_annotation(ax2, x3b, y3b, color=colors[2], extra_str="4*16,20,12,12,100", line_idx=0)

ax2.set_xlabel("Requests (#)")
ax2.set_title("(b) Llama-2-13b/1024+64/PC-Low", y=-0.4, fontsize=title_size)
# ax2.set_ylim(ax2.get_ylim()[0], max(y3b) * 1.1)
ax2.grid(True)

# ----- 子图3: Llama-2-13b/512+64/PC-High -----
line1c = ax3.plot(x1c, y1c, marker="o", color=colors[0])[0]
line2c = ax3.plot(x2c, y2c, marker="s", color=colors[1])[0]
line3c = ax3.plot(x3c, y3c, marker="^", color=colors[2])[0]

# add_max_annotation(ax3, x1c, y1c, colors[0], "2*32,70,1,1,100")
# add_max_annotation(ax3, x2c, y2c, colors[1], "8*32,65,1,1,0")
# add_max_annotation(ax3, x3c, y3c, colors[2], "4*16,80,4,8,0")

add_top_annotation(ax3, x1c, y1c, color=colors[0], extra_str="2*32,70,1,1,100", line_idx=2)
add_top_annotation(ax3, x2c, y2c, color=colors[1], extra_str="8*32,65,1,1,0", line_idx=1)
add_top_annotation(ax3, x3c, y3c, color=colors[2], extra_str="4*16,80,4,8,0", line_idx=0)

ax3.set_xlabel("Requests (#)")
ax3.set_title("(c) Llama-2-13b/512+64/PC-High", y=-0.4, fontsize=title_size)
# ax3.set_ylim(ax3.get_ylim()[0], max(y3c) * 1.1)
ax3.grid(True)

# ----- 子图4: Llama-2-13b/1024+64/PC-High -----
line1d = ax4.plot(x1d, y1d, marker="o", color=colors[0])[0]
line2d = ax4.plot(x2d, y2d, marker="s", color=colors[1])[0]
line3d = ax4.plot(x3d, y3d, marker="^", color=colors[2])[0]

# add_max_annotation(ax4, x1d, y1d, colors[0], "2*32,70,1,100")
# add_max_annotation(ax4, x2d, y2d, colors[1], "2*128,10,1,0")
# add_max_annotation(ax4, x3d, y3d, colors[2], "16*16,25,4,25")

add_top_annotation(ax4, x1d, y1d, color=colors[0], extra_str="2*32,70,1,100", line_idx=2)
add_top_annotation(ax4, x2d, y2d, color=colors[1], extra_str="2*128,10,1,0", line_idx=1)
add_top_annotation(ax4, x3d, y3d, color=colors[2], extra_str="16*16,25,4,25", line_idx=0)


ax4.set_xlabel("Requests (#)")
ax4.set_title("(d) Llama-2-13b/1024+64/PC-High", y=-0.4, fontsize=title_size)
ax4.set_ylim(ax4.get_ylim()[0], max(y3d) * 1.1)
ax4.grid(True)


# ---------------------------
# 第二行：Opt-13b 四种情形
# ---------------------------

# ----- 子图5: Opt-13b/512+64/PC-Low -----
line1e = ax5.plot(x1e, y1e, marker="o", label="Flexgen (CPU compute)", color=colors[0])[0]
line2e = ax5.plot(x2e, y2e, marker="s", label="Flexgen (No CPU compute)", color=colors[1])[0]
line3e = ax5.plot(x3e, y3e, marker="^", label="DynaGen", color=colors[2])[0]

# add_max_annotation(ax5, x1e, y1e, colors[0], "16*8,40,1,1,100")
# add_max_annotation(ax5, x2e, y2e, colors[1], "8*32,10,1,1,0")
# add_max_annotation(ax5, x3e, y3e, colors[2], "8*32,20,2,63,0")

add_top_annotation(ax5, x1e, y1e, color=colors[0], extra_str="4*32,25,1,1,100", line_idx=2)
add_top_annotation(ax5, x2e, y2e, color=colors[1], extra_str="8*32,10,1,1,0", line_idx=1)
add_top_annotation(ax5, x3e, y3e, color=colors[2], extra_str="8*32,20,2,63,0", line_idx=0)

ax5.set_xlabel("Requests (#)")
ax5.set_ylabel("Generation Throughput (token/s)")
ax5.set_title("(e) Opt-13b/512+64/PC-Low", y=-0.4, fontsize=title_size)
# ax5.set_ylim(ax5.get_ylim()[0], max(y3e) * 1.1)
ax5.grid(True)

# ----- 子图6: Opt-13b/1024+64/PC-Low -----
line1f = ax6.plot(x1f, y1f, marker="o", color=colors[0])[0]
line2f = ax6.plot(x2f, y2f, marker="s", color=colors[1])[0]
line3f = ax6.plot(x3f, y3f, marker="^", color=colors[2])[0]

# add_max_annotation(ax6, x1f, y1f, colors[0], "8*8,10,1,1,100")
# add_max_annotation(ax6, x2f, y2f, colors[1], "16*8,10,1,1,0")
# add_max_annotation(ax6, x3f, y3f, colors[2], "16*8,10,2,33,0")

add_top_annotation(ax6, x1f, y1f, color=colors[0], extra_str="8*8,30,1,1,100", line_idx=2)
add_top_annotation(ax6, x2f, y2f, color=colors[1], extra_str="8*16,20,1,1,0", line_idx=1)
add_top_annotation(ax6, x3f, y3f, color=colors[2], extra_str="4*16,20,8,16,100", line_idx=0)

ax6.set_xlabel("Requests (#)")
ax6.set_title("(f) Opt-13b/1024+64/PC-Low", y=-0.4, fontsize=title_size)
# ax6.set_ylim(ax6.get_ylim()[0], max(y3f) * 1.1)
ax6.grid(True)

# ----- 子图7: Opt-13b/512+64/PC-High -----
line1g = ax7.plot(x1g, y1g, marker="o", color=colors[0])[0]
line2g = ax7.plot(x2g, y2g, marker="s", color=colors[1])[0]
line3g = ax7.plot(x3g, y3g, marker="^", color=colors[2])[0]

# add_max_annotation(ax7, x1g, y1g, colors[0], "8*32,70,1,1,100")
# add_max_annotation(ax7, x2g, y2g, colors[1], "8*32,70,1,1,0")
# add_max_annotation(ax7, x3g, y3g, colors[2], "32*8,75,63,63,0")

add_top_annotation(ax7, x1g, y1g, color=colors[0], extra_str="8*32,70,1,1,100", line_idx=2)
add_top_annotation(ax7, x2g, y2g, color=colors[1], extra_str="8*32,70,1,1,0", line_idx=1)
add_top_annotation(ax7, x3g, y3g, color=colors[2], extra_str="32*8,75,63,63,0", line_idx=0)

ax7.set_xlabel("Requests (#)")
ax7.set_title("(g) Opt-13b/512+64/PC-High", y=-0.4, fontsize=title_size)
# ax7.set_ylim(ax7.get_ylim()[0], max(y3g) * 1.1)
ax7.grid(True)

# ----- 子图8: Opt-13b/1024+64/PC-High -----
line1h = ax8.plot(x1h, y1h, marker="o", color=colors[0])[0]
line2h = ax8.plot(x2h, y2h, marker="s", color=colors[1])[0]
line3h = ax8.plot(x3h, y3h, marker="^", color=colors[2])[0]

# add_max_annotation(ax8, x1h, y1h, colors[0], "32*8,70,1,1,100")
# add_max_annotation(ax8, x2h, y2h, colors[1], "16*8,70,1,1,0")
# add_max_annotation(ax8, x3h, y3h, colors[2], "16*8,70,70,70,0")

add_top_annotation(ax8, x1h, y1h, color=colors[0], extra_str="32*8,70,1,1,100", line_idx=2)
add_top_annotation(ax8, x2h, y2h, color=colors[1], extra_str="16*8,70,1,1,0", line_idx=1)
add_top_annotation(ax8, x3h, y3h, color=colors[2], extra_str="16*8,70,70,70,0", line_idx=0)

ax8.set_xlabel("Requests (#)")
ax8.set_title("(h) Opt-13b/1024+64/PC-High", y=-0.4, fontsize=title_size)
# ax8.set_ylim(ax8.get_ylim()[0], max(y3h) * 1.1)
ax8.grid(True)

# ---------------------------
# 4) 统一图例、保存
# ---------------------------
# 只需从第一子图里提取三条 line，即可作为全局图例
lines = [line1a, line2a, line3a]
labels = ["Flexgen (w CPU)", "Flexgen (w/o CPU)", "DynaGen"]

# 在上方（整个Figure顶端）放置图例
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
plt.subplots_adjust(wspace=0.2, hspace=0.7)

# plt.tight_layout()
plt.subplots_adjust(top=0.88)  # 给图例留一点空间
plt.savefig("throughput.png", bbox_inches="tight", dpi=300)
plt.close()
