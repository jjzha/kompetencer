import collections
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def results_plot():
    # The following results are obtained from metrics.py
    en_dev = {"f1_macro": [0.6283500042326187, 0.6276265927981012, 0.6290412232987682, 0.08770662216014889,
                           0.10067305372172827, 0.11566109629382289, 0.6290350525198593],
              "std"     : [0.004193635274779164, 0.005591972043771065, 0.0031352609784990404, 0.013499967381176974,
                           0.024202793800290674, 0.05227199443094042, 0.005620167202637677]
              }
    en_test = {"f1_macro": [0.6318688906422809, 0.6435128680015066, 0.6373146181129334, 0.07644968364040443,
                            0.09623185802307052, 0.09784540974443752, 0.6430031556309619],
               "std"     : [0.007002322007421526, 0.006402958697106986, 0.006969994090800549, 0.011563953630848006,
                            0.02387079662321612, 0.03966749028747183, 0.005866992534469711]

               }
    da_test = {"f1_macro": [0.03755559732073902, 0.06335100294502033, 0.3543884437975263, 0.19894354479795157,
                            0.3945671840443579, 0.165731744448591, 0.47181743922829755],
               "std"     : [0.008206191392167491, 0.005445293517421018, 0.020802363552575878, 0.057893288970488496,
                            0.020521129140777836, 0.141419579759643, 0.013538505797905703]

               }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")
    x_labels = ["BERT (EN)", "JobBERT (EN)", "RemBERT (EN)", "DaBERT (DA)", "DaJobBERT (DA)", "RemBERT (DA)",
                "RemBERT (EN+DA)"]
    x = np.arange(len(x_labels))

    width = 0.4
    data_labels = ["Dev (EN)", "Test (EN)", "Test (DA)"]
    colors = ["steelblue", "teal", "lightsalmon"]
    edgecolors = ["skyblue", "mediumturquoise", "orangered"]
    hatches = ["..", "//", "\\\\"]

    ax.bar(x - width / 2, en_dev["f1_macro"], label="dev", width=0.2, color=colors[0], hatch=hatches[0],
           edgecolor=edgecolors[0], linewidth=.5)
    ax.errorbar(
            x - width / 2, en_dev["f1_macro"], yerr=en_dev["std"],
            color="black", fmt="_", alpha=1., linestyle='', linewidth=1,
            solid_capstyle="projecting", capsize=3.5, capthick=1
            )

    ax.bar(x, en_test["f1_macro"], label="test", width=0.2, color=colors[1], hatch=hatches[1], edgecolor=edgecolors[1],
           linewidth=.5)
    ax.errorbar(
            x, en_test["f1_macro"], yerr=en_test["std"],
            color="black", fmt="_", alpha=1., linestyle='', linewidth=1,
            solid_capstyle="projecting", capsize=3.5, capthick=1
            )

    ax.bar(x + width / 2, da_test["f1_macro"], label="test", width=0.2, color=colors[2], hatch=hatches[2],
           edgecolor=edgecolors[2],
           linewidth=.5)
    ax.errorbar(
            x + width / 2, da_test["f1_macro"], yerr=da_test["std"],
            color="black", fmt="_", alpha=1., linestyle='', linewidth=1,
            solid_capstyle="projecting", capsize=3.5, capthick=1
            )

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")

    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=1, fc="white")
    func_annotate = lambda text, xyc, bbc: ax.annotate(text,
                                                       xy=xyc,
                                                       xycoords="axes fraction",
                                                       fontsize=9,
                                                       color="black",
                                                       va="center",
                                                       ha="center",
                                                       rotation=360,
                                                       bbox=bb(bbc))
    func_annotate("Zero-Shot", xyc=(0.23, 1.025), bbc="black")
    func_annotate("Few-Shot", xyc=(0.705, 1.025), bbc="black")

    # prepare y-axis
    ax.set_ylabel("Weighted macro-F1", alpha=.6)
    ax.legend(labels=data_labels, handles=handles, loc="upper center")
    ax.axvline(x=2.5, c="black")

    fig.tight_layout()
    # plt.show()
    plt.savefig("results_macro.pdf", format="pdf", dpi=300, bbox_inches="tight")


def prior_class_distribution():
    classes = defaultdict(list)
    uniq_labels = set()
    for file in os.listdir("labels"):
        with open(f"labels/{file}") as f:  # will be released upon acceptance
            for line in f:
                _, cls = line.strip().split("\t")
                uniq_labels.add(cls)
    for file in os.listdir("labels"):
        print(file)
        cls_l = []
        with open(f"labels/{file}") as f:
            for line in f:
                _, cls = line.strip().split("\t")
                cls_l.append(cls)
            count = collections.Counter(cls_l)
            for label in uniq_labels:
                if label in count:
                    classes[label].append(count[label])
                else:
                    classes[label].append(0)
    ord = collections.OrderedDict(sorted(classes.items()))

    train_da = []
    train_en = []
    dev_en = []

    for cls, values in ord.items():
        train_da.append(values[0])
        train_en.append(values[1])
        dev_en.append(values[2])

    fig, ax = plt.subplots(figsize=(10, 5), nrows=3)

    for a in ax:
        a.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    lbs = ["0000", "A1", "A2", "K00", "K01", "K02", "K03", "K04", "K05", "K06", "K07", "K08", "K09", "K10", "K99", "L1",
           "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "K?", "S?"]
    width = 0.6
    data_labels = ["Train (EN)", "Dev (EN)", "Train (DA)"]
    colors = ["orangered", "darkmagenta", "plum"]
    edgecolors = ["lightsalmon", "plum", "darkmagenta"]
    hatches = ["..", "//", "\\\\"]

    for i, data in enumerate([train_en, dev_en, train_da]):
        ax[i].bar(np.arange(len(ord)), data, label="dev", width=width, color=colors[i], hatch=hatches[i],
                  edgecolor=edgecolors[i], linewidth=.5)
        ax[i].set_xticks(np.arange(len(ord)))
        ax[i].set_xticklabels(["" for i in range(len(ord))], rotation=40, ha="right")
        # prepare y-axis
        ax[i].legend(labels=[data_labels[i]])

    ax[1].set_ylabel("Count", alpha=.6)
    ax[-1].set_xticklabels(lbs, rotation=40, ha="right")
    fig.tight_layout()
    plt.savefig("label_distribution.pdf", format="pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    results_plot()
    prior_class_distribution()
