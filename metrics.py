# metrics.py
# Author: Mike Zhang
# Date: 22/12-2021
# Calculate weighted macro-F1 for each model

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, f1_score


if __name__ == "__main__":
    model_perf = []
    model_std = []
    gold = []

    models = {"bert"        : "BERT (EN)",
              "jobbert"     : "JobBERT (EN)",
              "rembert_en"  : "RemBERT (EN)",
              "dabert"      : "DaBERT (DA)",
              "dajobbert"   : "DaJobBERT (DA)",
              "rembert_da"  : "RemBERT (DA)",
              "rembert_enda": "RemBERT (EN+DA)"}
    with open("results/silver_en_dev.tsv") as g:  # change for other split
        for line in g:
            span, silver = line.strip().split("\t")
            gold.append(silver)
        for model in models:
            metric = []
            for file in os.listdir(f"preds_en_dev/{model}/"):  # change for other split
                preds = []
                if file.endswith("out"):
                    with open(f"preds_en_dev/{model}/" + file) as f:  # change for other split
                        for line in f:
                            _, pred = line.strip().split("\t")
                            preds.append(pred)
                            metric.append(f1_score(gold, preds, average="weighted"))
                        if file.startswith("1"):
                            plot = ConfusionMatrixDisplay.from_predictions(gold, preds, xticks_rotation=45.0,
                                                                           colorbar=False)
                            plot.ax_.set_xlabel("Predicted Label", alpha=.6, fontsize='x-large')
                            plot.ax_.set_ylabel("True Label", alpha=.6, fontsize='x-large')
                            plot.ax_.set_title(f"{models[model]}", alpha=.6, fontsize='x-large')
                            plot.figure_.set_size_inches(8.5, 8.5)
                            plt.tight_layout()
                            plt.savefig(f"matrix_{model}.pdf", format="pdf", dpi=300, bbox_inches="tight")

            print(f"{model}: {metric}")
            model_perf.append(np.mean(metric))
            model_std.append(np.std(metric))

    print(f"All models macro-F1: {model_perf}")
    print(f"All models Standard Dev.: {model_std)")
