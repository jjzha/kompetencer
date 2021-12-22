# significance.py
# Author: Mike Zhang
# Date: 22/12-2021
# Significance testing with the deepsig package

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from deepsig import multi_aso

N = 5  # Number of random seeds
M = 7  # Number of different models / algorithms

models = ["BERT (EN)", "JobBERT (EN)", "RemBERT (EN)", "DaBERT (DA)", "DaJobBERT (DA)", "RemBERT (DA)",
          "RemBERT (EN+DA)"]

# Following values are obtained from metrics.py

dev = {"BERT"         : np.array(
        [0.6297771671570598, 0.6329352540487885, 0.6312882015804178, 0.6210302431711945, 0.6267191552056328]),
        "JobBERT"     : np.array(
                [0.6232580568597211, 0.6262642409013168, 0.6253847383732379, 0.638634446211839, 0.6245914816443906]),
        "RemBERT_en"  : np.array(
                [0.6251099675734428, 0.6319119527032638, 0.6253851437768524, 0.6307054998098834, 0.6320935526303988]),
        "DaBERT"      : np.array([0.10038673890878519, 0.08055841708707516, 0.10728388468492062, 0.07580735080954466,
                                  0.07449671931041886]),
        "DaJobBERT"   : np.array(
                [0.10189371338241245, 0.13294492221325102, 0.11960917128278162, 0.0657135996823695,
                 0.0832038620478267]),
        "RemBERT_da"  : np.array([0.07322654462242563, 0.18816230774268838, 0.17046353985914917, 0.07322654462242563,
                                  0.07322654462242563]),
        "RemBERT_enda": np.array(
                [0.6204651237307777, 0.6291735328096579, 0.6320774893934726, 0.6262208654111966, 0.6372382512541911])
        }

test = {"BERT"        : np.array(
        [0.6262736175315267, 0.628787223206733, 0.6433547750074539, 0.6363162641585185, 0.6246125733071723]),
        "JobBERT"     : np.array(
                [0.6518750492327318, 0.6347867264779783, 0.6475328183143162, 0.6459319100477973, 0.6374378359347097]),
        "RemBERT_en"  : np.array(
                [0.6273869010608704, 0.649261013320665, 0.6362351538062523, 0.6369495561246475, 0.6367404662522315]),
        "DaBERT"      : np.array([0.09078185663931468, 0.06897717196370758, 0.09024946531730611, 0.06744948036297804,
                                  0.06479044391871576]),
        "DaJobBERT"   : np.array([0.08912297325979454, 0.1303706645868277, 0.11749303995151247, 0.06934522112832897,
                                  0.07482739118888895]),
        "RemBERT_da"  : np.array([0.06566291264270285, 0.15317962743365493, 0.13905868336042412, 0.06566291264270285,
                                  0.06566291264270285]),
        "RemBERT_enda": np.array(
                [0.639795845559789, 0.6524257466800424, 0.6382466834949903, 0.6472410371000864, 0.6373064653199011])
        }

da_test = {"BERT"     : np.array(
        [0.03899208871345757, 0.03952929323460853, 0.04719741907218171, 0.0222944137486076, 0.039764771834839716]),
        "JobBERT"     : np.array(
                [0.07195948978943609, 0.06197466638419256, 0.05925680218641819, 0.06681044323663586,
                 0.056753613128418985]),
        "RemBERT_en"  : np.array(
                [0.3848347497267764, 0.3686730674037911, 0.3547841970432886, 0.33449888815980505, 0.3291513166539705]),
        "DaBERT"      : np.array([0.2540115340660215, 0.2740112680630302, 0.17622120779504208, 0.11519191735409537,
                                  0.17528179671156854]),
        "DaJobBERT"   : np.array(
                [0.3994241101446081, 0.43047735478040794, 0.3872774994072669, 0.38728748282742276,
                 0.36836947306208384]),
        "RemBERT_da"  : np.array(
                [0.46255679422221896, 0.4711952160862969, 0.4865238172905017, 0.45205059411044546, 0.48676077443202465]
                ),
        "RemBERT_enda": np.array([0.45153061224489793, 0.4642857142857143, 0.48086734693877553, 0.44642857142857145,
                                  0.48341836734693877])
        }

for i in [dev, test, da_test]:
    eps_min_test = multi_aso(dev, confidence_level=0.05, num_jobs=16, return_df=True, use_symmetry=True)
    eps_min_test = eps_min_test.to_numpy()  # transform back to numpy for ordering

    idx = 0
    for i in range(len(eps_min_test)):
        eps_min_test[i, idx] = 2  # dumb solution for masking
        idx += 1
    new_eps_min_test = np.ma.masked_where(eps_min_test == 2, eps_min_test)  # mask out diagonal

    fig, ax = plt.subplots()
    cmap = cmr.get_sub_cmap('Greens_r', 0.2, 1.0)
    im = ax.imshow(new_eps_min_test, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap)
    cbar.ax.set_ylabel("e_min", rotation=-90, va="bottom", alpha=.6)

    # make white grid lines
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(new_eps_min_test.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(new_eps_min_test.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(test)), labels=models.values())
    ax.set_yticks(np.arange(len(test)), labels=models.values())

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(test)):
        for j in range(len(test)):
            if isinstance(new_eps_min_test[i, j], np.float):
                new_eps_min_test[i, j] = np.round(new_eps_min_test[i, j], 2)
                text = ax.text(j, i, new_eps_min_test[i, j], ha="center", va="center", color="black")

    ax.set_title("Significance Testing", alpha=0.6)
    fig.tight_layout()
    plt.savefig("matrix_significance.pdf", format="pdf", dpi=300, bbox_inches="tight")
