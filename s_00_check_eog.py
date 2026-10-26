import numpy as np
from itertools import combinations
import mne
import matplotlib.pyplot as plt
import pandas as pd


def compare_eog_eeg(raw):

    # raw: dein bereits geladenes MNE-Raw-Objekt
    eeg = raw.get_data(picks="eeg")
    eog = raw.get_data(picks=["HEOG", "VEOG"])

    eeg_names = raw.copy().pick("eeg").ch_names

    for eog_idx, eog_name in enumerate(["HEOG", "VEOG"]):
        target = eog[eog_idx]
        results = []

        for i, j in combinations(range(len(eeg_names)), 2):
            diff = eeg[i] - eeg[j]

            # Korrelation unabhängig von der Polarität
            corr = np.corrcoef(target, diff)[0, 1]
            results.append(
                (abs(corr), corr, eeg_names[i], eeg_names[j])
            )

        results.sort(reverse=True)

        print(f"\n{eog_name}:")
        for abs_corr, corr, ch1, ch2 in results[:10]:
            print(
                f"{ch1} - {ch2}: "
                f"r = {corr:.5f}"
            )

def check_reference(raw):
    print(raw.info)
    print("\nReferenzstatus:")
    print("custom_ref_applied:", raw.info["custom_ref_applied"])

    print("\nEEG-Projektoren:")
    for proj in raw.info["projs"]:
        print(proj)

    print("\nKanalinformationen:")
    for ch in raw.info["chs"][:5]:
        print(ch["ch_name"], ch.get("loc", [])[:3])

def check_origin(raw):
    print("Dateiname:", raw.filenames)
    print("Beschreibung:", raw.info.get("description"))
    print("Experiment:", raw.info.get("experimenter"))
    print("Projekt-ID:", raw.info.get("proj_id"))
    print("Akquisitionsparameter:", raw.info.get("acq_pars"))
    print("Referenzstatus:", raw.info["custom_ref_applied"])


def check_konstant(raw):
    for name in raw.copy().pick("eeg").ch_names:
        x = raw.get_data(picks=[name])[0]
        if np.std(x) < 1e-12:
            print("Nahezu flacher EEG-Kanal:", name)


def plot_raw_electrodes_3d(raw, show_names=True):
    fig = mne.viz.plot_sensors(
        info=raw.info,
        kind="3d",
        ch_type="eeg",
        show_names=show_names,
        show=False
    )

    plt.show()