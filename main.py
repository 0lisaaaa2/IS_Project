import s_00_fetch_data, s_00_inspect_data, s_01_ica, s_01_opticat, s_02_plots, s_00_check_eog, s_02_blink_ica_analysis
import config
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

def extended_pipeline():
    runs= []
    # Runs aus der Config laden, z. B. [1, 2, ..., 8]
    for run_number in config.runs:
        print(f"# Lade Run {run_number} ... ############################################################################")

        raw = s_00_fetch_data.load_data(
            config.subject,
            config.task,
            run_number
        )
        #raw.plot(block=True)

        s_00_inspect_data.inspect_data(raw)

        # regular ica
        #clean_raw, n_excluded, ica, labels = s_01_ica.ica(raw)
        #clean_raw.plot(block=True)

        # opticat ica
        clean_raw_opticat, ica, n_excluded_opticat, labels = s_01_opticat.opticat(raw)
        #clean_raw_opticat.plot(block=True)
        
        # blink analysis
        s_02_blink_ica_analysis.analyze_blink_ica(raw, ica, clean_raw_opticat, labels)

        #runs.append((f"Run {run_number}", clean_raw_opticat))

        #s_02_plots.main(raw) # topoplots

    # Alle Raw-Runs gemeinsam plotten
    s_02_plots.plot_saccade_frps_raw_runs(runs)


if __name__ == "__main__":
    extended_pipeline()


