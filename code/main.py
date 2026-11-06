import s_00_fetch_data, s_01_blink_saccades, s_01_ica, s_01_opticat, s_02_plots
import config

def extended_pipeline():
    # 1. Daten laden und Blink-Fenster markieren
    raws= {}
    for run in config.runs:
        print(f"# Lade Run {run} ... ############################################################################")
        raw = s_00_fetch_data.load_data(config.subject, config.task, run)
        raws[run] = s_01_blink_saccades.mark_blink_periods(raw)

    # 2. ICA trainieren
    if config.opticat == True:
        ica_model, labels = s_01_opticat.fit(list(raws.values()))  # ICA mit Opticat
    else:
        ica_model, labels = s_01_ica.fit(list(raws.values())) # normales ICA

    # 3. Auf jeden Run einzeln anwenden
    cleaned = {run: ica_model.apply(raw.copy()) for run, raw in raws.items()}

    # 4. FRPs
    s_02_plots.plot_saccade_frps_raw_runs(
        [(f"Run {r}", raw) for r, raw in cleaned.items()])

if __name__ == "__main__":
    extended_pipeline()


