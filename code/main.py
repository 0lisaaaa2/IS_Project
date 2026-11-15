import s_00_fetch_data, s_00_temp_fix_pipeline, s_01_blink_saccades, s_01_ica, s_01_opticat, s_02_plots, s_01_eyetracking_analysis
import config


def extended_pipeline():
    # 1. Daten laden und Blink-Fenster markieren
    raws= {}
    for run in config.runs:
        print(f"# Lade Run {run} ... ############################################################################")
        raw = s_00_fetch_data.load_data(config.subject, config.task, run)

        raw, ok = s_00_temp_fix_pipeline.correct_et_lag(raw, eye=config.eye)
        if not ok:
            print(f"Run {run} übersprungen (Sync unklar)")
            continue

        raws[run] = s_01_blink_saccades.mark_blink_periods(raw)

    # 2. ICA trainieren
    ica_model, labels = s_01_ica.fit(list(raws.values()))

    # 3. Auf jeden Run einzeln anwenden
    cleaned = {run: ica_model.apply(raw.copy()) for run, raw in raws.items()}

    # 4. Plots
    s_02_plots.main(raws, cleaned)

if __name__ == "__main__":
    extended_pipeline()


