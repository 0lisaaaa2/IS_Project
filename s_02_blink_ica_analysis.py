
import mne
import numpy as np


def analyze_blink_ica(
    raw,
    ica,
    raw_clean,
    tmin=-0.5,
    tmax=0.5,
    top_n_components=10
):
    """
    Untersucht den Zusammenhang zwischen linken Blink-Annotationen
    und ICA-Komponenten.

    Parameters
    ----------
    raw : mne.io.Raw
        EEG vor ICA-Korrektur, mit ET-Blink-Annotationen.

    ica : mne.preprocessing.ICA
        Bereits gefittetes ICA-Objekt.

    raw_clean : mne.io.Raw
        EEG nach ICA-Korrektur.

    tmin, tmax : float
        Zeitfenster relativ zum Blink-Onset.

    top_n_components : int
        Anzahl der ICA-Komponenten, die in der Detailansicht
        dargestellt werden.
    """

    # --------------------------------------------------
    # 1. BLINK-EVENTS EXTRAHIEREN
    # --------------------------------------------------

    print("\n--- BLINK-ANALYSE ---")

    events, event_id = mne.events_from_annotations(
        raw,
        regexp=r"^ET_Blink L$",
        verbose=False
    )

    if not event_id or len(events) == 0:
        raise ValueError("Keine ET_Blink L Events gefunden.")

    blink_event_id = event_id["ET_Blink L"]

    print("Blink-Annotation:", "ET_Blink L")
    print("Anzahl linker Blinks:", len(events))

    # --------------------------------------------------
    # 2. DATEN KONTROLLIEREN
    # --------------------------------------------------

    if raw.info["sfreq"] != raw_clean.info["sfreq"]:
        raise ValueError("Sampling Rates stimmen nicht überein.")

    if raw.first_samp != raw_clean.first_samp:
        raise ValueError("First Samples stimmen nicht überein.")

    eeg_channels = [
        ch for ch, typ in zip(
            raw.ch_names,
            raw.get_channel_types()
        )
        if typ == "eeg"
    ]

    if not set(eeg_channels).issubset(raw_clean.ch_names):
        raise ValueError("EEG-Kanäle stimmen nicht überein.")

    print("EEG-Kanäle:", len(eeg_channels))
    print("Sampling Rate:", raw.info["sfreq"], "Hz")

    # --------------------------------------------------
    # 3. EEG-EPOCHEN VOR ICA
    # --------------------------------------------------

    epochs_before = mne.Epochs(
        raw,
        events,
        event_id={"ET_Blink L": blink_event_id},
        tmin=tmin,
        tmax=tmax,
        picks=eeg_channels,
        baseline=None,
        preload=True,
        reject_by_annotation=False
    )

    print("\nEpochs vor ICA:", len(epochs_before))

    # Gemitteltes EEG vor ICA
    evoked_before = epochs_before.average()

    # --------------------------------------------------
    # 4. ICA-QUELLEN BERECHNEN
    # --------------------------------------------------

    ica_sources = ica.get_sources(raw)

    epochs_ica = mne.Epochs(
        ica_sources,
        events,
        event_id={"ET_Blink L": blink_event_id},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=False
    )

    print("ICA-Epochen:", len(epochs_ica))

    # Gemittelte ICA-Aktivität
    evoked_ica = epochs_ica.average(picks="all")

    # --------------------------------------------------
    # 5. ICA-KOMPONENTEN NACH BLINK-AKTIVITÄT RANKEN
    # --------------------------------------------------

    ica_data = epochs_ica.get_data()

    # Mittelwert über alle Blink-Epochen
    mean_ica = ica_data.mean(axis=0)

    # Peak-to-Peak-Amplitude pro Komponente
    blink_amplitudes = np.ptp(mean_ica, axis=1)

    # Komponenten mit stärkster Blink-Aktivität
    top_components = np.argsort(
        blink_amplitudes
    )[::-1][:top_n_components]

    print("\n--- ICA-KOMPONENTEN ---")

    for component in top_components:
        print(
            f"IC {component}: "
            f"{blink_amplitudes[component]:.3f}"
        )

    # --------------------------------------------------
    # 6. EEG-EPOCHEN NACH ICA
    # --------------------------------------------------

    epochs_after = mne.Epochs(
        raw_clean,
        events,
        event_id={"ET_Blink L": blink_event_id},
        tmin=tmin,
        tmax=tmax,
        picks=eeg_channels,
        baseline=None,
        preload=True,
        reject_by_annotation=False
    )

    print("\nEpochs nach ICA:", len(epochs_after))

    evoked_after = epochs_after.average()

    # --------------------------------------------------
    # 7. PLOT 1: EEG VOR ICA
    # --------------------------------------------------

    print("\nPlot 1: EEG vor ICA")

    evoked_before.plot_joint(
        title="EEG vor ICA – linke Blinks"
    )

    # --------------------------------------------------
    # 8. PLOT 2: ICA-KOMPONENTEN
    # --------------------------------------------------

    print("\nPlot 2: ICA-Komponenten")

    # Anzahl der tatsächlich gefitteten ICA-Komponenten
    n_components = ica.n_components_

    print("Anzahl ICA-Komponenten:", n_components)

    # Alle Komponenten auswählen
    all_components = np.arange(n_components)

    # Alle Komponenten plotten
    evoked_ica.plot(
        picks=all_components,
        spatial_colors=False,
        show=True
    )

    # evoked_ica.plot(
    #     picks=top_components,
    #     spatial_colors=False,
    #     show=True
    # )

    # # Zusätzliche Darstellung der einzelnen Blink-Epochen
    # epochs_ica.plot_image(
    #     picks=top_components
    # )

    

    # --------------------------------------------------
    # 9. PLOT 3: EEG NACH ICA
    # --------------------------------------------------

    print("\nPlot 3: EEG nach ICA")

    evoked_after.plot_joint(
        title="EEG nach ICA – linke Blinks"
    )

    # --------------------------------------------------
    # 10. ERGEBNISSE ZURÜCKGEBEN
    # --------------------------------------------------

    return {
        "events": events,
        "epochs_before": epochs_before,
        "epochs_ica": epochs_ica,
        "epochs_after": epochs_after,
        "evoked_before": evoked_before,
        "evoked_ica": evoked_ica,
        "evoked_after": evoked_after,
        "top_components": top_components,
        "blink_amplitudes": blink_amplitudes
    }
