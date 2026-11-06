
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_saccade_blinks(raw, window=0.2, plot=True):
    """
    Prüft die zeitliche Beziehung zwischen ET_Saccade L
    und ET_Blink L/R Events.

    Zusätzlich:
    - Unterscheidet Blinks vor und nach Sakkaden
    - Analysiert die Richtung innerhalb des Analysefensters
    - Visualisiert die zeitliche Verteilung (-500 bis +500 ms)
    - Analysiert die Dauer der Blink-Annotations

    Parameters
    ----------
    raw : mne.io.Raw
        EEG-Daten mit Annotations.
    window : float
        Zeitfenster in Sekunden für potenzielle Überlappungen.
    plot : bool
        Erstellt ein Histogramm der zeitlichen Abstände.

    Returns
    -------
    results : pandas.DataFrame
        Zeitlicher Abstand zum nächsten Blink pro Saccade.
    """

    events, event_id = mne.events_from_annotations(raw)

    inv_event_id = {v: k for k, v in event_id.items()}

    def get_event_samples(event_name):
        return events[
            np.array([
                inv_event_id[e] == event_name
                for e in events[:, 2]
            ])
        ][:, 0]

    # --------------------------------------------------
    # 1. Events auslesen
    # --------------------------------------------------

    saccades = get_event_samples("ET_Saccade L")
    blink_L = get_event_samples("ET_Blink L")
    blink_R = get_event_samples("ET_Blink R")

    blinks = np.sort(np.concatenate([blink_L, blink_R]))

    if len(saccades) == 0:
        raise ValueError("Keine ET_Saccade L Events gefunden.")

    if len(blinks) == 0:
        raise ValueError("Keine ET_Blink Events gefunden.")

    sfreq = raw.info["sfreq"]

    # --------------------------------------------------
    # 2. Abstand zum nächsten Blink
    # --------------------------------------------------

    distances = np.min(
        np.abs(saccades[:, None] - blinks[None, :]),
        axis=1
    ) / sfreq

    # Vorzeichenbehafteter Abstand
    signed_distances = []

    for sac in saccades:
        differences = (blinks - sac) / sfreq
        nearest = differences[np.argmin(np.abs(differences))]
        signed_distances.append(nearest)

    signed_distances = np.array(signed_distances)

    # --------------------------------------------------
    # 3. Ergebnisse erstellen
    # --------------------------------------------------

    results = pd.DataFrame({
        "Saccade_sample": saccades,
        "Saccade_time": saccades / sfreq,
        "Nearest_Blink_Distance_s": distances,
        "Blink_relative_to_Saccade_s": signed_distances,
        "Potential_Blink_Overlap": distances <= window
    })

    # --------------------------------------------------
    # 4. Allgemeine Zusammenfassung
    # --------------------------------------------------

    n_total = len(saccades)
    n_overlap = np.sum(distances <= window)

    print("\n--- Saccade/Blink Analysis ---")
    print(f"Total Saccade L events: {n_total}")

    for w in [0.05, 0.1, 0.2, 0.3, 0.5]:
        n = np.sum(distances <= w)

        print(
            f"Blink within ±{w*1000:.0f} ms: "
            f"{n}/{n_total} ({n/n_total*100:.1f}%)"
        )

    print(
        f"\nPotential overlaps ({window*1000:.0f} ms): "
        f"{n_overlap}/{n_total}"
    )

    # --------------------------------------------------
    # 5. Zeitliche Richtung aller Events
    # --------------------------------------------------

    n_before = np.sum(signed_distances < 0)
    n_after = np.sum(signed_distances > 0)
    n_exact = np.sum(signed_distances == 0)

    print("\n--- Temporal Direction (all events) ---")
    print(f"Blink before Saccade: {n_before}")
    print(f"Blink after Saccade: {n_after}")
    print(f"Same onset: {n_exact}")

    # --------------------------------------------------
    # 6. Zeitliche Richtung innerhalb des Fensters
    # --------------------------------------------------

    within_window = np.abs(signed_distances) <= window

    blinks_before = np.sum(
        (signed_distances < 0) & within_window
    )

    blinks_after = np.sum(
        (signed_distances > 0) & within_window
    )

    blinks_exact = np.sum(
        (signed_distances == 0) & within_window
    )

    print(
        f"\n--- Direction within ±{window*1000:.0f} ms ---"
    )

    print(f"Blinks before Saccade: {blinks_before}")
    print(f"Blinks after Saccade: {blinks_after}")
    print(f"Same onset: {blinks_exact}")
    print(
        f"Total: {blinks_before + blinks_after + blinks_exact}"
    )

    # --------------------------------------------------
    # 7. Dauer der Blink-Annotations untersuchen
    # --------------------------------------------------

    blink_durations = np.array([
        duration
        for description, duration
        in zip(
            raw.annotations.description,
            raw.annotations.duration
        )
        if description in ["ET_Blink L", "ET_Blink R"]
    ])

    print("\n--- Blink Annotation Durations ---")

    if len(blink_durations) > 0:

        print(f"Number of blink annotations: {len(blink_durations)}")
        print(f"Mean duration: {np.mean(blink_durations):.3f} s")
        print(f"Median duration: {np.median(blink_durations):.3f} s")
        print(f"Minimum duration: {np.min(blink_durations):.3f} s")
        print(f"Maximum duration: {np.max(blink_durations):.3f} s")

    else:
        print("No blink annotation durations found.")

    # --------------------------------------------------
    # 8. Histogramm: Detailansicht ±500 ms
    # --------------------------------------------------

    if plot:

        plot_range = 0.5

        # Nur Abstände innerhalb ±500 ms
        plot_distances = signed_distances[
            np.abs(signed_distances) <= plot_range
        ]

        plt.figure(figsize=(9, 4))

        plt.hist(
            plot_distances * 1000,
            bins=np.arange(-500, 526, 25)
        )

        plt.axvline(
            0,
            color="red",
            linestyle="--",
            label="Saccade onset"
        )

        plt.axvline(
            -window * 1000,
            color="gray",
            linestyle=":",
            label=f"±{window*1000:.0f} ms"
        )

        plt.axvline(
            window * 1000,
            color="gray",
            linestyle=":"
        )

        plt.xlim(-500, 500)

        plt.xlabel(
            "Blink onset relative to Saccade onset (ms)"
        )

        plt.ylabel("Number of events")

        plt.title(
            "Temporal relationship: Saccade L and Blink"
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    return results
