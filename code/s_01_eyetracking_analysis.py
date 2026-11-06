"""
Kanaäle

L POR X [px] Horizontale Position des linken Auges in Pixeln
L POR Y [px] Vertikale Position des linken Auges in Pixeln
L Mapped Diameter [mm] Durchmesser der Pupille des linken Auges in Millimetern
L Validaty Validität der Messung des linken Auges (0 = ungültig, 1 = gültig)
Pupil Confidence Konfidenz der Pupillenerkennung

"""


import mne
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# normaler mne plot zum scrollen
def plot_left_eye_events_scroll(raw, duration=20):
    """
    Scrollbare Visualisierung der Eye-Tracking-Daten
    mit linken Blink- und Sakkaden-Annotationen.
    """

    print("\n--- EYE-TRACKING ANALYSE ---")

    # ---------------------------------------------
    # 1. RELEVANTE ANNOTATIONS AUSWÄHLEN
    # ---------------------------------------------

    relevant_types = [
        "ET_Blink L",
        "ET_Saccade L"
    ]

    annotation_mask = np.isin(
        raw.annotations.description,
        relevant_types
    )

    selected_annotations = raw.annotations[annotation_mask]

    print("\n--- ANNOTATIONEN ---")

    counts = Counter(selected_annotations.description)

    for annotation_type, count in counts.items():
        print(f"{annotation_type}: {count}")

    if len(selected_annotations) == 0:
        raise ValueError(
            "Keine ET_Blink L oder ET_Saccade L Annotationen gefunden."
        )

    # ---------------------------------------------
    # 2. EIGENE RAW-KOPIE MIT NUR DIESEN ANNOTATIONS
    # ---------------------------------------------

    raw_eye = raw.copy()

    raw_eye.set_annotations(selected_annotations)

    # Events aus den Annotationen erstellen
    events, event_id = mne.events_from_annotations(
        raw_eye,
        regexp=r"^ET_(Blink|Saccade) L$",
        verbose=False
    )

    print("\nEvent IDs:", event_id)
    print("Gesamtzahl Events:", len(events))

    # ---------------------------------------------
    # 3. RELEVANTE EYE-TRACKING-KANÄLE
    # ---------------------------------------------

    gaze_channels = [
        "L POR X [px]",
        "L POR Y [px]",
        "L Mapped Diameter [mm]",
        "L Validity",
        "Pupil Confidence"
    ]

    # Prüfen, welche Kanäle vorhanden sind
    available_channels = [
        ch for ch in gaze_channels
        if ch in raw_eye.ch_names
    ]

    missing_channels = [
        ch for ch in gaze_channels
        if ch not in raw_eye.ch_names
    ]

    if missing_channels:
        print("Fehlende Kanäle:", missing_channels)

    if not available_channels:
        raise ValueError("Keine relevanten Eye-Tracking-Kanäle gefunden.")

    print("\nGeplottete Kanäle:", available_channels)

    # ---------------------------------------------
    # 4. SCROLL-PLOT: BLICKPOSITION
    # ---------------------------------------------

    raw_eye.plot(
        picks=available_channels,
        events=events,
        event_id=event_id,
        duration=duration,
        n_channels=len(available_channels),
        scalings="auto",
        remove_dc=False,
        show_scrollbars=True,
        title="Eye Tracking – linke Blinks und Sakkaden",
        show=True
    )

    # # pupillendurchmesser
    # # bei blink events ausfall
    # # bei sakkaden stabil
    # raw_eye.plot(
    #     picks=["L Mapped Diameter [mm]"],
    #     events=events,
    #     event_id=event_id,
    #     duration=duration,
    #     scalings="auto",
    #     remove_dc=False,
    #     title="Linkes Auge – Pupillendurchmesser",
    #     show=True
    # )

    # # messqualität
    # # bei sakkaden gültig, bei blinks häufig ungültig
    # raw_eye.plot(
    #     picks=["L Validity"],
    #     events=events,
    #     event_id=event_id,
    #     duration=duration,
    #     scalings="auto",
    #     remove_dc=False,
    #     title="Linkes Auge – Validity",
    #     show=True
    # )


    return events, event_id



# matplotlib plot, damit skalen mehr separat 
# duration: wie viele sekunden gleichzeitig im diagram zu sehen sind (wie nah rangezoomt an daten. e.g. =2: 0-2s, 2-4s, etc.)
# timestep: Legt fest, in welchen Abständen die Beschriftungen der X-Achse erscheinen.
def plot_left_eye_events_plt(raw, duration=10, time_step=0.25):

    print("\n--- EYE-TRACKING ANALYSE ---")

    # --------------------------------------------------
    # 1. Annotationen des linken Auges auswählen
    # --------------------------------------------------

    relevant_types = ["ET_Blink L", "ET_Saccade L"]

    annotation_mask = np.isin(
        raw.annotations.description,
        relevant_types
    )

    selected_annotations = raw.annotations[annotation_mask]

    print("\n--- ANNOTATIONEN ---")

    counts = Counter(selected_annotations.description)

    for annotation_type, count in counts.items():
        print(f"{annotation_type}: {count}")

    if len(selected_annotations) == 0:
        raise ValueError(
            "Keine ET_Blink L oder ET_Saccade L Annotationen gefunden."
        )

    # --------------------------------------------------
    # 2. Annotationen vorbereiten
    # --------------------------------------------------

    raw_eye = raw.copy()
    raw_eye.set_annotations(selected_annotations)

    events, event_id = mne.events_from_annotations(
        raw_eye,
        regexp=r"^ET_(Blink|Saccade) L$",
        verbose=False
    )

    print("\nEvent IDs:", event_id)
    print("Gesamtzahl Events:", len(events))

    sfreq = raw_eye.info["sfreq"]
    first_time = raw_eye.first_time

    # Event-Dauern
    blink_events = []
    saccade_events = []

    for annotation in selected_annotations:

        onset = annotation["onset"] - first_time
        offset = onset + annotation["duration"]

        event = {
            "onset": onset,
            "offset": offset,
            "duration": annotation["duration"]
        }

        if annotation["description"] == "ET_Blink L":
            blink_events.append(event)

        elif annotation["description"] == "ET_Saccade L":
            saccade_events.append(event)


    # --------------------------------------------------
    # 4. Eye-Tracking-Kanäle auswählen
    # --------------------------------------------------

    gaze_channels = [
        "L POR X [px]",
        "L POR Y [px]",
        "L Mapped Diameter [mm]",
        "L Validity",
        "Pupil Confidence"
    ]

    missing_channels = [
        ch for ch in gaze_channels
        if ch not in raw_eye.ch_names
    ]

    if missing_channels:
        raise ValueError(
            f"Fehlende Kanäle: {missing_channels}"
        )

    data = raw_eye.get_data(picks=gaze_channels)
    times = raw_eye.times

    (
        por_x,
        por_y,
        pupil_diameter,
        validity,
        confidence
    ) = data


    channels = [
        "L Validity",
        "Pupil Confidence"
    ]

    for channel in channels:

        values = raw.get_data(picks=[channel])[0]

        unique, counts = np.unique(
            values,
            return_counts=True
        )

        print(f"\n--- {channel} ---")

        for value, count in zip(unique, counts):
            print(
                f"Wert {value}: "
                f"{count} Samples "
                f"({count / len(values) * 100:.2f} %)"
            )

    # --------------------------------------------------
    # 5. Abbildung erstellen
    # --------------------------------------------------

    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 12),
        sharex=True
    )

    plt.subplots_adjust(
        hspace=0.35,
        bottom=0.17,
        top=0.91
    )

    # --------------------------------------------------
    # 6. Blickposition
    # --------------------------------------------------

    axes[0].plot(
        times,
        por_x,
        label="POR X",
        color="royalblue",
        linewidth=0.8
    )

    axes[0].plot(
        times,
        por_y,
        label="POR Y",
        color="darkorange",
        linewidth=0.8
    )

    axes[0].set_ylabel("Position (px)")
    axes[0].set_title("Blickposition – linkes Auge")
    axes[0].grid(alpha=0.2)

    # --------------------------------------------------
    # 7. Pupillendurchmesser (mm)
    # --------------------------------------------------

    axes[1].plot(
        times,
        pupil_diameter,
        color="seagreen",
        linewidth=0.8,
        label="L Mapped Diameter"
    )

    axes[1].set_ylabel("Durchmesser (mm)")
    axes[1].set_title("Gemessener Pupillendurchmesser")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right", fontsize=8)

    # Automatische Skalierung anhand der Messwerte
    finite_diameter = pupil_diameter[
        np.isfinite(pupil_diameter)
    ]

    if len(finite_diameter) > 0:

        y_min = np.min(finite_diameter)
        y_max = np.max(finite_diameter)

        if y_max > y_min:
            margin = (y_max - y_min) * 0.1
            axes[1].set_ylim(
                y_min - margin,
                y_max + margin
            )
        else:
            axes[1].set_ylim(
                y_min - 0.1,
                y_max + 0.1
            )
    # --------------------------------------------------
    # 8. Validity
    # --------------------------------------------------

    axes[2].step(
        times,
        validity,
        where="post",
        color="darkorange",
        linewidth=1
    )

    axes[2].set_ylabel("Validity")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels([
        "Ungültig (0)",
        "Gültig (1)"
    ])

    axes[2].set_ylim(-0.2, 1.2)
    axes[2].set_title("Messvalidität")
    axes[2].grid(alpha=0.2)
    
    # --------------------------------------------------
    # 9. Pupil Confidence
    # --------------------------------------------------

    axes[3].plot(
        times,
        confidence,
        color="mediumpurple",
        linewidth=0.8
    )

    axes[3].set_ylabel("Confidence")
    axes[3].set_title("Pupil Confidence")
    axes[3].set_xlabel("Zeit (s)")
    axes[3].grid(alpha=0.2)

    finite_confidence = confidence[np.isfinite(confidence)]

    if (
        len(finite_confidence) > 0
        and finite_confidence.min() >= 0
        and finite_confidence.max() <= 1
    ):
        axes[3].set_ylim(-0.05, 1.05)

    # --------------------------------------------------
    # 10. Event-Dauern darstellen
    # --------------------------------------------------

    for ax in axes:

        # Blinks: rote Flächen
        for i, event in enumerate(blink_events):

            ax.axvspan(
                event["onset"],
                event["offset"],
                color="red",
                alpha=0.16,
                label="ET_Blink L" if i == 0 else None
            )

            ax.axvline(
                event["onset"],
                color="red",
                linewidth=0.8,
                linestyle="--",
                alpha=0.8
            )

            ax.axvline(
                event["offset"],
                color="red",
                linewidth=0.8,
                linestyle="-",
                alpha=0.8
            )

        # Sakkaden: blaue Flächen
        for i, event in enumerate(saccade_events):

            ax.axvspan(
                event["onset"],
                event["offset"],
                color="blue",
                alpha=0.10,
                label="ET_Saccade L" if i == 0 else None
            )

            ax.axvline(
                event["onset"],
                color="blue",
                linewidth=0.7,
                linestyle=":",
                alpha=0.8
            )

            ax.axvline(
                event["offset"],
                color="blue",
                linewidth=0.7,
                linestyle="-",
                alpha=0.8
            )

        ax.legend(loc="upper right", fontsize=8)

    # --------------------------------------------------
    # 11. Feinere Zeitachse
    # --------------------------------------------------

    for ax in axes:
        ax.xaxis.set_major_locator(
            MultipleLocator(time_step)
        )

        ax.xaxis.set_major_formatter(
            FormatStrFormatter("%.2f")
        )

    # --------------------------------------------------
    # 12. Slider
    # --------------------------------------------------

    total_duration = times[-1]
    window = min(duration, total_duration)

    slider_ax = fig.add_axes([
        0.15, 0.06, 0.7, 0.025
    ])

    time_slider = Slider(
        slider_ax,
        "Startzeit (s)",
        0,
        max(0.001, total_duration - window),
        valinit=0,
        valstep=1 / sfreq
    )

    def update(val):

        start = time_slider.val
        end = start + window

        for ax in axes:
            ax.set_xlim(start, end)

        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    # Initiales Zeitfenster
    for ax in axes:
        ax.set_xlim(0, window)

    fig.suptitle(
        "Eye Tracking – linkes Auge: Blinks und Sakkaden",
        fontsize=14
    )

    plt.show()

    return events, event_id, blink_events, saccade_events


