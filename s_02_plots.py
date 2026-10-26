import config
import numpy as np
import matplotlib.pyplot as plt
import mne


def get_info(raw):
    df = raw.annotations.to_data_frame(time_format=None)

    saccades = df.loc[
        df["description"].eq("ET_Saccade L"),
        ["onset", "Amplitude"]
    ].dropna(subset=["Amplitude"])

    large = saccades[saccades["Amplitude"] >= 10]
    small = saccades[saccades["Amplitude"] <= 2]

    return large, small


"""
Plot small, medium und large saccades for several runs.
"""
def plot_saccade_frps_raw_runs(runs):
    tmin, tmax = -0.3, 0.5
    electrode = config.electrode
    baseline = (tmin, 0)

    # Drei Plots untereinander
    fig, axes = plt.subplots(
        3, 1,
        figsize=(11, 12),
        sharex=True
    )

    groups = [
        ("small", "Kleine Sakkaden (≤ 2°)", axes[0]),
        ("medium", "Mittlere Sakkaden (2° < Amplitude < 10°)", axes[1]),
        ("large", "Große Sakkaden (≥ 10°)", axes[2]),
    ]

    for run_label, raw in runs:
        df = raw.annotations.to_data_frame(time_format=None)

        saccades = df.loc[
            df["description"].eq("ET_Saccade L"),
            ["onset", "Amplitude"]
        ].dropna(subset=["Amplitude"])

        # Sakkaden nach Amplitude aufteilen
        small = saccades[saccades["Amplitude"] <= 2]
        medium = saccades[
            (saccades["Amplitude"] > 2)
            & (saccades["Amplitude"] < 10)
        ]
        large = saccades[saccades["Amplitude"] >= 10]

        saccade_groups = {
            "small": small,
            "medium": medium,
            "large": large,
        }

        for group, title, ax in groups:
            saccades_group = saccade_groups[group]

            if saccades_group.empty:
                print(
                    f"{run_label}: keine {group} Sakkaden gefunden"
                )
                continue

            # Sakkaden-Onsets in EEG-Samples umrechnen
            onsets = saccades_group["onset"].to_numpy(dtype=float)

            samples = (
                raw.first_samp
                + np.round(
                    (onsets - raw.first_time)
                    * raw.info["sfreq"]
                ).astype(int)
            )

            events = np.column_stack((
                samples,
                np.zeros(len(samples), dtype=int),
                np.ones(len(samples), dtype=int)
            ))

            # EEG-Epochen erstellen
            epochs = mne.Epochs(
                raw,
                events,
                event_id={"saccade": 1},
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                picks="eeg",
                preload=True,
                reject_by_annotation=True,
                event_repeated="drop",
                verbose=False
            )

            if len(epochs) == 0:
                print(
                    f"{run_label}: keine gültigen Epochen für {group}"
                )
                continue

            if electrode not in epochs.ch_names:
                raise ValueError(
                    f"Elektrode {electrode} fehlt in {run_label}. "
                    f"Verfügbare Kanäle: {epochs.ch_names}"
                )

            # Über die Epochen mitteln
            evoked = epochs.average()

            voltage = (
                evoked.copy()
                .pick([electrode])
                .data[0] * 1e6
            )

            ax.plot(
                evoked.times,
                voltage,
                label=f"{run_label} (n={len(epochs)})"
            )

            ax.set_title(title)
            ax.set_ylabel("Voltage (µV)")
            ax.axvline(
                0, color="black",
                linestyle="--", linewidth=0.8
            )
            ax.axhline(
                0, color="gray",
                linestyle=":", linewidth=0.8
            )
            ax.grid(alpha=0.2)
            ax.legend()

    axes[2].set_xlabel("Zeit relativ zum Sakkadenbeginn (s)")
    axes[2].set_xlim(tmin, tmax)

    fig.suptitle(
        f"FRPs über bereinigte Runs – Elektrode {electrode}",
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()




"""
topoplot for several timestamps (call in s_02_plots.py)
"""
def plot_saccade_topomaps(raw, saccades, title="Sakkaden"):
    # Zeitfenster in Sekunden
    time_bins = [
        (-0.3, -0.2),
        (-0.2, -0.1),
        (-0.1,  0.0),
        ( 0.0,  0.1),
        ( 0.1,  0.2),
        ( 0.2,  0.3),
        ( 0.3,  0.4),
        ( 0.4,  0.5),
    ]

    tmin, tmax = -0.5, 0.5

    # Sakkaden-Onsets in EEG-Samples umrechnen
    onsets = saccades["onset"].to_numpy(dtype=float)

    samples = (
        raw.first_samp
        + np.round(
            (onsets - raw.first_time) * raw.info["sfreq"]
        ).astype(int)
    )

    events = np.column_stack((
        samples,
        np.zeros(len(samples), dtype=int),
        np.ones(len(samples), dtype=int),
    ))

    # EEG-Epochen erstellen
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"saccade": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=(-0.3, -0.1),
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        event_repeated="drop",
        verbose=False,
    )

    if len(epochs) == 0:
        raise ValueError("Keine gültigen EEG-Epochen gefunden.")

    # Über Sakkaden mitteln
    evoked = epochs.average()

    # Mittelwerte für jedes Zeitfenster berechnen
    bin_data = []
    bin_labels = []

    for start, end in time_bins:
        evoked_bin = evoked.copy().crop(
            tmin=start,
            tmax=end,
            include_tmax=False,
        )

        data = evoked_bin.data.mean(axis=1) * 1e6  # µV
        bin_data.append(data)
        bin_labels.append(f"{start:.1f} bis {end:.1f} s")

    # Einheitliche, symmetrische Farbskala für alle Topografien
    vmax = max(np.max(np.abs(data)) for data in bin_data)
    if vmax == 0:
        vmax = 1e-12

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Acht Topografien zeichnen
    for ax, data, label in zip(
        axes.flat, bin_data, bin_labels
    ):
        image, _ = mne.viz.plot_topomap(
            data,
            evoked.info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            contours=6,
        )
        ax.set_title(label)

    # Platz für die Farbskala rechts freihalten
    fig.subplots_adjust(
        top=0.88,
        bottom=0.08,
        left=0.05,
        right=0.88,
        wspace=0.25,
        hspace=0.35,
    )

    # Gemeinsame Farbskala rechts neben den Köpfen
    cbar_ax = fig.add_axes([0.91, 0.20, 0.02, 0.60])

    fig.colorbar(
        image,
        cax=cbar_ax,
        label="Spannung (µV)",
    )

    fig.suptitle(
        f"{title} – Topografien (n={len(epochs)} Epochen)",
        fontsize=14,
    )

    plt.show()



def main(raw):
    large, small = get_info(raw)
    plot_saccade_topomaps(raw, small, title="Kleine Sakkaden (≤ 2°)") # large
    plot_saccade_topomaps(raw, large, title="Große Sakkaden (> 10°)")