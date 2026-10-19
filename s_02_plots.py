
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


def plot_saccade_frps(large, small, clean):
    tmin, tmax = -0.5, 2.0

    def make_epochs(saccades):
        onsets = saccades["onset"].to_numpy(dtype=float)
        samples = clean.first_samp + np.round(
            (onsets - clean.first_time) * clean.info["sfreq"]
        ).astype(int)

        events = np.column_stack((
            samples,
            np.zeros(len(samples), dtype=int),
            np.ones(len(samples), dtype=int)
        ))

        return mne.Epochs(
            clean, events, event_id={"saccade": 1},
            tmin=tmin, tmax=tmax, baseline=(tmin, 0),
            picks="eeg", preload=True,
            reject_by_annotation=True,
            event_repeated="drop", verbose=False
        )

    epochs_large = make_epochs(large)
    epochs_small = make_epochs(small)

    if len(epochs_large) == 0 or len(epochs_small) == 0:
        raise ValueError("Eine Gruppe enthält keine gültigen Epochen.")

    evoked_large = epochs_large.average()
    evoked_small = epochs_small.average()

    # Mittelwert über alle EEG-Kanäle (mit Vorsicht interpretieren)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        evoked_small.times,
        evoked_small.data.mean(axis=0) * 1e6,
        color="royalblue", label=f"Klein (n={len(epochs_small)})"
    )
    ax.plot(
        evoked_large.times,
        evoked_large.data.mean(axis=0) * 1e6,
        color="gold", label=f"Groß (n={len(epochs_large)})"
    )

    ax.axvline(0, color="black", linestyle="--")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set(xlim=(tmin, tmax), xlabel="Zeit (s)",
           ylabel="Voltage (µV)", title="FRPs: kleine vs. große Sakkaden")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def main(raw, clean):
    large, small = get_info(raw)
    plot_saccade_frps(large, small, clean)
