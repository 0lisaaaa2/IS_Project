import mne
from mne import Annotations
from mne_icalabel import label_components
import config
import numpy as np
import matplotlib.pyplot as plt
from jamica import fit_ica

"""
Applies ICA to the raw EEG data, identifies and excludes "bad" components, and returns the cleaned data.
Optionally performs Opticat augmentation before ICA fitting. 
"""

# get copy of EEG data and prepare it for ICA
def prepare_ica_data(raw):
    # Use EEG channels only; exclude channels already marked as bad
    raw_ica = raw.copy().pick(
        picks=["eeg"],
        exclude="bads",
    )

    # Reduce slow drifts by high-pass filtering before ICA fitting
    raw_ica.filter(
        l_freq=config.highpass,
        h_freq=None,
    )

    print(f"EEG channels used: {raw_ica.info['nchan']}")
    print(f"Bad channels excluded: {raw.info['bads']}")
    print(f"Duration: {raw_ica.times[-1]:.2f} seconds")

    return raw_ica


# set ICA to use for fitting. method is defined in config.py
def get_ica(raw):
    ica = mne.preprocessing.ICA(
        n_components=len(mne.pick_types(raw.info, eeg=True)), 
        method=config.ica,
        max_iter="auto",
        random_state=97,
        verbose=False
    )
    return ica


# label components using icalabel
def label_components_ica(raw, ica):
    labels = label_components(raw, ica, method='iclabel')  # probabilities per IC and category
    print(labels)
    return labels

def exclude_components(ica, labels, ratios=None):
    mode = "both"

    # ocular ICs from the eye-tracker-based criterion
    ocular = []
    if mode in ("varratio", "both"):
        ocular = np.flatnonzero(ratios > config.varratio_threshold).tolist()

    # ICLabel
    iclabel_excl = []
    if mode in ("iclabel", "both"):
        skip = ['brain', 'other']
        iclabel_excl = [
            idx for idx, lbl in enumerate(labels['labels'])
            if lbl not in skip and labels['y_pred_proba'][idx].max() >= 0.8
        ]

    excluded = sorted(set(ocular) | set(iclabel_excl))
    print(f"Variance-ratio (ocular) ICs: {ocular}")
    print(f"ICLabel ICs:                 {iclabel_excl}")
    print(f"Excluding components:        {excluded}")
    ica.exclude = excluded

# plot topographies of ICA components
def plot_ica_components(ica, labels):

    n_components = ica.n_components_
    n_cols = 5
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 3.5 * n_rows)
    )

    axes = np.atleast_1d(axes).ravel()

    for idx in range(n_components):

        ax = axes[idx]

        # Topoplot erstellen
        ica.plot_components(
            picks=idx,
            axes=ax,
            show=False
        )

        # ICLabel-Klasse und Wahrscheinlichkeit
        label = labels['labels'][idx]
        probability = labels['y_pred_proba'][idx].max() * 100

        # Beschriftung direkt unter dem Kopf
        ax.text(
            0.5, -0.05,
            f"{label} ({probability:.1f}%)",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10
        )

    # Nicht benötigte Achsen entfernen
    for idx in range(n_components, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

# prepare data for OptiCAT. amount of additional saccade data is defined by config.ow_factor
def prepare_opticat_data(raw, raw_ica=None):

    data = raw_ica.get_data()
    n_times = data.shape[1]
    sfreq = raw.info["sfreq"]

    # Extract saccade events
    if f"ET_Saccade {config.eye}" in raw.annotations.description:
        events, _ = mne.events_from_annotations(
            raw,
            event_id={f"ET_Saccade {config.eye}": 1},
            verbose=False
        )
    else:
        events = np.empty((0, 3), dtype=int)

    # Convert event positions to indices relative to the beginning of the data
    saccade_samples = events[:, 0] - raw.first_samp

    # Convert time window from seconds to samples
    start_offset = int(round(config.saccade_window[0] * sfreq))
    stop_offset = int(round(config.saccade_window[1] * sfreq))

    windows = []

    for sample in saccade_samples:

        start = sample + start_offset
        stop = sample + stop_offset + 1

        # Ignore windows outside the recording
        if start < 0 or stop > n_times:
            continue

        window = data[:, start:stop].copy()

        # Remove temporal mean of each epoch
        if config.remove_epochmean:
            window -= window.mean(axis=1, keepdims=True)

        windows.append(window)

    # Append saccade intervals to continuous data
    if config.ow_factor > 0 and len(windows) > 0:

        # combine all extracted saccade windows into one array
        saccade_data = np.concatenate(windows, axis=1)
        target_samples = int(round(config.ow_factor * n_times))

        if target_samples > 0:
            repetitions = int(np.ceil(target_samples / saccade_data.shape[1]))
            repeated = np.tile(saccade_data, (1, repetitions))
            saccade_data = repeated[:, :target_samples]
            augmented_data = np.concatenate([data, saccade_data], axis=1) # concatenate orignal and additonal saccade data

        else:
            augmented_data = data.copy()

    else:
        print("No valid saccade windows found. Using original data.")
        augmented_data = data.copy()

    # Create Raw object containing augemented data for ica fitting
    raw_augmented = mne.io.RawArray(
        augmented_data,
        raw_ica.info.copy(),
        verbose=False
    )

    print(f"OptiCAT: Original samples: {n_times}")
    print(f"OptiCAT: Added samples: {augmented_data.shape[1] - n_times}")
    print(f"OptiCAT: Total samples: {augmented_data.shape[1]}")

    return raw_augmented

# neue jamica version
def fit_ica_model(raw_fit):
    if config.ica == "jamica":
        ica = fit_ica(
            raw_fit,
            n_components=None,      # None = Rang der Daten (siehe unten)
            max_iter=2000,
            random_state=97,
            fit_params=dict(do_reject=True, rejsig=3.0,
                            rejstart=2, rejint=3, numrej=5),
        )
        res = ica.amica_result_
        print(f"AMICA do_reject: {res.n_rejected_} Samples verworfen "
              f"({res.n_rejected_ / raw_fit.n_times:.1%})")
    else:
        ica = get_ica(raw_fit)
        ica.fit(raw_fit)
    return ica


# set the same bad channels in all runs (union), so that one ICA fits all runs
# NOTE: changes raw.info["bads"] of the given runs in place
def unify_bads(raws):
    all_bads = sorted(set().union(*(raw.info["bads"] for raw in raws)))
    for raw in raws:
        raw.info["bads"] = list(all_bads)
    print(f"Bad channels (union over all runs): {all_bads}")
    return all_bads

# Helpfunction for variance ratio (start and duration of intervalls)
def get_saccade_start_and_duration(raw, description):
    annotations = raw.annotations
    select = annotations.description == description
    sfreq = raw.info["sfreq"]

    #fallback if no annotations match
    if not select.any():
        return np.array([], dtype=int), np.array([], dtype=int)

    starts = np.round((annotations.onset[select] - raw.first_time) * sfreq).astype(int)
    durations = np.round(annotations.duration[select] * sfreq).astype(int)

    return starts, durations


def get_variants_ratio(raw, ica):
    sfreq = raw.info["sfreq"]
    n_times = raw.n_times
    pre = int(round(config.varratio_pre * sfreq))
    post = int(round(config.varratio_post * sfreq))
    sources = ica.get_sources(raw).get_data()      # (n_ICs, n_times)

    # Sakkaden: Fenster um pre/post VERLÄNGERN
    var_sac_list = []
    starts, durations = get_saccade_start_and_duration(raw, f"ET_Saccade {config.eye}")
    for s, d in zip(starts, durations):
        a = max(s - pre, 0)
        b = min(s + d + post, n_times)
        if b - a > 1:
            var_sac_list.append(sources[:, a:b].var(axis=1, ddof=1))

    # Fixationen: Fenster um pre/post KÜRZEN
    var_fix_list = []
    starts, durations = get_saccade_start_and_duration(raw, f"ET_Fixation {config.eye}")
    for s, d in zip(starts, durations):
        a = s + post              # Anfang: was die Sakkade davor nach hinten verlängert hat
        b = s + d - pre           # Ende: was die nächste Sakkade nach vorne verlängert hat
        if a >= 0 and b <= n_times and b - a > 2:
            var_fix_list.append(sources[:, a:b].var(axis=1, ddof=1))

    # Listen -> ein Mittelwert pro IC
    var_sac = np.mean(var_sac_list, axis=0)
    var_fix = np.mean(var_fix_list, axis=0)
    print(f"Varianz-Kriterium: {len(var_sac_list)} Sakkaden, {len(var_fix_list)} Fixationen")
    return var_sac / var_fix

def plot_variance_ratios(ratios):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(ratios)), ratios)
    ax.axhline(config.varratio_threshold, color="r", ls="--",
               label=f"threshold = {config.varratio_threshold}")
    ax.set_yscale("log")
    ax.set_xlabel("IC")
    ax.set_ylabel("var(sac) / var(fix)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# fit ONE ICA on all runs together; apply it later to each run separately
def fit(raws):
    unify_bads(raws)

    # concatenate copies (concatenate_raws changes the first raw in place)
    raw_all = mne.concatenate_raws([raw.copy() for raw in raws])
    raw_ica = prepare_ica_data(raw_all)

    # select data used to fit. usage of Opitcat is defined in config.py
    if config.opticat:
        print("Using OptiCAT for ICA fitting.")
        raw_fit = prepare_opticat_data(raw_all, raw_ica)
    else:
        print("Using standard ICA.")
        raw_fit = raw_ica

    #ICA wird nur neu gefittet, wenn keine Datei in results ist, oder das in der config auf False steht
    config.results_dir.mkdir(exist_ok=True)
    ica_path = config.results_dir / f"sub-{config.subject}_{config.ica}_opticat-{config.opticat}_ica.fif"
    if config.reuse_ica and ica_path.exists():
        print(f"Lade gespeicherte ICA: {ica_path.name}")
        ica = mne.preprocessing.read_ica(ica_path)
    else:
        ica = fit_ica_model(raw_fit)
        try:
            ica.save(ica_path, overwrite=True)
        except Exception as e:
            print(f"WARNUNG: ICA konnte nicht gespeichert werden: {e}")

    labels = label_components_ica(raw_ica, ica)

    # eye-tracker-based variance ratio on the original concated data
    ratios = get_variants_ratio(raw_all, ica)
    labels["variance_ratio"] = ratios
    plot_variance_ratios(ratios)

    exclude_components(ica, labels, ratios)
    
    # plot ica results and label components
    if config.components_plot:
        ica.plot_sources(raw_ica, show_scrollbars=True, show=True)
        plot_ica_components(ica, labels)

    return ica, labels