import mne
from mne_icalabel import label_components
import config
import numpy as np
import matplotlib.pyplot as plt

"""
Applies ICA to the raw EEG data, identifies and excludes "bad" components, and returns the cleaned data.
"""



# get copy of EEG data and prepare it for ICA
def prepare_ica_data(raw):
    # Use EEG channels only; exclude channels already marked as bad.
    raw_ica = raw.copy().pick(
        picks=["eeg"],
        exclude="bads",
    )

    # Reduce slow drifts before ICA fitting.
    raw_ica.filter(
        l_freq=config.highpass,
        h_freq=None,
    )

    print(f"EEG channels used: {raw_ica.info['nchan']}")
    print(f"Bad channels excluded: {raw.info['bads']}")
    print(f"Duration: {raw_ica.times[-1]:.2f} seconds")

    return raw_ica


# set ICA to use for fitting
def get_ica(raw):
    ica = mne.preprocessing.ICA(
        n_components=len(mne.pick_types(raw.info, eeg=True)), 
        method="infomax",  #use infomax because of icalabel recommendation
        # method="picard", #use if picard
        # fit_params={"extended": True}, #use if picard
        # method ="jamica", #use if jamica
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


# exclude components that are not brain or other with a probability of at least 0.8
def exclude_components(ica, labels):
    excluded_components = [
        idx for idx, lbl in enumerate(labels['labels'])
        if lbl not in ['brain', 'other']
        and labels['y_pred_proba'][idx].max() >= 0.8
    ]
    print(f"Excluding components: {excluded_components}")
    ica.exclude = excluded_components


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

# main function to do ICA and return cleaned data
def ica(raw):
    raw_ica = prepare_ica_data(raw)
    ica = get_ica(raw_ica)

    ica.fit(raw_ica)

    labels = label_components_ica(raw_ica, ica)

    # show time series of ICs
    # ica.plot_sources(raw, show_scrollbars=True, show=True)
    #ica.plot_components()

    # Plot ICA components with ICLabel classification
    plot_ica_components(ica, labels)

    # Exclude components
    exclude_components(ica, labels)
    n_excluded = len(ica.exclude)
    raw_cleaned = ica.apply(raw.copy())
    return raw_cleaned, n_excluded