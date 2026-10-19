from matplotlib.pylab import sample
import mne
from collections import Counter
import numpy as np
from mne_icalabel import label_components

# constants
FREQ_HIGH_PASS = 1.0 # high-pass filter at 2-2.5?

# what does this do?
OW_FACTOR = 1 # value for overweighting of SPs (1 = add spike potentials corresponding to 100% of original data length)
REMOVE_EPOCHMEAN = True # remove mean from each epoch before ICA fitting (recommended)

SACCADE_LABEL = 'ET_Saccade L' # label of saccade events in annotations
SACCADE_WINDOW = (-0.020, 0.010) # time window around saccade events in seconds (start, end)

def opticat(raw):
    # create copy of data used as training data and high-pass filter it 
    raw_ica = raw.copy().pick(
        picks=["eeg", "eog"], # include eog channels 
        exclude="bads",)
    raw_ica.filter(l_freq=FREQ_HIGH_PASS, h_freq=None) 

    # cut training data into epochs, e.g. around stimulus onsets 
    # kopiert aus altem eeg-projekt. aber um was herum epoching? no "stimulus" events per se 
    #epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject_by_annotation=True, event_repeated='drop', reject=reject_criteria)

    # check present events/annotations
    # events, event_id = mne.events_from_annotations(raw)
    # print(f"Event IDs: {event_id}")

    # # check event frequencies?
    # event_counts = Counter(events[:, 2])
    # for description, event_code in event_id.items():
    #     print(f"{description}: {event_counts[event_code]}")


    


    # sakkaden untersuchen: work_around
    # sakkaden extrahieren
    saccade_mask = raw.annotations.description == 'ET_Saccade L'
    saccade_onsets = raw.annotations.onset[saccade_mask]
    #print(f"Number of left-eye saccades: {len(saccade_onsets)}")
    #print(saccade_onsets[:10])

    # sakkaden fenster bestimmen: −20 ms bis +10 ms
    saccade_samples = raw.time_as_index(saccade_onsets) - raw.first_samp
    # print("Erste Sakkadenindizes:", saccade_samples[:5])
    # print("Letzte Sakkadenindizes:", saccade_samples[-5:])
    # print("Anzahl Zeitpunkte in raw:", raw.n_times)
    sfreq = raw.info["sfreq"]
    pre_samples = int(0.020 * sfreq)
    post_samples = int(0.010 * sfreq)

    # sakkadenfenster ausschneiden
    data = raw_ica.get_data()
    n_times = data.shape[1]

    windows = []
    valid_samples = []

    for sample in saccade_samples: # fenster schneiden
        start = sample - pre_samples
        stop = sample + post_samples

        # Prüfen, ob das Fenster innerhalb der Daten liegt
        if start < 0 or stop > n_times:
            continue

        windows.append(data[:, start:stop])
        valid_samples.append(sample)

    # print("Anzahl gültiger Fenster:", len(windows))
    # print("Form eines Fensters:", windows[0].shape)

    # berechnen,w ei oft die sakkadenfenster benötigt werden für OW_FACTOR (sollten genau so viele zusätzliche datenpunkte hinzufügen, wie der ursprüngliche datensatz enthält)
    saccade_data = np.concatenate(windows, axis=1)

    target_samples = data.shape[1]
    current_samples = saccade_data.shape[1]

    repetitions = int(np.ceil(target_samples / current_samples))

    print("Ursprüngliche Datenpunkte:", target_samples)
    print("Sakkaden-Datenpunkte pro Durchlauf:", current_samples)
    print("Benötigte Wiederholungen:", repetitions)

    # ergebnis: wiederholen sakkaden-fenster 8-mal
    # Sakkadenfenster mehrfach zusammenfügen
    saccade_data_repeated = np.tile(saccade_data, (1, repetitions))

    # Originaldaten und zusätzliche Sakkadendaten kombinieren
    augmented_data = np.concatenate(
        [data, saccade_data_repeated],
        axis=1
    )

    # print("Originaldaten:", data.shape)
    # print("Zusätzliche Sakkadendaten:", saccade_data_repeated.shape)
    # print("Erweiterte Trainingsdaten:", augmented_data.shape)
    
    # mne raw object erstellen mit den erweiterten daten
    raw_augmented = mne.io.RawArray(
    augmented_data,
    raw_ica.info.copy(),
    verbose=False,
    )

    print("Original training data:", raw_ica.get_data().shape)
    print("Augmented training data:", raw_augmented.get_data().shape)

    # ica fitten
    ica_opticat = mne.preprocessing.ICA(
        n_components=len(mne.pick_types(raw_ica.info, eeg=True)),
        method="infomax",
        max_iter="auto",
        random_state=97,
        verbose=False,
    )

    ica_opticat.fit(raw_augmented)

    print("ICA fitted on augmented data.")
    print("Number of components:", ica_opticat.n_components_)

    labels = label_components(
        raw_ica,
        ica_opticat,
        method="iclabel",
    )

    excluded_components = [
            idx for idx, lbl in enumerate(labels['labels'])
            if lbl not in ['brain', 'other'] and labels['y_pred_proba'][idx] >= 0.8
        ]
        #print(f"Excluding components: {excluded_components}")
    ica_opticat.exclude = excluded_components

    print("Excluded components:", ica_opticat.exclude)
    print("Number excluded:", len(ica_opticat.exclude))

    # ica auf ursprüngliche daten anwenden
    raw_cleaned_opticat = ica_opticat.apply(raw.copy())

    return raw_cleaned_opticat, len(ica_opticat.exclude)
#

# new version of function
def opticat_clean(raw): 

    # create copy and high-pass filter it 
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    raw_ica = raw.copy().pick(picks=eeg_picks)
    raw_ica.filter(l_freq=FREQ_HIGH_PASS, h_freq=None)

    data = raw_ica.get_data()
    n_channels, n_times = data.shape

    # skip epoching 

    # extract short intervals around saccade onsets (-20 ms to +10 ms)
    events, _ = mne.events_from_annotations(raw, event_id={SACCADE_LABEL: 1})
    saccade_samples = events[:, 0] - raw.first_samp # convert indices relative to first sample 
    pre_samples = int(round(abs(SACCADE_WINDOW[0]) * raw.info['sfreq'])) # convert window limits into sample offsets
    post_samples = int(round(SACCADE_WINDOW[1] * raw.info['sfreq']))
    
    windows = []
    for sample in saccade_samples:
        start = sample - pre_samples
        stop = sample + post_samples + 1 
        if start < 0 or stop > n_times:  # stop for windows outside data range
            continue 
        window = data[:, start:stop].copy()
        if REMOVE_EPOCHMEAN: # remove each window's temporal mean, per channel
            window -= window.mean(axis=1, keepdims=True)  # remove mean of each epoch
        windows.append(window)

    # append the saccade intervals to continuous data accrording to OW_FACTOR
    if OW_FACTOR > 0:
        saccade_data = np.concatenate(windows, axis=1)
        target_samples = int(round(OW_FACTOR * n_times))  # target number of samples to add
        if target_samples > 0: 
            repetitions = int(np.ceil(target_samples / saccade_data.shape[1])) # calculate how many times to repeat the saccade data
            repeated = np.tile(saccade_data, (1, repetitions)) # repeat the saccade data along the time axis
            saccade_data = repeated[:, :target_samples] # trim the repeated data to the target number of samples
            augmented_data = np.concatenate([data, saccade_data],axis=1,) # append the saccade data to the original data 
        else: 
            augmented_data = data.copy()
    else: 
        augmented_data = data.copy()

    raw_augmented = mne.io.RawArray(augmented_data, raw_ica.info.copy(), verbose=False) # turn augmented data into raw object for ica fitting

    # fit infomax ica on augmented data
    ica_opticat = mne.preprocessing.ICA(
            n_components=len(mne.pick_types(raw_ica.info, eeg=True)),
            method="infomax",
            max_iter="auto",
            random_state=97,
            verbose=False)
    ica_opticat.fit(raw_augmented)

    # classify components using icalabel
    labels = label_components(
            raw_ica,
            ica_opticat,
            method="iclabel")
    
    excluded_components = [
            idx for idx, lbl in enumerate(labels['labels'])
            if lbl not in ['brain', 'other'] and labels['y_pred_proba'][idx] >= 0.8]
    ica_opticat.exclude = excluded_components

    # apply to original data
    raw_cleaned_opticat = ica_opticat.apply(raw.copy())
    return raw_cleaned_opticat, len(ica_opticat.exclude)