import mne
import numpy as np
from mne_icalabel import label_components
import config

# constants

# what does this do?
OW_FACTOR = 1 # value for overweighting of SPs (1 = add spike potentials corresponding to 100% of original data length)
REMOVE_EPOCHMEAN = True # remove mean from each epoch before ICA fitting (recommended)

SACCADE_LABEL = 'ET_Saccade L' # label of saccade events in annotations
SACCADE_WINDOW = (-0.020, 0.010) # time window around saccade events in seconds (start, end)


# new version of function
def opticat(raw): 

    # create copy and high-pass filter it 
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    raw_ica = raw.copy().pick(picks=eeg_picks)
    raw_ica.filter(l_freq=config.highpass, h_freq=None)

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
            method=config.ica,
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
    return raw_cleaned_opticat, ica_opticat, len(ica_opticat.exclude), labels



# set the same bad channels in all runs (union), so that one ICA fits all runs
# NOTE: changes raw.info["bads"] of the given runs in place
def unify_bads(raws):
    all_bads = sorted(set().union(*(raw.info["bads"] for raw in raws)))
    for raw in raws:
        raw.info["bads"] = list(all_bads)
    print(f"Bad channels (union over all runs): {all_bads}")
    return all_bads


# filtered EEG data + saccade windows of ONE run (same steps as in opticat())
def prepare_run(raw):
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    raw_ica = raw.copy().pick(picks=eeg_picks)
    raw_ica.filter(l_freq=config.highpass, h_freq=None)

    data = raw_ica.get_data()
    n_times = data.shape[1]

    events, _ = mne.events_from_annotations(raw, event_id={SACCADE_LABEL: 1})
    saccade_samples = events[:, 0] - raw.first_samp
    pre_samples = int(round(abs(SACCADE_WINDOW[0]) * raw.info['sfreq']))
    post_samples = int(round(SACCADE_WINDOW[1] * raw.info['sfreq']))

    windows = []
    for sample in saccade_samples:
        start = sample - pre_samples
        stop = sample + post_samples + 1
        if start < 0 or stop > n_times:
            continue
        window = data[:, start:stop].copy()
        if REMOVE_EPOCHMEAN:
            window -= window.mean(axis=1, keepdims=True)
        windows.append(window)

    return raw_ica, data, windows


# fit ONE OPTICAT-ICA on all runs together; apply it later to each run separately
def fit(raws):
    unify_bads(raws)

    infos, datas, windows = [], [], []
    for raw in raws:
        raw_ica, data, run_windows = prepare_run(raw)
        infos.append(raw_ica.info)
        datas.append(data)
        windows.extend(run_windows)

    # continuous data of all runs
    data = np.concatenate(datas, axis=1)
    n_times = data.shape[1]
    info = infos[0].copy()

    # append saccade intervals of all runs according to OW_FACTOR (relative to total length)
    if OW_FACTOR > 0 and len(windows) > 0:
        saccade_data = np.concatenate(windows, axis=1)
        target_samples = int(round(OW_FACTOR * n_times))
        repetitions = int(np.ceil(target_samples / saccade_data.shape[1]))
        saccade_data = np.tile(saccade_data, (1, repetitions))[:, :target_samples]
        augmented_data = np.concatenate([data, saccade_data], axis=1)
    else:
        augmented_data = data.copy()

    raw_augmented = mne.io.RawArray(augmented_data, info.copy(), verbose=False)

    ica_opticat = mne.preprocessing.ICA(
            n_components=len(mne.pick_types(info, eeg=True)),
            #method="infomax",
            method="jamica", # combine opticat + jamica
            max_iter="auto",
            random_state=97,
            verbose=False)
    ica_opticat.fit(raw_augmented)

    # classify components on the (non-augmented) data of all runs
    raw_all = mne.io.RawArray(data, info.copy(), verbose=False)
    labels = label_components(raw_all, ica_opticat, method="iclabel")

    excluded_components = [
            idx for idx, lbl in enumerate(labels['labels'])
            if lbl not in ['brain', 'other'] and labels['y_pred_proba'][idx] >= 0.8]
    ica_opticat.exclude = excluded_components

    return ica_opticat, labels