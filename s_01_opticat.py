import mne
from collections import Counter

# constants
FREQ_HIGH_PASS = 2.0 # high-pass filter at 2-2.5?

# what does this do?
OW_FACTOR = 1 # value for overweighting of SPs (1 = add spike potentials corresponding to 100% of original data length)
REMOVE_EPOCHMEAN = True # remove mean from each epoch before ICA fitting (recommended)

def opticat(raw):
    # create copy of data used as training data and high-pass filter it 
    raw_ica = raw.copy().pick(
        picks=["eeg", "eog"], # include eog channels 
        exclude="bads",)
    raw_ica.filter(l_freq=FREQ_HIGH_PASS, h_freq=None) 

    # cut training data into epochs, e.g. around stimulus onsets 
    events, event_id = mne.events_from_annotations(raw)
    print(f"Event IDs: {event_id}")

    # check event frequencies?
    event_counts = Counter(events[:, 2])

    for description, event_code in event_id.items():
        print(f"{description}: {event_counts[event_code]}")

    for description in [
    "ET_# Message: 14",
    "ET_# Message: 20",
    "ET_# Message: 92",
    "trialResponse",
    ]:
        print(f"\n{description}")

        mask = raw.annotations.description == description

        for onset in raw.annotations.onset[mask]:
            print(f"{onset:.3f} s")

    #epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject_by_annotation=True, event_repeated='drop', reject=reject_criteria)
    return None
#