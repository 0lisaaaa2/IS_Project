from mne_bids import BIDSPath, read_raw_bids
import config
from pathlib import Path
import mne

"""
Load raw EEG data for a given subject and task using MNE-BIDS.
"""

# def load_data(sub,task, run):

#     bids_path = BIDSPath(
#         subject=sub,
#         task=task,
#         run=run,
#         datatype="eeg",
#         processing="eyelink",
#         suffix="raw",
#         extension=".fif",
#         root=config.bids_root
#     )

#     print("Looking for:", bids_path.fpath)

#     raw = read_raw_bids(bids_path)
#     raw.load_data()
#     return raw

def load_data(subject, task, run):
    file_path = (
        Path(config.bids_root)
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_task-{task}_run-{run}_proc-eyelink_raw.fif"
    )

    # reads a specific fif file (doesn't require bids structure. fix for now since raw is no elgible suffix)
    raw = mne.io.read_raw_fif(file_path, preload=True)

    return raw