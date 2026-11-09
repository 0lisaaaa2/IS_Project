from pathlib import Path

# Pfade (relativ zu dieser Datei, egal von wo gestartet wird)
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # code/ → IS_Project/
bids_root = PROJECT_ROOT / "data"
results_dir = PROJECT_ROOT / "results"

subject = "NDARHT518WEM"
task = "allTasks"
#runs = ["1", "2", "3", "4", "5", "7", "8"] #6 ist komisch, deswegen nicht drin
runs = ["2"]
electrode = "E25" # above left eye 
#processing = "eyelink"
highpass = 2.0 # change to 2.0
eye="L"

# Welches ICA möchtest du verwenden?
opticat = False # (True | False)
ica = "infomax" # ("infomax" | "jamica")
# OptiCAT settings
ow_factor = 1
remove_epochmean = True
saccade_window = (-0.020, 0.010)

# Zeiten um Blinks, die markiert verden
blink_pre_ms =50
blink_post_ms = 100