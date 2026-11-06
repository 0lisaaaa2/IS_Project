import config
import mne

def extract_eyetracking_events(raw):
    annotations = raw.annotations
    blink_events = []
    saccade_events = []
    fixation_events = []

    for onset, duration, description in zip(annotations.onset, annotations.duration,annotations.description):
        event = {
            "onset": onset,
            "offset": onset + duration,
            "duration": duration,
            "description": description
        }

        if "ET_Blink" in description:
            blink_events.append(event)
        elif "ET_Saccade" in description:
            saccade_events.append(event)
        elif "ET_Fixation" in description:
            fixation_events.append(event)

    return blink_events, saccade_events, fixation_events

def mark_blink_periods(raw):
    eye = config.eye
    pre_ms = config.blink_pre_ms
    post_ms = config.blink_post_ms
    label= "ET_bad_blink"
    blinks, saccades, _ = extract_eyetracking_events(raw)
    blinks   = [b for b in blinks   if b["description"].endswith(eye)]
    saccades = [s for s in saccades if s["description"].endswith(eye)]

    onsets, durations = [], []
    for b in blinks:
        start = b["onset"]  - pre_ms / 1000
        end   = b["offset"] + post_ms / 1000

        # angrenzende Sakkaden einschließen → Fenster ggf. verlängern
        for s in saccades:
            if s["onset"] <= end and s["offset"] >= start:
                start = min(start, s["onset"])
                end   = max(end,   s["offset"] + post_ms / 1000)

        onsets.append(start)
        durations.append(end - start)

    ann = mne.Annotations(onsets, durations, [f"{label} {eye}"] * len(onsets),
                          orig_time=raw.annotations.orig_time)
    out = raw.copy()
    out.set_annotations(out.annotations + ann)
    return out
