import mne
import pandas as pd
import numpy as np

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

def mark_blink_periods(raw, eye="L", pre_ms=50, post_ms=150,
                       label="ET_bad_blink"):
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

def remove_saccades_around_blinks(raw):
    extend_ms = 50
    blink_events, saccade_events, fixation_events = (extract_eyetracking_events(raw))
    extend_sec = extend_ms / 1000

    bad_onsets = []
    bad_durations = []
    removed_saccades = []

    for saccade in saccade_events:
        remove = False
        for blink in blink_events:
            blink_start = blink["onset"] - extend_sec
            blink_end = blink["offset"] + extend_sec
            if (saccade["onset"] <= blink_end and saccade["offset"] >= blink_start):
                remove = True
                break

        if remove:
            removed_saccades.append(saccade)

            # Zeitraum der Sakkade als BAD markieren
            bad_onsets.append(saccade["onset"])
            bad_durations.append(saccade["duration"])

    # Raw kopieren, damit das Original erhalten bleibt
    cleaned_raw = raw.copy()

    # BAD-Annotations hinzufügen
    if bad_onsets:
        bad_annotations = mne.Annotations(
            onset=bad_onsets,
            duration=bad_durations,
            description=["BAD_saccade_blink"] * len(bad_onsets),
            orig_time=cleaned_raw.annotations.orig_time
        )

        cleaned_raw.set_annotations(
            cleaned_raw.annotations + bad_annotations
        )

    print("Blinks:", len(blink_events))
    print("Original saccades:", len(saccade_events))
    print("Removed saccades:", len(removed_saccades))

    return cleaned_raw
