import numpy as np
import mne


def estimate_et_lag(raw, eye="L", heog="HEOG", max_lag_ms=300, lowpass=20.0):
    """Schätzt den Versatz zwischen Eyetracking und EEG.

    Kreuzkorrelation der Geschwindigkeit von HEOG (EEG) und horizontaler
    Blickposition (Eyetracker). Positiver Lag = EEG reagiert später,
    d.h. die ET-Events liegen zu früh.
    Returns: lag in Sekunden, Korrelation r
    """
    sfreq = raw.info["sfreq"]
    gaze_ch = f"{eye} POR X [px]"
    data = raw.get_data(picks=[heog, gaze_ch]).copy()

    # Lücken in den Blickdaten (Blinks, Trackverlust) linear interpolieren
    gaze = data[1]
    bad = ~np.isfinite(gaze)
    if bad.all():
        return np.nan, np.nan
    if bad.any():
        gaze[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), gaze[~bad])

    data = mne.filter.filter_data(data, sfreq, None, lowpass, verbose=False)

    # Geschwindigkeit, z-standardisiert
    vel = np.diff(data, axis=1)
    vel = (vel - vel.mean(axis=1, keepdims=True)) / vel.std(axis=1, keepdims=True)
    v_eeg, v_et = vel

    max_lag = int(round(max_lag_ms / 1000 * sfreq))
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.array([
        np.mean(v_eeg[L:] * v_et[:len(v_et) - L]) if L >= 0
        else np.mean(v_eeg[:L] * v_et[-L:])
        for L in lags
    ])
    best = np.argmax(r)       # positive Korrelation: gleiche Bewegungsrichtung
    return lags[best] / sfreq, r[best]


def correct_et_lag(raw, eye="L", max_valid_ms=150, min_r=0.15):
    """Verschiebt alle 'ET_'-Annotationen um den geschätzten Versatz.

    Gibt (raw_korrigiert, lag_ms, r, ok) zurück. Ist die Schätzung unplausibel
    (|lag| > max_valid_ms oder r < min_r), bleibt raw unverändert und ok=False.
    """
    lag, r = estimate_et_lag(raw, eye=eye)
    lag_ms = lag * 1000
    ok = np.isfinite(lag) and abs(lag_ms) <= max_valid_ms and r >= min_r
    if not ok:
        print(f"WARNUNG: Versatz unplausibel (lag={lag_ms:.0f} ms, r={r:.2f}) – nicht korrigiert")
        return raw, False

    ann = raw.annotations
    is_et = np.array([d.startswith("ET_") for d in ann.description])
    onset = ann.onset + np.where(is_et, lag, 0.0)

    new_ann = mne.Annotations(onset, ann.duration, ann.description,
                              orig_time=ann.orig_time, extras=ann.extras)
    raw = raw.copy().set_annotations(new_ann)
    print(f"ET-Annotationen um {lag_ms:+.0f} ms verschoben (r={r:.2f}, {is_et.sum()} Annotationen)")
    return raw, True