from envs.frozenlake_env.Lib.collections import Counter
import mne
from collections import Counter

def inspect_data(raw):
    print("\n--- ALLGEMEINE INFORMATIONEN ---")
    print(f"Kanäle: {len(raw.ch_names)}")
    print(f"Sampling Rate: {raw.info['sfreq']} Hz")
    print(f"Dauer: {raw.times[-1]:.3f} Sekunden")
    print(f"Anzahl Samples: {raw.n_times}")
    print(f"Bad Channels: {raw.info['bads']}")
    print(f"Annotations: {len(raw.annotations)}")

    print("\n--- ANNOTATIONS ---")
    print(raw.annotations)
    # for i, annotation in enumerate(raw.annotations):
    #     print(f"Annotation {i}:")
    #     print("  Onset:", annotation["onset"])
    #     print("  Duration:", annotation["duration"])
    #     print("  Description:", annotation["description"])

    # Annotationstypen zählen
    annotation_counts = Counter(raw.annotations.description)
    print("\n--- UNIQUE ANNOTATION TYPES ---")
    for annotation_type, count in annotation_counts.items():
        print(f"{annotation_type}: {count}")

    # print("\n--- KANALTYPEN ---")
    # types = raw.get_channel_types()
    # for channel_type in sorted(set(types)):
    #     names = [
    #         name for name, typ in zip(raw.ch_names, types)
    #         if typ == channel_type
    #     ]
    #     print(f"{channel_type}: {len(names)}")
    #     print(names)

    # print("\n--- ALLE KANALNAMEN ---")
    # for i, name in enumerate(raw.ch_names):
    #     print(f"{i:3d}: {name}")

    # print("\n--- MONTAGE ---")
    # montage = raw.get_montage()
    # if montage is None:
    #     print("Kein Montage-Objekt vorhanden.")
    # else:
    #     print("Montage vorhanden.")
    #     print("Kanalpositionen:", len(montage.ch_names))

    # print("\n--- REFERENZ ---")
    # print("EEG reference:", raw.info.get("custom_ref_applied"))
    # print("Projektoren:", raw.info["projs"])

    # for ch in raw.info["chs"]:
    #     if ch["ch_name"] in ["HEOG", "VEOG"]:
    #         print(ch["ch_name"])
    #         print("Typ:", ch["kind"])
    #         print("Einheit:", ch["unit"])
    #         print("Kalibrierung:", ch["cal"])
    #         print("Zusatzinformationen:", ch)

    # print("custom_ref_applied:",
    #   raw.info.get("custom_ref_applied"))

    # print("EEG-Kanäle:", len(
    #     mne.pick_types(raw.info, eeg=True, eog=False)
    # ))

    # print("EOG-Kanäle:", len(
    #     mne.pick_types(raw.info, eeg=False, eog=True)
    # ))

    # print("Kanalnamen:", raw.ch_names[:10])