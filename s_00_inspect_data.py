def inspect_data(raw):
    print("\n--- ALLGEMEINE INFORMATIONEN ---")
    print(f"Kanäle: {len(raw.ch_names)}")
    print(f"Sampling Rate: {raw.info['sfreq']} Hz")
    print(f"Dauer: {raw.times[-1]:.3f} Sekunden")
    print(f"Anzahl Samples: {raw.n_times}")
    print(f"Bad Channels: {raw.info['bads']}")
    print(f"Annotations: {len(raw.annotations)}")

    print("\n--- KANALTYPEN ---")
    types = raw.get_channel_types()
    for channel_type in sorted(set(types)):
        names = [
            name for name, typ in zip(raw.ch_names, types)
            if typ == channel_type
        ]
        print(f"{channel_type}: {len(names)}")
        print(names)

    print("\n--- ALLE KANALNAMEN ---")
    for i, name in enumerate(raw.ch_names):
        print(f"{i:3d}: {name}")

    print("\n--- MONTAGE ---")
    montage = raw.get_montage()
    if montage is None:
        print("Kein Montage-Objekt vorhanden.")
    else:
        print("Montage vorhanden.")
        print("Kanalpositionen:", len(montage.ch_names))

    print("\n--- REFERENZ ---")
    print("EEG reference:", raw.info.get("custom_ref_applied"))