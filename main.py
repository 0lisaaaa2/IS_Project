import s_00_fetch_data, s_00_inspect_data, s_01_ica
import config
import numpy as np


def extended_pipeline():
    raw = s_00_fetch_data.load_data(config.subject, config.task, config.run)
    raw.plot(block=True)

    #s_00_inspect_data.inspect_data(raw)

    # regular ica
    clean_raw, n_excluded = s_01_ica.ica(raw)
    clean_raw.plot(block=True)
    
if __name__ == "__main__":
    extended_pipeline()