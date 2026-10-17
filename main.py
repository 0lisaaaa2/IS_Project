import s_00_fetch_data
import config


def extended_pipeline():
    raw = s_00_fetch_data.load_data(config.subject, config.task, config.run)
    raw.plot(block=True)

if __name__ == "__main__":
    extended_pipeline()