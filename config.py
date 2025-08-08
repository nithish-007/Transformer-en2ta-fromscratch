# -----------------
# config.py
# -----------------

from pathlib import Path

def get_config():
    return{
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 500,
        "d_model": 512,
        "datasource": 'Hemanth-thunder/en_ta',
        "lang_src": "en",
        "lang_tgt": "ta",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/en_ta_model"
    }

def get_weights_file_path(config, epoch:str):
    model_folder = f"{config["datasource"].replace('/',"_")}_{config['model_folder']}"
    model_filename = f"{config["model_basename"]}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config["datasource"].replace("/", "_")}_{config["model_folder"]}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if not weights_files:
        return None
    weights_files.sort()
    return str(weights_files[-1])