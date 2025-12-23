# Add ai_worker_modality.json file for AI worker collected data -> move the file to the data meta with modality.json name

# Created ai_worker_config.py to register the new embodiment configuration

# Update stats.py to load the new embodiment config when making stats for the new embodiment data
'''
# Load new embodiment modality config
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")
    
load_modality_config("data/ai_worker/ai_worker_config.py")
'''
# Run stats.py to make stats and rel stats file for the new embodiment data
python gr00t/data/stats.py --dataset-path data/ffw_sg2_rev1_clear_item --embodiment-tag NEW_EMBODIMENT