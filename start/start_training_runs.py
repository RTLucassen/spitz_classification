"""
Start model training.
"""

import json    
import platform
from natsort import natsorted

from training.training_service import TrainingService

# define paths
if platform.system() == 'Linux':
    superbatch_directory = '/projects/spitz_classification/superbatches'
    model_directory = '/projects/spitz_classification/models'
    device = 'cuda'
    workers = 4
    pin_memory = True
elif platform.system() == 'Windows':
    superbatch_directory = r"projects\spitz_classification\superbatches"
    model_directory = r"projects\spitz_classification\models"
    device = 'cpu'
    workers = 0
    pin_memory = False
else:
    raise NotImplementedError

pipeline_config_file = r'pipeline.json'
training_config_directory = r'training_configs'
subset_config_path = r'configs/subsets.json'
tile_division_path = r'configs/hipt_vit4k_tiles.json'

if __name__ == '__main__':

    # load subset config
    with open(subset_config_path, 'r') as f:
        subset_config = json.loads(f.read())

    # load tile selection
    with open(tile_division_path, 'r') as f:
        tile_division = json.loads(f.read())

    # loop over training run configurations
    for training_config_path in natsorted(training_config_directory.iterdir()):
        # load training config
        with open(training_config_path, 'r') as f:
            training_config = json.loads(f.read())

        try:
            service = TrainingService(
                model_directory=model_directory,
                superbatch_directory=superbatch_directory,
                pipeline_config_file=pipeline_config_file,
                training_config=training_config,
                subset_config=subset_config,
                tile_division=tile_division,
                device=device,
                num_workers=workers,
                pin_memory=pin_memory,
            )
            service.start()
        except FileExistsError:
            print('Continuing with the next training configuration.')