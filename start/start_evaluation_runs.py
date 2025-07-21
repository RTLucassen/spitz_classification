"""
Start model evaluation.
"""

import json
import platform

import numpy as np

from evaluation.evaluation_service import EvaluationService

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

pipeline_config_file = 'pipeline.json'
training_config_file = 'training.json'
subsets_file = 'subsets.json'
tile_division_path = r'configs/hipt_vit4k_tiles.json'

# define experiment names
experiment_names = [
    '001__hipt_4k__val_fold-1',
    '002__hipt_4k__val_fold-2',
    '003__hipt_4k__val_fold-3',
    '004__hipt_4k__val_fold-4',
    '005__hipt_4k__val_fold-5',
    '006__hipt_4k__val_fold-1',
    '007__hipt_4k__val_fold-2',
    '008__hipt_4k__val_fold-3',
    '009__hipt_4k__val_fold-4',
    '010__hipt_4k__val_fold-5',
    '011__hipt_4k__val_fold-1',
    '012__hipt_4k__val_fold-2',
    '013__hipt_4k__val_fold-3',
    '014__hipt_4k__val_fold-4',
    '015__hipt_4k__val_fold-5',
]

if __name__ == '__main__':

    # load tile selection
    with open(tile_division_path, 'r') as f:
        tile_division = json.loads(f.read())

    for experiment_name in experiment_names:
        service = EvaluationService(
            experiment_name=experiment_name,
            thresholds=np.arange(0, 1.001, 0.005),
            bootstrap_iterations=None,
            bootstrap_stratified=True,
            confidence_level=0.95,
            calibration_bins=10,
            repetitions=10,
            model_directory=model_directory,
            superbatch_directory=superbatch_directory,
            pipeline_config_file=pipeline_config_file,
            training_config_file=training_config_file,
            subsets_file=subsets_file,
            tile_division=tile_division,
            device=device,
            num_workers=workers,
            pin_memory=pin_memory,
            image_extension='png',
            verbose=True,
        )
        service.start(['internal', 'consultation'], ['validation', 'test'], '')