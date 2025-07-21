"""
Start preprocessing data.
"""

import platform

from pipeline.feature_extraction_utils import extract_ViT_features
from pipeline.feature_extraction_service import FeatureExtractionService

# define paths
if platform.system() == 'Linux':
    superbatch_directory = '/projects/spitz_classification/superbatches'
    device = 'cuda'
    workers = 4
    pin_memory = True
elif platform.system() == 'Windows':
    superbatch_directory = r"projects\spitz_classification\superbatches"
    device = 'cpu'
    workers = 0
    pin_memory = False,
else:
    raise NotImplementedError
pipeline_config_file = r'pipeline.json'

if __name__ == '__main__':
    
    service = FeatureExtractionService(superbatch_directory, pipeline_config_file, 
                                       extract_ViT_features, device=device, 
                                       num_workers=workers, pin_memory=pin_memory)
    service.start()