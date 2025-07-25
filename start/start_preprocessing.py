"""
Start preprocessing data.
"""

import platform

from pipeline.preprocessing_service import PreprocessingService
from pipeline.preprocessing_utils import get_unique_filenames

# define paths
if platform.system() == 'Linux':
    superbatch_directory = '/projects/spitz_classification/superbatches'
    device = 'cuda'
elif platform.system() == 'Windows':
    superbatch_directory = r"projects\spitz_classification\superbatches"
    device = 'cpu'
else:
    raise NotImplementedError
pipeline_config_file = r'pipeline.json'

if __name__ == '__main__':

    service = PreprocessingService(superbatch_directory, pipeline_config_file, 
                                   get_unique_filenames, device=device)
    service.start()