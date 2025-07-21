"""
Implementation of whole slide image (WSI) training service.
"""

import io
import json
import os
import platform
import random
import time
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchinfo import summary
from tqdm import tqdm

from models.ViT import ViT
from pipeline.base_service import BaseService
from training.training_utils import (
    FeatureDataset, 
    get_patient_vector, 
    get_label,
    read_feature_information,
    select_tiles,
    FocalLoss,
)


class TrainingService(BaseService):

    def __init__(
        self, 
        model_directory: Union[str, Path],
        superbatch_directory: Union[str, Path], 
        pipeline_config_file: str,
        training_config: dict[str, Any],
        subset_config: dict[str, str],
        tile_division: dict[str, Any],
        device: str = 'cpu',
        num_workers: int = 0,
        pin_memory: bool = False,
        progress_bar: bool = False,
    ) -> None:
        """
        Initialize service for training.

        Args:
            model_directory:  Directory where all model files are stored.
            superbatch_directory:  Directory where all superbatches are stored.
            pipeline_config_file:  Name of pipeline config file.
            training_config:  Dictionary with training configuration.
            subset_config:  Dictionary with data subsets.
            tile_division:  Dictionary with per image the rank, origin, and tile indices.
            device:  Name of device for feature extraction model inference.
            num_workers:  Number of workers for the dataloader.
            pin_memory:  Indicates whether pinned memory is used for the dataloader.
            progress_bar:  Indicates whether a progress bar is used to show 
                training progress.
        """
        super().__init__(superbatch_directory, pipeline_config_file)

        # initialize additional instance attributes
        self.model_directory = Path(model_directory)
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.progress_bar = progress_bar

        # define settings attribute for training and data subsets
        self.settings = training_config
        self.subsets = subset_config
        self.tile_division = tile_division

        # create general model directory if it does not exist yet
        if not self.model_directory.exists():
            os.makedirs(self.model_directory)
            self.logger.info(f'Created directory: {self.model_directory}')

        # create experiment_folder
        self.experiment_directory = self.model_directory / self.settings['experiment_name']
        if self.experiment_directory.exists():
            raise FileExistsError(
                f'Experiment folder already exists: {self.settings["experiment_name"]}'
            )
        else:
            os.makedirs(self.experiment_directory)
            self.logger.info(f'Created directory: {self.experiment_directory}')
            # create subfolder for checkpoints
            self.checkpoint_directory = self.experiment_directory / 'checkpoints'
            os.makedirs(self.checkpoint_directory)
            self.logger.info(f'Created directory: {self.checkpoint_directory}')
            
    def start(self) -> None:
        """ 
        Start model training.
        """
        # perform check to determine if training is possible
        if not self.config['feature_extraction']:
            raise ValueError('Training is expected to be performed on feature vectors.')

        self.load_status()
        # check if feature extraction has completed
        variant = self.settings['variant']       
        self.logger.info((
            'Check if preprocessing and feature extraction have been completed '
            f'for the following variant: {variant}'
        ))
        continue_checking = True
        while continue_checking:
            all_completed = True
            if variant not in self.status:
                all_completed = False
            elif not self.status[variant]['feature_extraction_completed']:
                all_completed = False
            
            if all_completed:
                continue_checking = False
                self.logger.info((
                    'Preprocessing and feature extraction have been completed '
                    f'for the following variant: {variant}'
                ))
            else:
                self.logger.info(f'Retry check after {self.pause_duration} seconds')
                time.sleep(self.pause_duration)
                self.load_status()

        # start training
        self.train_model()
        self.logger.info('Finished training')
    
    def train_model(self) -> None:
        """
        Start training.
        """
        # save and log training settings
        with open(self.experiment_directory / self.settings['config_file'], 'w') as f:
            json.dump(self.settings, f)
        self.logger.info(f'Training config: {str(self.settings)}')

        # save subsets
        with open(self.experiment_directory / self.settings['subsets_file'], 'w') as f:
            json.dump(self.subsets, f)

        # configure the device
        if self.device == 'cuda':
            if torch.cuda.is_available():
                self.logger.info(f'CUDA available: {torch.cuda.is_available()}')
            else:
                self.device = 'cpu'
                self.logger.info('CUDA unavailable')
        self.logger.info(f'Device: {self.device}')

        # seed randomness
        seed = self.config['seed'] + (self.settings['seed'] if 'seed' in self.settings else 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        self.logger.info(f'Using seed: {seed}')

        # selected data variant
        self.variant = self.settings['variant']

        # get all superbatches for the first epoch of the variant
        superbatches = list(
            self.status[self.variant]['epochs']['0']['superbatches'].keys(),
        )
        feature_information = {}
        for superbatch in superbatches:
            # define paths to the feature information file 
            # and the feature directory
            filename = self.config['feature_extraction_settings']['output_filename']
            feature_information_path = self.directory / superbatch / filename
            feature_directory = self.directory / superbatch / self.optional_folders[1]
            
            # read the feature information for the superbatch 
            superbatch_feature_information = read_feature_information(
                feature_information_path=feature_information_path,
                feature_directory=feature_directory,
            )
            # add it to the combined feature information dictionary
            feature_information = {
                **superbatch_feature_information,
                **feature_information, 
            }
        
        # create tile selection based on settings
        tile_selection = select_tiles(
            tile_division=self.tile_division, 
            origin=self.settings['origin'],
            rank=self.settings['rank'],
        )         
        # create the subset mapping
        subset_mapping = {
            value: key for key, values in self.settings['subsets_config'].items() for value in values
        }
        # store the feature information per subset
        feature_information_subsets = {'training': {}, 'validation': {}}
        for key, feature in feature_information.items():
            additional_information = self.subsets[feature['specimen']]
            subset = subset_mapping[additional_information['subset']]
            # for the Spitz subtyping and dignity prediction task, skip all melanoma cases
            if (self.settings['task'].lower() in ['spitz subtype', 'spitz dignity']
                and feature['category'].lower() == 'conv melanoma'):
                continue
            # skip all cases which do not have any features 
            # (e.g., training on internal but only consultation slides available)
            elif not len(tile_selection[key]):
                continue
            elif subset in feature_information_subsets:
                feature = {**feature, **additional_information}
                feature_information_subsets[subset][key] = (feature)

        # define patient vector function
        if self.settings['model_config']['model_arguments']['patient_dim'] is None:
            patient_vector_func = None
        else:
            patient_vector_func = partial(
                get_patient_vector, 
                mul_factor=self.settings['training_config']['patient_mul_factor'],
            )

        # constuct dictionary to map class indices to weights for balancing classes
        if self.settings['training_config']['class_weights'] == 'balanced':
            train_labels = [get_label(value, self.settings['task']).tolist().index(1) 
                            for value in feature_information_subsets['training'].values()]
            class_counts = {}
            for label in train_labels:
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
            class_weights = [sum(class_counts.values())/(len(class_counts)*class_counts[i]) 
                             for i in range(len(class_counts))]
        else:
            class_weights = self.settings['training_config']['class_weights']

        # initialize training dataset and dataloader instances
        # NOTE: because the sequences of feature vectors differ per specimen,
        # a batch size of 1 with gradient accumulation is used
        training_dataset = FeatureDataset(
            feature_information=feature_information_subsets['training'],
            selection=tile_selection,
            task=self.settings['task'],
            patient_vector_func=patient_vector_func,
            label_func=get_label,
            weight_func=None,
            length=(self.settings['training_config']['iterations']
                    * self.settings['training_config']['batch_size']),
            section_dropout_prob=self.settings['training_config']['section_dropout_prob'],
            feature_dropout_prob=self.settings['training_config']['feature_dropout_prob'],
            maximum_features=self.settings['training_config']['maximum_features'],
        )
        train_dataloader = DataLoader(
            dataset=training_dataset,
            sampler= RandomSampler(training_dataset, replacement=True),
            batch_size=self.settings['training_config']['batch_size'],
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        # initialize validation dataset and dataloader instances
        validation_dataset = FeatureDataset(
            feature_information=feature_information_subsets['validation'],
            selection=tile_selection,
            task=self.settings['task'],
            patient_vector_func=patient_vector_func,
            label_func=get_label,
            weight_func=None,
            length=None,
            repetitions=self.settings['training_config']['eval_repetitions'],
            section_dropout_prob=0.0,
            feature_dropout_prob=0.0,
            maximum_features=self.settings['training_config']['maximum_features'],
        )
        validation_dataloader = DataLoader(
            dataset=validation_dataset,
            sampler= SequentialSampler(validation_dataset),
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        # record number of training and validation cases to check correctness
        self.logger.info(f'Training cases: {len(feature_information_subsets["training"])}')
        self.logger.info(f'Validation cases: {len(feature_information_subsets["validation"])}')

        # select model type
        if self.settings['model'] == 'ViT':
            network = ViT
        else:
            ValueError(f'Invalid model name: {self.settings["model"]}')
        
        # initialize model
        model = network(**self.settings['model_config']['model_arguments'])
        # transfer the model to the selected device
        model = model.to(self.device)

        # capture model summary in variable
        f = io.StringIO()
        with redirect_stdout(f):
            summary(model=model, depth=4, col_names=['num_params'])
        self.logger.info('\n'+f.getvalue())
        self.logger.info(model)
            
        # compile the model
        if (self.settings['model_config']['compile_model'] 
            and platform.system() == 'Linux'):
            model = torch.compile(model)

        # initialize optimizer
        learning_rate = self.settings['training_config']['learning_rate']
        if self.settings['training_config']['optimizer'] == 'AdamW':
            optimizer = Adam(model.parameters(), lr=learning_rate)        
        elif self.settings['training_config']['optimizer'] == 'Adam':
            optimizer = AdamW(model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError(
                'Specified optimizer not implemented: '
                f'{self.settings["training_config"]["optimizer"]}.'
            )

        # initialize loss function
        if self.settings['training_config']['loss_function'] == 'CrossEntropyLoss':
            loss_function = FocalLoss(gamma=0, class_weights=class_weights)
        elif self.settings['training_config']['loss_function'] == 'FocalLoss':
            loss_function = FocalLoss(
                gamma=self.settings['training_config']['focal_gamma'],
                class_weights=class_weights,
            )
        else:
            raise NotImplementedError(
                'Specified loss function not implemented: '
                f'{self.settings["training_config"]["loss_function"]}.'
            )

        # define progress bar if specified
        progress = lambda x: (tqdm(x, total=self.settings['training_config']['iterations']) 
                              if self.progress_bar else x)

        # start training loop
        specimen_indices = []
        accumulated_loss = []
        training_loss = []
        training_index = []
        validation_loss = []
        validation_index = []
        best_validation_loss = None
        for i, batch in progress(enumerate(train_dataloader)):
            index = i+1
            # update learning rate
            if index % self.settings['training_config']['iterations_per_decay'] == 0:
                learning_rate *= self.settings['training_config']['decay_factor']
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate
                self.logger.info(f'Iteration {str(index).zfill(7)} - '
                                 f'Learning rate set to {learning_rate:.1e}')

            # ---------------- TRAINING -------------------

            # get the specimen index
            specimen_indices.append(int(batch[-1]))
            batch = batch[:-1]

            if True:
                # bring the data to the correct device
                batch = [item.to(self.device) for item in batch]
                x, x_patient, pos, weight, y = batch

                # convert dummy patient vector with NaNs to None
                x_patient = None if torch.sum(torch.isnan(x_patient)) > 0 else x_patient

                # get model prediction
                y_pred = model(x, x_patient, pos)

                # calculate loss and account for gradient accumulation
                loss = loss_function(y_pred, y, weight)
                loss /= self.settings['training_config']['iterations_per_update']
                accumulated_loss.append(loss.item())
                
                # perform the backwards pass
                loss.backward()

            # for debugging purposes
            if False: 
                for item in batch:
                    print(item.shape)

            if index % self.settings['training_config']['iterations_per_update'] == 0:
                # update the network parameters and reset the gradient
                optimizer.step()
                optimizer.zero_grad() # set the gradient to 0 again

                # calculate the training loss for the batch
                training_loss.append(sum(accumulated_loss))
                training_index.append(index)
                self.logger.info((
                    f'Iteration {str(index).zfill(7)} - '
                    f'Training loss: {training_loss[-1]:0.4f} - '
                    f'Specimen indices: {specimen_indices}'
                ))
                specimen_indices = []
                accumulated_loss = []

            # --------------- VALIDATION ------------------
            # periodically evaluate on the validation set
            if index % self.settings['training_config']['iterations_per_checkpoint'] == 0:

                # set the model in evaluation mode
                model.eval()

                loss_per_image = []
                # deactivate autograd engine (backpropagation not required here)
                with torch.no_grad():
                    for batch in validation_dataloader:

                        # get the specimen index
                        specimen_indices.append(int(batch[-1]))
                        batch = batch[:-1]

                        if True:
                            # bring the data to the correct device
                            batch = [item.to(self.device) for item in batch]
                            x, x_patient, pos, weight, y = batch

                            # for debugging purposes
                            if False: 
                                for item in batch:
                                    print(item.shape)

                            # get model prediction
                            y_pred = model(x, x_patient, pos)

                            # calculate loss
                            loss = loss_function(y_pred, y, weight)
                            loss_per_image.append(loss.item())

                validation_loss.append(sum(loss_per_image)/len(loss_per_image))
                validation_index.append(index)
                self.logger.info((
                    f'Iteration {str(index).zfill(7)} - '
                    f'Validation loss: {validation_loss[-1]:0.4f} - '
                    f'Specimen indices: {specimen_indices}'
                ))
                specimen_indices = []

                # determine if the last model checkpoint achieved the best validation loss
                save_checkpoint = False
                if best_validation_loss is None:
                    best_validation_loss = validation_loss[-1]
                    save_checkpoint = True
                elif best_validation_loss > validation_loss[-1]:
                    best_validation_loss = validation_loss[-1]
                    save_checkpoint = True
                elif 'save_all_checkpoints' in self.settings:
                    if self.settings['save_all_checkpoints']:
                        save_checkpoint = True

                # save model checkpoint
                if save_checkpoint:
                    torch.save({
                        'iteration': index,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': validation_loss[-1],
                        },
                        self.checkpoint_directory / f'checkpoint_I{str(index).zfill(7)}.tar',
                    )

                # set the model to training mode
                model.train()

        # save training and validation loss as excel file
        train_loss_df = pd.DataFrame.from_dict({'index': training_index,
                                                'loss': training_loss})
        validation_loss_df = pd.DataFrame.from_dict({'index': validation_index,
                                                     'loss': validation_loss})
        
        # create a excel writer object
        with pd.ExcelWriter(self.experiment_directory / 'loss.xlsx') as writer:
            train_loss_df.to_excel(writer, sheet_name='Training', index=False)
            validation_loss_df.to_excel(writer, sheet_name="Validation", index=False)

        # plot training and validation loss
        fig, ax = plt.subplots()

        # plot losses
        ax.plot(training_index, training_loss, zorder=1,
                color='royalblue', alpha=0.40, label=f'Train')
        ax.plot(validation_index, validation_loss, 
                color='forestgreen', zorder=2, label=f'Validation')
        ax.scatter(validation_index, validation_loss, zorder=2, marker='o', 
                   facecolor='white', edgecolor='forestgreen', lw=1.5, s=15)

        # change axis setup
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_ylim(bottom=0)
        plt.xlim([-25, training_index[-1]+25])
        plt.legend()
        plt.savefig(os.path.join(self.experiment_directory, 'loss.png'), dpi=300)
