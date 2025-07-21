"""
Implementation of whole slide image (WSI) evaluation service.
"""

import io
import json
import os
import platform
import random
import shutil
import stat
import time
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from natsort import natsorted
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
)
from torch.utils.data import DataLoader, SequentialSampler
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
)

class EvaluationService(BaseService):

    remove_existing_folder = True

    def __init__(
        self, 
        experiment_name: str,
        thresholds: Union[float, list[float]],
        bootstrap_iterations: Optional[int],
        bootstrap_stratified: bool,
        confidence_level: float,
        calibration_bins: int,
        repetitions: int,
        model_directory: Union[str, Path],
        superbatch_directory: Union[str, Path], 
        pipeline_config_file: str,
        training_config_file: str,
        subsets_file: str,
        tile_division: dict[str, Any],
        device: str = 'cpu',
        num_workers: int = 0,
        pin_memory: bool = False,
        image_extension: str = '.png',
        verbose: bool = False,
    ) -> None:
        """
        Initialize service for evaluation.

        Args:
            experiment_name:  Name of experiment folder 
            thresholds:  One or more binarization thresholds for evaluating 
                the model performance.
            bootstrap_iterations:  Number of iterations used for bootstrapping 
                to calculate confidence intervals.
            bootstrap_stratified:  Indicates whether stratified sampling is used.
            confidence_level:  Confidence level for interval calculated using bootstrapping.
            calibration_bins:  Number of bins in calibration plot and calculation.
            repetitions:  Number of times each case is sampled if no fixed length is used.
            model_directory:  Directory where all model files are stored.
            superbatch_directory:  Directory where all superbatches are stored.
            pipeline_config_file:  Name of pipeline config file.
            training_config_file:  Name of training config file.
            subsets_file:  Name of subsets file.
            tile_division:  Dictionary with per image the rank, origin, and tile indices.
            device:  Name of device for feature extraction model inference.
            num_workers:  Number of workers for the dataloader.
            pin_memory:  Indicates whether pinned memory is used for the dataloader.
            image_extension:  Name of image extension for saving plots of results.
            verbose:  Indicates whether completing inference for each case is logged.
        """
        super().__init__(superbatch_directory, pipeline_config_file)

        # initialize additional instance attributes
        self.experiment_name = experiment_name
        self.thresholds = thresholds
        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_stratified = bootstrap_stratified
        self.confidence_level = confidence_level
        self.calibration_bins = calibration_bins
        self.repetitions = repetitions
        self.model_directory = Path(model_directory)
        self.experiment_directory = self.model_directory / self.experiment_name
        self.tile_division = tile_division
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_extension = image_extension
        self.verbose = verbose
        self.name_addition = ''

        # load training config and subsets
        with open(self.experiment_directory / training_config_file, 'r') as f:
            self.settings = json.loads(f.read())
        with open(self.experiment_directory / subsets_file, 'r') as f:
            self.subsets = json.loads(f.read())

        # check if model and experiment directory exist
        if not self.model_directory.exists():
            raise FileExistsError('Directory for models does not exist.')
        elif not self.experiment_directory.exists():
            raise FileExistsError('Experiment directory does not exist.')
        
    def start(
        self, 
        origins: Union[list[str], str], 
        subsets: Union[list[str], str],
        name_addition: str = '',
    ) -> None:
        """ 
        Start the evaluation after checking if the data has been transferred 
        and preprocessed (and optionally features extracted).

        Args:
            origins:  Origins of data on which to evaluate the model 
                (e.g., internal, consultation).
            subsets:  Subsets of dataset on which to evaluate the model
                (e.g, training, validation, test).
            name_addition:  String added to the end of the evaluation name.
        """
        # perform check to determine if evaluation is possible
        if not self.config['feature_extraction']:
            raise ValueError('Training is expected to be performed '
                             'on feature vectors.')

        # if a single subset was specified, put it in a list
        if isinstance(subsets, str):
            subsets = [subsets]

        # check if the specified subsets are valid
        for subset in subsets:
            if subset not in ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5', 
                              'training', 'validation', 'test']:
                raise ValueError(f'Invalid subset: {subset}')

        if 'validation' in subsets:
            subsets.extend(self.settings['subsets_config']['validation'])
            subsets.remove('validation')

        # if a single variant was specified, put it in a list
        variants = self.settings['variant']
        if isinstance(variants, str):
            variants = [variants]

        self.load_status()
        # check if the specified variants are valid
        for variant in variants:
            if variant not in self.config['variants']:
                raise ValueError(f'Invalid variant: {variant}')

        self.logger.info((
            'Check if preprocessing and feature extraction have been completed '
            f'for the following variants: {", ".join(variants)}'
        ))
        continue_checking = True
        while continue_checking:
            all_completed = True
            for variant in variants:
                if variant not in self.status:
                    all_completed = False
                elif not self.status[variant]['feature_extraction_completed']:
                    all_completed = False
            if all_completed:
                continue_checking = False
                self.logger.info((
                    'Preprocessing and feature extraction have been completed '
                    f'for the following variants: {", ".join(variants)}'
                ))
            else:
                self.logger.info(f'Retry check after {self.pause_duration} seconds')
                time.sleep(self.pause_duration)
                self.load_status()

        # save name addition
        self.name_addition = name_addition

        # start evaluation
        for origin in origins:
            for subset in subsets:
                self.evaluate_model(origin, subset)
        self.logger.info('Finished evaluation')
    
    def evaluate_model(self, origin: str, subset: str) -> None:
        """
        Start model evaluation.

        Args:
            origin:  Origin of the data on which to evaluate the model.
            subset:  Subset of dataset on which to evaluate the model.
        """
        # create results directory
        results_directory = self.experiment_directory / f'results_{origin}_{subset}_{self.settings["task"]}{self.name_addition}'
        if results_directory.exists():
            if self.remove_existing_folder:
                shutil.rmtree(results_directory, onerror=on_rm_error)
            else:
                raise FileExistsError('Results directory already exists.')
        results_directory.mkdir()

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

        # select model type
        if self.settings['model'] == 'ViT':
            network = ViT
        else:
            ValueError(f'Invalid model name: {self.settings["model"]}')

        # initialize model
        model = network(**self.settings['model_config']['model_arguments'])
        # get the last checkpoint saved
        checkpoints_directory = self.experiment_directory / 'checkpoints'
        last_checkpoint = natsorted(os.listdir(checkpoints_directory))[-1]
        self.logger.info(f'Load the model parameters from last checkpoint: {last_checkpoint}')
        # load model parameters
        state_dict = torch.load(checkpoints_directory / last_checkpoint, 
                                map_location=self.device)
        model.load_state_dict(state_dict=state_dict['model_state_dict'])
        model.to(self.device)
        model.eval()

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

        self.logger.info(f'Start model evaluation on subset: {subset}, '
                         f'for origin: {origin}.')

        # create tile selection based on settings
        tile_selection = select_tiles(
            tile_division=self.tile_division, 
            origin=origin if origin != 'combined' else ['internal', 'consultation'],
            rank=self.settings['rank'],
        )  
        # get all superbatches for the first epoch of the variant
        superbatches = list(
            self.status[self.settings['variant']]['epochs']['0']['superbatches'].keys(),
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

        # store the feature information per subset
        feature_information_subset = {}
        for key, feature in feature_information.items():
            additional_information = self.subsets[feature['specimen']]
            # for the Spitz subtyping and dignity prediction task, skip all melanoma cases
            if (self.settings['task'].lower() in ['spitz subtype', 'spitz dignity']
                and feature['category'].lower() == 'conv melanoma'):
                continue
            # skip all cases which do not have any features 
            # (e.g., training on internal but only consultation slides available)
            elif not len(tile_selection[key]):
                continue
            elif additional_information['subset'] == subset:
                feature = {**feature, **additional_information}
                feature_information_subset[key] = (feature)

        # define patient vector function
        if self.settings['model_config']['model_arguments']['patient_dim'] is None:
            patient_vector_func = None
        else:
            patient_vector_func = partial(
                get_patient_vector, 
                mul_factor=self.settings['training_config']['patient_mul_factor'],
            )

        # initialize dataset and dataloader instances
        dataset = FeatureDataset(
            feature_information=feature_information_subset,
            selection=tile_selection,
            task=self.settings['task'],
            patient_vector_func=patient_vector_func,
            label_func=get_label,
            length=None,
            section_dropout_prob=0.0,
            feature_dropout_prob=0.0,
            maximum_features=25000,
            repetitions=self.repetitions,
            only_repeat_cases_exceeding_maximum=True,
        )
        dataloader = DataLoader(
            dataset=dataset,
            sampler= SequentialSampler(dataset),
            batch_size=1,
            num_workers=0,
            pin_memory=self.pin_memory,
        )

        results = {}
        with torch.no_grad():
            for (x, x_patient, pos, _, y, index) in dataloader:
                # get the specimen index
                specimen_index = int(index)
                if specimen_index not in results:
                    results[specimen_index] = {
                        'y_true': [], 
                        'y_pred': [], 
                        'feat_vector': [],
                        'attn_dicts': [],
                    }
                # bring the data to the correct device
                x = x.to(self.device)
                x_patient = x_patient.to(self.device)
                pos = pos.to(self.device)

                # get model prediction and optionally the self-attention
                pred, feat, attn = model(
                    x=x, x_patient=x_patient, pos=pos,
                    return_last_feat_vector=True, 
                    return_last_self_attention=True,
                )
                pred = torch.softmax(pred.to('cpu'), dim=1)

                # add true and predicted label to lists
                results[specimen_index]['y_true'].append(y[0, :].tolist())
                results[specimen_index]['y_pred'].append(pred[0, :].tolist())
                results[specimen_index]['feat_vector'].append(feat[0, :].tolist())

                # save attention for last layer
                if attn is not None:
                    attn_dict = {}
                    for i in range(pos.shape[1]):
                        position = str(tuple(pos[0, i, :].tolist()))
                        attn_values = [round(v, 3) for v in attn[0, :, 0, i+1].tolist()]
                        attn_dict[position] = attn_values
                    results[specimen_index]['attn_dicts'].append(attn_dict)
                
                if self.verbose:
                    self.logger.info(f'Finished inference for specimen {index[0]}.')

        # save JSON with attention values
        if attn is not None:
            with open(results_directory / f'results_{origin}_{subset}.json', 'w') as f:
                f.write(json.dumps(results))

        # save prediction result per specimen
        predictions = {
            'specimen_index': [item['specimen_index'] for item in feature_information_subset.values()],
            'patient': [item['patient'] for item in feature_information_subset.values()],
            'sex': [item['sex'] for item in feature_information_subset.values()],
            'age': [item['age'] for item in feature_information_subset.values()],
            'location': [item['location'] for item in feature_information_subset.values()],
            'specimen': [item['specimen'] for item in feature_information_subset.values()],
            'diagnosis': [item['diagnosis'] for item in feature_information_subset.values()],
            'y_true': [], 'y_pred': [], 'y_pred_stdev': [],
        }

        # get the mean and standard deviation over the repititions
        for specimen_index in predictions['specimen_index']:
            predictions['y_true'].append([mean(values) for values in list(zip(*results[specimen_index]['y_true']))])
            predictions['y_pred'].append([mean(values) for values in list(zip(*results[specimen_index]['y_pred']))])
            predictions['y_pred_stdev'].append(
                [None if len(values) < 2 else stdev(values) for values in list(zip(*results[specimen_index]['y_pred']))]
            )
        # perform evaluation for binary classification task
        N_classes = self.settings['model_config']['model_arguments']['n_classes']
        if N_classes == 2:
            # perform threshold-dependent evaluation
            threshold_results = evaluate_with_thresholds(
                y_true=[pred[1] for pred in predictions['y_true']], 
                y_pred=[pred[1] for pred in predictions['y_pred']], 
                thresholds=self.thresholds,
            )
            # perform threshold-independent evaluation 
            area_results = evaluate_without_thresholds(
                y_true=[pred[1] for pred in predictions['y_true']], 
                y_pred=[pred[1] for pred in predictions['y_pred']],
                bins=self.calibration_bins,                                                    
                roc_figure_path=results_directory / f'ROC_curve_{origin}_{subset}.{self.image_extension.replace(".", "")}',
                pr_figure_path=results_directory / f'PR_curve_{origin}_{subset}.{self.image_extension.replace(".", "")}',
                cal_figure_path=results_directory / f'Calibration_{origin}_{subset}.{self.image_extension.replace(".", "")}',
            )
            # perform bootstrapping for confidence intervals
            if self.bootstrap_iterations is not None:
                bootstrap_output = {'accuracy': {'threshold':[]}, 'AUROC': {}}
                for threshold in self.thresholds:
                    # define accuracy wrapper
                    def accuracy_wrapper(y_true, y_pred):
                        return evaluate_with_thresholds(y_true, y_pred, threshold)['accuracy'][0]
                    # perform bootstrapping for accuracy at specific threshold value
                    bootstrap_accuracy_output = bootstrapper(
                        y_true=[pred[1] for pred in predictions['y_true']], 
                        y_pred=[pred[1] for pred in predictions['y_pred']], 
                        eval_func=accuracy_wrapper, 
                        iterations=self.bootstrap_iterations, 
                        stratified=self.bootstrap_stratified,
                        confidence_level=self.confidence_level, 
                        seed=seed,
                        show_progress=False,
                    )
                    bootstrap_output['accuracy']['threshold'].append(threshold)
                    for key, value in bootstrap_accuracy_output.items():
                        if key not in bootstrap_output['accuracy']:
                            bootstrap_output['accuracy'][key] = [value]
                        else:
                            bootstrap_output['accuracy'][key].append(value)

                # define AUROC wrapper
                def AUROC_wrapper(y_true, y_pred):
                    return evaluate_without_thresholds(y_true, y_pred, self.calibration_bins)['AUROC'][0]
                # perform bootstrapping for AUROC
                bootstrap_AUROC_output = bootstrapper(
                    y_true=[pred[1] for pred in predictions['y_true']], 
                    y_pred=[pred[1] for pred in predictions['y_pred']], 
                    eval_func=AUROC_wrapper, 
                    iterations=self.bootstrap_iterations, 
                    stratified=self.bootstrap_iterations,
                    confidence_level=self.confidence_level,
                    seed=seed,
                    show_progress=False,
                )
                for key, value in bootstrap_AUROC_output.items():
                    if key not in bootstrap_output['AUROC']:
                        bootstrap_output['AUROC'][key] = [value]
                    else:
                        bootstrap_output['AUROC'][key].append(value)

        # perform evaluation for multi-class classification task
        elif N_classes > 2:
            # perform evaluation based the highest probability class
            threshold_results = evaluate_highest_probability(
                y_true=predictions['y_true'], 
                y_pred=predictions['y_pred'],
            )
            # perform threshold-independent evaluation per class
            area_results = {}
            for class_index in range(N_classes):
                cl = f'class_{class_index}'
                area_results_per_class = evaluate_without_thresholds(
                    y_true=[pred[class_index] for pred in predictions['y_true']], 
                    y_pred=[pred[class_index] for pred in predictions['y_pred']],
                    bins=self.calibration_bins,                                                    
                    roc_figure_path=results_directory / f'ROC_curve_{cl}_{origin}_{subset}.{self.image_extension.replace(".", "")}',
                    pr_figure_path=results_directory / f'PR_curve_{cl}_{origin}_{subset}.{self.image_extension.replace(".", "")}',
                    cal_figure_path=results_directory / f'Calibration_{cl}_{origin}_{subset}.{self.image_extension.replace(".", "")}',
                )
                for key, value in area_results_per_class.items():
                    area_results[f'{cl}_{key}'] = value

            # perform bootstrapping for confidence intervals
            if self.bootstrap_iterations is not None:
                bootstrap_output = {'accuracy': {}, 'AUROC': {}}
                # define accuracy wrapper
                def accuracy_wrapper(y_true, y_pred):
                    return evaluate_highest_probability(y_true, y_pred)['accuracy'][0]
                # perform bootstrapping for accuracy
                bootstrap_accuracy_output = bootstrapper(
                    y_true=predictions['y_true'], 
                    y_pred=predictions['y_pred'], 
                    eval_func=accuracy_wrapper,
                    iterations=self.bootstrap_iterations, 
                    stratified=self.bootstrap_stratified,
                    confidence_level=self.confidence_level, 
                    seed=seed,
                    show_progress=False,
                )
                for key, value in bootstrap_accuracy_output.items():
                    if key not in bootstrap_output['accuracy']:
                        bootstrap_output['accuracy'][key] = [value]
                    else:
                        bootstrap_output['accuracy'][key].append(value)

                for class_index in range(N_classes):
                    cl = f'class_{class_index}'
                    # define AUROC wrapper
                    def AUROC_wrapper(y_true, y_pred):
                        return evaluate_without_thresholds(y_true, y_pred, self.calibration_bins)['AUROC'][0]
                    # perform bootstrapping for AUROC
                    bootstrap_AUROC_output = bootstrapper(
                        y_true=[pred[class_index] for pred in predictions['y_true']], 
                        y_pred=[pred[class_index] for pred in predictions['y_pred']], 
                        eval_func=AUROC_wrapper, 
                        iterations=self.bootstrap_iterations, 
                        stratified=self.bootstrap_stratified,
                        confidence_level=self.confidence_level,
                        seed=seed,
                        show_progress=False,
                    )
                    for key, value in bootstrap_AUROC_output.items():
                        if key not in bootstrap_output['AUROC']:
                            bootstrap_output['AUROC'][f'{cl}_{key}'] = [value]
                        else:
                            bootstrap_output['AUROC'][f'{cl}_{key}'].append(value)

        # convert dictionaries to dataframes
        predictions_df = pd.DataFrame.from_dict(predictions)
        threshold_results_df = pd.DataFrame.from_dict(threshold_results)
        area_results_df = pd.DataFrame.from_dict(area_results)
        if self.bootstrap_iterations is not None:
            bootstrap_accuracy_df =  pd.DataFrame.from_dict(bootstrap_output['accuracy'])
            bootstrap_AUROC_df = pd.DataFrame.from_dict(bootstrap_output['AUROC'])

        # save evaluation results in spreadsheet
        with pd.ExcelWriter(results_directory / f'results_{origin}_{subset}.xlsx') as writer:
            predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
            threshold_results_df.to_excel(writer, sheet_name='Results (threshold)', index=False)
            area_results_df.to_excel(writer, sheet_name='Results (area)', index=False)
            if self.bootstrap_iterations is not None:
                bootstrap_accuracy_df.to_excel(writer, sheet_name='Bootstrap accuracy results', index=False)
                bootstrap_AUROC_df.to_excel(writer, sheet_name='Bootstrap AUROC results', index=False)


def evaluate_with_thresholds(
    y_true: list[int], 
    y_pred: list[float], 
    thresholds: Union[float, list[float]],
) -> dict[str, list]:
    """ 
    Evaluate predictive performance of model using threshold-dependent metrics.

    Args:
        y_true:  True label per case.
        y_pred:  Predicted probability by the model per case.
        thresholds:  One or more thresholds to use for evaluation.

    Returns:
        results:  Results of evaluation per threshold.
    """
    # define dictionary to save model performance in terms of several metrics 
    # for each threshold      
    results = {'threshold': [], 'TP': [], 'TN': [], 'FP': [], 'FN': [],
               'sensitivity / recall': [], 'specificity': [], 'precision': [], 
               'f1_score': [], 'accuracy': [], 'balanced_accuracy': []}
    
    # if a single threshold was provided, put it in a list
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    # calculate performance metrics for each threshold
    for threshold in thresholds:
        binary_y_pred = [1 if pred > threshold else 0 for pred in y_pred]
        combinations = zip(binary_y_pred, y_true)
        counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for combination in combinations:
            if combination == (1, 1):
                counts['TP'] += 1
            elif combination == (0, 0):
                counts['TN'] += 1
            elif combination == (1, 0):
                counts['FP'] += 1
            elif combination == (0, 1):
                counts['FN'] += 1
        if (counts['TP'] + counts['FN']) == 0:
            sensitivity = None
        else:
            sensitivity = counts['TP'] / (counts['TP'] + counts['FN'])
        if (counts['TP'] + counts['FN']) == 0:
            specificity =  None
        else:
            specificity = counts['TN'] / (counts['TN'] + counts['FP'])
        if (counts['TP'] + counts['FP']) == 0:
            precision = None
        else:
            precision = counts['TP'] / (counts['TP'] + counts['FP'])
        f1_score = 2*counts['TP'] / (2*counts['TP'] + counts['FP'] + counts['FN'])
        accuracy = ((counts['TP']+counts['TN']) 
                    / (counts['TP'] + counts['TN'] + counts['FP'] + counts['FN']))
        if sensitivity is None or specificity is None:
            balanced_accuracy = None
        else:
            balanced_accuracy = (sensitivity + specificity) / 2

        # add values to results dictionary
        results['threshold'].append(threshold)
        results['TP'].append(counts['TP'])
        results['TN'].append(counts['TN'])
        results['FP'].append(counts['FP'])
        results['FN'].append(counts['FN'])
        results['sensitivity / recall'].append(sensitivity)
        results['specificity'].append(specificity)
        results['precision'].append(precision)
        results['f1_score'].append(f1_score)
        results['accuracy'].append(accuracy)
        results['balanced_accuracy'].append(balanced_accuracy)

    return results


def evaluate_highest_probability(
    y_true: list[int], 
    y_pred: list[float],
    one_hot_labels: bool = False,
) -> dict[str, list]:
    """ 
    Evaluate predictive performance of model using threshold-dependent metrics.

    Args:
        y_true:  True label per case.
        y_pred:  Predicted probability by the model per case.
        one_hot_labels:  Indicates whether the labels in y_true are one hot encodings.

    Returns:
        results:  Results of evaluation per threshold.
    """
    # define dictionary to save model performance in terms of several metrics 
    results = {}

    # get index of class with highest predicted probability
    y_pred_index = [pred.index(max(pred)) for pred in y_pred]
    if not one_hot_labels:
        y_true_index = [true.index(max(true)) for true in y_true]
    else:
        y_true_index = y_true
    combinations = list(zip(y_pred_index, y_true_index))
    results['accuracy'] = [sum([1 if c[0]==c[1] else 0 for c in combinations])/len(combinations)]

    # loop over the classes to compute per-class metrics
    for class_index in range(len(y_pred[0])):
        counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for combination in combinations:
            if combination[0] == combination[1] == class_index:
                counts['TP'] += 1
            elif (combination[0] != class_index) and (combination[1] != class_index):
                counts['TN'] += 1
            elif (combination[0] == class_index) and (combination[1] != class_index):
                counts['FP'] += 1
            elif (combination[0] != class_index) and (combination[1] == class_index):
                counts['FN'] += 1
        # define class prefix for metric names
        cl = f'class_{class_index}'
        # save all the metrics in the results dictionary
        for key in ['TP', 'TN', 'FP', 'FN']:
            results[f'{cl}_{key}'] = [counts[key]]
        if (counts['TP'] + counts['FN']) == 0:
            results[f'{cl}_sensitivity'] = [None]
        else:
            results[f'{cl}_sensitivity'] = [counts['TP'] / (counts['TP'] + counts['FN'])]
        if (counts['TN'] + counts['FP']) == 0:
            results[f'{cl}_specificity'] = [None]
        else:
            results[f'{cl}_specificity'] = [counts['TN'] / (counts['TN'] + counts['FP'])]
        if (counts['TP'] + counts['FP']) == 0:
            results[f'{cl}_precision'] = [None]
        else:
            results[f'{cl}_precision'] = [counts['TP'] / (counts['TP'] + counts['FP'])]
        results[f'{cl}_f1_score'] = [
            2*counts['TP'] / (2*counts['TP'] + counts['FP'] + counts['FN'])
        ]
        results[f'{cl}_accuracy'] = [
            ((counts['TP']+counts['TN']) / (counts['TP'] + counts['TN'] 
                                            + counts['FP'] + counts['FN']))
        ]
        if results[f'{cl}_sensitivity'][0] is None or results[f'{cl}_specificity'][0] is None:
            results[f'{cl}_balanced_accuracy'] = None
        else:
            results[f'{cl}_balanced_accuracy'] = [
                (results[f'{cl}_sensitivity'][0] + results[f'{cl}_specificity'][0]) / 2
            ]

    return results


def evaluate_without_thresholds(
    y_true: list[int], 
    y_pred: list[float], 
    bins: int,
    roc_figure_path: Optional[Union[str, Path]] = None,
    pr_figure_path: Optional[Union[str, Path]] = None,
    cal_figure_path: Optional[Union[str, Path]] = None,
) -> dict[str, list]:
    """
    Evaluate predictive performance of model using threshold-independent metrics
    (Area under ROC and PR curves).

    Args:
        y_true:  True label per case.
        y_pred:  Predicted probability by the model per case.
        scanner:  Name of scanner per case.
        bins:  Number of bins used in the calibration assessment.
        roc_figure_path:  Output path for saving ROC curve figure.
        pr_figure_path:  Output path for saving PR curve figure.
        cal_figure_path:  Output path for saving calibration figure.

    Returns:
        area_results:  Results of area under the curves evaluation.
    """
    # define dictionary to save model performance in terms of AUROC and AP 
    # per scanner type
    results = {'AUROC': [], 'AP': [], 'ECE': []}

    # plot ROC curve and calculate AUC-ROC
    if roc_figure_path is not None:
        fig, ax = plt.subplots(figsize=(5,5))

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.3)

        # format ticks
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="both", direction="in", length=5, width=1.3)
        ax.tick_params(which='minor', axis="both", direction="in", 
                        right='on', top='on', length=2.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.200))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.xaxis.set_major_locator(MultipleLocator(0.200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
        # format axes
        offset = 0.001
        plt.xlim([0-offset, 1+offset])
        plt.ylim([0-offset, 1+offset])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')

    # prepare ROC curve and calculate AUC-ROC
    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred)
        results['AUROC'].append(auroc)
        if roc_figure_path is not None:
            plt.plot(fpr, tpr, label=f'ROC curve (AUC: {auroc:0.3f})', 
                        lw=2, color="mediumblue")
    else:
        results['AUROC'].append(None)
    
    if roc_figure_path is not None and results['AUROC'] is not None:
        plt.plot([0, 1], [0, 1], ls='--', color='black', lw=0.75)
        plt.legend(loc=4, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
        plt.savefig(roc_figure_path)
        plt.close()

    # plot precision-recall (PR) curve and calculate AUC-PR
    if pr_figure_path is not None:
        fig, ax = plt.subplots(figsize=(5,5))

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        # format ticks
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="both", direction="in", length=5, width=1.3)
        ax.tick_params(which='minor', axis="both", direction="in", 
                        right='on', top='on', length=2.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.200))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.xaxis.set_major_locator(MultipleLocator(0.200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
        # format axes
        offset = 0.001
        plt.xlim([0-offset, 1+offset])
        plt.ylim([0-offset, 1+offset])
        plt.xlabel('Recall')
        plt.ylabel('Precision')

    # prepare PR curve and calculate AUC-PR
    if len(set(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        results['AP'].append(ap)
        if pr_figure_path is not None:
            plt.plot(recall, precision, label=f'PR curve (AUC: {ap:0.3f})', 
                    lw=2, color='mediumblue')
    else:
        results['AP'].append(None)

    if pr_figure_path is not None and results['AP'] is not None:
        plt.legend(loc=4, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
        plt.savefig(pr_figure_path)
        plt.close()

    # assign each case to a predicted probability bin and store the label
    labels = {i: [] for i in range(bins)}
    probabilities = {i: [] for i in range(bins)}
    for pred, true in zip(y_pred, y_true):
        index = min(int(pred*bins), bins-1)
        probabilities[index].append(pred)
        labels[index].append(true)
    
    # calculate the faction of true labels for each bin
    count = [len(labels[i]) for i in range(bins)]
    valid = [True if count[i] > 0 else False for i in range(bins)]
    true_prob = [sum(labels[i])/len(labels[i]) if len(labels[i]) > 0 else None for i in range(bins)]
    pred_prob = [sum(probabilities[i])/len(probabilities[i]) if len(probabilities[i]) > 0 else None for i in range(bins)]
    ece = sum([(count[i]/sum(count))*abs(true_prob[i]-pred_prob[i]) for i in range(bins) if valid[i]])
    results['ECE'].append(ece)

    # plot confidence calibration
    if cal_figure_path is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 5))

        for i in range(2):
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[i].spines[axis].set_linewidth(1.5)

            # format ticks
            ax[i].tick_params(bottom=True, top=True, left=True, right=True)
            ax[i].tick_params(axis="both", direction="in", length=5, width=1.3)
        ax[0].xaxis.set_major_locator(MultipleLocator(0.200))
        ax[0].yaxis.set_major_locator(MultipleLocator(0.200))
        ax[1].xaxis.set_major_locator(MultipleLocator(0.200))

        # format axes
        offset = 0
        ax[0].set_xlim([0-offset, 1+offset])
        ax[0].set_ylim([0-offset, 1+offset])
        ax[0].set_xlabel('Confidence')
        ax[0].set_ylabel('Accuracy')
        ax[1].set_xlim([0-offset, 1+offset])
        ax[1].set_xlabel('Confidence')
        ax[1].set_ylabel('Count')

        true_prob = [0.0 if prob is None else prob for prob in true_prob]
        pred_prob = [0.0 if prob is None else prob for prob in pred_prob]

        ax[0].bar(x=[(i+0.5)/bins for i in range(bins)], height=true_prob, 
                    width=[1/bins]*bins, edgecolor='black', linewidth=1.3, 
                    color='gray', label='Outputs')
        ax[0].bar(x=[(i+0.5)/bins for i in range(bins)], 
                    height=[pred_prob[i]-true_prob[i] for i in range(bins)], 
                    width=[1/bins]*bins, bottom=true_prob, color='red', 
                    edgecolor='firebrick', linewidth=1.3, hatch='/', alpha=0.3,
                    label='Gap')
        ax[0].bar(x=[(i+0.5)/bins for i in range(bins)], 
                    height=[pred_prob[i]-true_prob[i] for i in range(bins)], 
                    width=[1/bins]*bins, bottom=true_prob, edgecolor='firebrick', 
                    linewidth=1.3, fill=False)
        ax[0].plot([0, 1], [0, 1], ls='--', color='black', lw=0.75)
        ax[0].set_title(f'Expected Calibration Error (ECE): {ece:0.3f}')
        ax[0].legend(loc=2, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
        
        ax[1].bar(x=[(i+0.5)/bins for i in range(bins)], height=count, 
                    width=[1/bins]*bins, edgecolor='black', linewidth=1.3, 
                    color='gray')
        plt.savefig(cal_figure_path)
        plt.close()

    return results


def bootstrapper(
    y_true: list[int], 
    y_pred: list[float], 
    eval_func: Callable,
    iterations: int,
    stratified: bool,
    confidence_level: float,
    seed: int,
    show_progress: bool = True,
) -> dict[str, list]:
    """
    """
    # set seed
    random.seed(seed)

    # group per class
    classes = {}
    for i, label in enumerate(y_true):
        label = str(label)
        if label not in classes:
            classes[label] = [i]
        else:
            classes[label].append(i)

    # initialize iterator
    if show_progress:
        iterator = tqdm(range(iterations))
    else:
        iterator = range(iterations)

    # sample with replacement with or without stratification          
    sample_results = []
    for _ in iterator:
        # generate indices
        if stratified:
            sample_indices = []
            for values in classes.values():
                sample_indices.extend(
                    [values[random.randint(0, len(values)-1)] for _ in range(len(values))]
                )
        else:
            sample_indices = [random.randint(0, len(y_true)-1) for _ in range(len(y_true))]
        # select cases with replacement based on generated indices
        y_true_sampled = [y_true[i] for i in sample_indices]
        y_pred_sampled = [y_pred[i] for i in sample_indices]
        sample_results.append(eval_func(y_true_sampled, y_pred_sampled))

    if None in sample_results:
        sample_results.remove(None)
    
    # calculate mean and confidence intervals for bootstrap samples
    bootstrap_results = {
        'mean': np.mean(sample_results),
        f'{confidence_level}% CI lower': np.quantile(sample_results, (1-confidence_level)/2),
        f'{confidence_level}% CI upper': np.quantile(sample_results, 1-((1-confidence_level)/2)),
    }

    return bootstrap_results


def on_rm_error(func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod(path, stat.S_IWRITE )
    os.unlink(path )