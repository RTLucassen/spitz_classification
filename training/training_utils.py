"""
Utility functions for model training.
"""

import json
import random
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        feature_information: dict[int, dict[str, Any]],
        selection: dict[int, list],
        task: str,
        patient_vector_func: Optional[Callable],
        label_func: Callable,
        weight_func: Optional[Callable] = None,
        length: Optional[int] = None,
        repetitions: int = 1,
        section_dropout_prob: float = 0.0,
        feature_dropout_prob: float = 0.0,
        maximum_features: Optional[int] = None,
        only_repeat_cases_exceeding_maximum: bool = False,
    ) -> None:
        """
        Initializes dataset instance to load feature vectors.

        Args:
            feature_information:  Dictionary with feature information.
            selection:  Dictionary with selected features per case.
            task:  Name of task for returning the correct labels.
            patient_vector_func:  Function to create the vector with the encoded
                patient information.
            label_func:  Function to create label based on specimen information.
            weight_func:  Function to calculate weight based on specimen information.
            length:  Number of items in the dataset (can be set arbitrarily for
                training without specifying epochs) (overrules repetitions).
            section_dropout_prob:  Probability for randomly dropping out a cross-section.
            feature_dropout_prob:  Probability for randomly dropping out a feature.
            maximum_features:  Maximum number of features. If more features are available, 
                randomly select a subset equal to the maximum.
            repetitions:  Number of times each case is sampled if no fixed length is used.
            only_repeat_cases_exceeding_maximum:  Indicates whether the repetitions
                only apply for cases with more features than the maximum.
        """
        # define instance attributes
        self.feature_information = feature_information
        self.selection = selection
        self.task = task
        self.patient_vector_func = patient_vector_func
        self.label_func = label_func
        self.weight_func = weight_func
        self.length = length
        self.section_dropout_prob = section_dropout_prob
        self.feature_dropout_prob = feature_dropout_prob
        self.maximum_features = maximum_features
        self.repetitions = repetitions
        self.only_repeat_cases_exceeding_maximum = only_repeat_cases_exceeding_maximum

        # get list with specimen indices
        if self.length is not None:
            self.indices = list(self.feature_information.keys())
        elif only_repeat_cases_exceeding_maximum:
            # determine for which cases the total number of features 
            # in the selection exceeds the maximum
            self.indices = []
            for specimen_index in self.feature_information.keys():
                N_features = len(self.selection[specimen_index])
                if N_features > self.maximum_features:
                    self.indices.extend([specimen_index]*self.repetitions)
                else:
                    self.indices.append(specimen_index)
        else:
            self.indices = list(self.feature_information.keys())*self.repetitions

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        else:
            return len(self.indices)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
        """
        Load a tile and prepare it.

        Args:
            index:  Index for selecting tile from the dataset.

        Returns:
            features:  Concatenated feature vectors.
            patient_vector:  Vector with patient information encoded.
            positions:  Contatenated position vectors corresponding 
                to the feature vectors.
            label:  Specimen label.
        """
        # get the specimen index
        if self.length is not None:
            specimen_index = self.indices[index % len(self.indices)]
        else:
            specimen_index = self.indices[index]
        # get specimen information
        specimen_information = self.feature_information[specimen_index]
        # get specimen feature selection
        specimen_feature_selection = self.selection[specimen_index]   
        # load the dictionary with feature vectors
        feature_dict = torch.load(specimen_information['feature_paths'][0])  
        feature_dict = feature_dict[list(feature_dict.keys())[0]]
        keys = specimen_feature_selection

        # randomly drop out features
        if self.feature_dropout_prob > 0.0:
            # get the indices for the features of the selected features
            selection_keys = [key for key in keys if random.random() > self.feature_dropout_prob]
            # include a random cross-section if none were selected
            if not len(selection_keys):
                selection_keys = [keys[random.randint(0, len(keys)-1)]]
            keys = selection_keys

        # randomly drop out cross-sections
        if self.section_dropout_prob > 0.0:
            # select the cross-sections to drop out
            sections_dict = {
                i: random.random() > self.section_dropout_prob for i in set([key[:-1] for key in keys])
            }
            # set a random cross-section to True if all are False (i.e., dropped out)
            if True not in set(sections_dict.values()):
                random_section = list(sections_dict.keys())[random.randint(0, len(sections_dict)-1)]
                sections_dict[random_section] = True

            # get the indices for the features of the selected cross-sections
            keys = [key for key in keys if sections_dict[key[:-1]]]     

        # randomly select features up to the maximum number
        if self.maximum_features is not None and len(keys) > self.maximum_features:
            random.shuffle(keys)
            keys = sorted(keys[:self.maximum_features])

        features = []
        positions = []
        # select the features and positions
        for key in keys:
            features.append(torch.tensor(feature_dict[key]['feature'], dtype=torch.float32))
            positions.append(torch.tensor(feature_dict[key]['position'], dtype=torch.float32))

        # concatenate feature vectors
        features = torch.concat(tensors=features)
        positions = torch.concat(tensors=positions)

        # create a patient vector and define variables for the weight and label
        label = self.label_func(specimen_information, self.task)
        if self.patient_vector_func is None:
            patient_vector = torch.nan
        else:
            patient_vector = self.patient_vector_func(specimen_information)
        
        if self.weight_func is None:
            weight = 1
        else:
            weight = self.weight_func(specimen_information)

        return features, patient_vector, positions, weight, label, specimen_index


# helper function for reading feature information and creating patient vector
def read_feature_information(
    feature_information_path: Union[str, Path], 
    feature_directory: Optional[Union[str, Path]] = None, 
) -> dict[int, dict[str, Any]]:
    """
    Reads feature information

    Args:
        feature_information_path:  Path to feature with tile information.
        image_directory:  Path to folder where all images are stored.

    Returns:
        data_dict:  Dictionary with specimen information, feature paths,
            and corresponding tile position information.
    """
    # read feature information from file
    with open(feature_information_path, 'r') as f:
        lines = f.readlines()

    # check if the number of lines is a multiple of three
    if len(lines) % 3 != 0:
        raise ValueError(f'The number of lines in {feature_information_path} '
                         'must be a multiple of three.')

    # initialize dictionary for storing the feature information 
    data_dict = {}
    # loop over feature information per specimen
    for i in range(int(len(lines)/3)):
        image_filenames = eval(lines[i*3])
        specimen_information = json.loads(lines[i*3+1])
        feature_filenames = eval(lines[i*3+2])
        if isinstance(feature_filenames, str):
            feature_filenames = [feature_filenames] 
        if feature_directory is not None:
            feature_paths = [(Path(feature_directory)/name) for name in feature_filenames]
        else:
            feature_paths = feature_filenames

        # add information to dictionary and indices to list
        specimen_index = specimen_information['specimen_index']
        if specimen_index in data_dict:
            data_dict[specimen_index]['feature_paths'].extend(feature_paths)
        else:
            data_dict[specimen_index] = {
                **specimen_information,
                'images': image_filenames,
                'feature_paths': feature_paths,
            }

    return data_dict


def get_patient_vector(
    specimen_information: dict[str, Any], 
    mul_factor: Union[int, float] = 1,
) -> torch.Tensor:
    """
    Get vector with patient information encoded.

    Args:
        specimen_information:  Dictionary with specimen information.

    Returns:
        patient_vector:  Vector with patient information encoded.
    """
    # check if the required keys are available in the specimen information
    for key in ['sex', 'age', 'location_group']:
        if key not in specimen_information:
            raise ValueError(f'Patient information not specified for specimen: {key}.')
    
    patient_vector = []
    # add sex information
    if specimen_information['sex'].lower() in ['m', 'male']:
        patient_vector.append(-1.0)
    else:
        patient_vector.append(1.0)
    
    # add age information
    patient_vector.append((specimen_information['age']/50)-1)

    # add location information
    location = str(specimen_information['location_group']).lower()
    
    # HACK: correct for one case with a missing location group during feature extraction
    if location == 'nan': location = 'trunk'
    
    groups = [
        ['trunk', 'buttock'], 
        ['head', 'neck'], 
        ['foot', 'footsole', 'hand', 'handpalm'], 
        ['lower extremity'], 
        ['upper extremity'],
    ]
    for group in groups:
        if location in group:
            patient_vector.append(1.0)
        else:
            patient_vector.append(-1.0)
    
    # convert list to vector
    patient_vector = torch.tensor(patient_vector)*mul_factor

    return patient_vector


def get_label(specimen_information: dict[str, Any], task: str) -> torch.Tensor:
    """
    Get specimen label.

    Args:
        specimen_information:  Dictionary with specimen information.
        task:  Name of prediction task.

    Returns:
        label: Specimen label.
    """
    task = task.lower()
    if task == 'spitz vs conventional melanoma':
        label = F.one_hot(
            torch.tensor(specimen_information['label_spitz']), 
            num_classes=2,
        )
    elif task == 'spitz subtype':
        label = F.one_hot(
            torch.tensor(specimen_information['label_spitz_signature']), 
            num_classes=4,
        )
    elif task == 'spitz dignity':
        mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
        label = F.one_hot(
            torch.tensor(mapping[specimen_information['label_spitz_dignity']]), 
            num_classes=3,
        )
    else:
        raise ValueError(f'Invalid task: {task}.')
    
    return label         


def select_tiles(
    tile_division: dict[str, Any], 
    origin: str, 
    rank: Union[list, str],
) -> dict[int, list]:
    """
    Select tiles based on origin and rank configuration

    Args:
        tile_division:  Dictionary with origin, rank, and tile indices per image name.
        origin:  Selection for origin of slide (i.e., internal, consultation, or all).
        rank:  Selection for slide representativity rank, either a list with ranks or "top X".

    Returns:
        tile_selection:  Dictionary with per case index all selected tile indices.
    """
    # determine how to select cases based on the rank
    top = None
    if isinstance(rank, str):
        if rank[:3] == 'top':
            top = int(rank.split(' ')[1])

    # loop over the cases and select the tiles
    tile_selection = {}
    for case_index, case in tile_division.items():
        selected_tile_indices = []
        # select based on top N ranked tiles
        if top is not None:
            included = []
            for name, slide_info in case.items():
                if slide_info['origin'] not in origin:
                    continue
                else:
                    corrected_rank = 999 if slide_info['rank'] is None else slide_info['rank']
                    included.append((corrected_rank, name))

            selected_names = [item[1] for item in sorted(included)[:min(top, len(case))]]
            for name in selected_names:
                selected_tile_indices.extend([tuple(tile) for tile in case[name]['tiles']])
        # select based on rank number
        else:
            for name, slide_info in case.items():
                if slide_info['origin'] not in origin:
                    continue
                if slide_info['rank'] not in rank:
                    continue
                selected_tile_indices.extend([tuple(tile) for tile in slide_info['tiles']])

        tile_selection[int(case_index)] = selected_tile_indices

    return tile_selection    


class FocalLoss(nn.Module):

    def __init__(
        self,
        sigmoid: bool = False, 
        gamma: float = 0.0,
        class_weights: Optional[Union[list[float], torch.Tensor]] = None,
    ) -> None:
        """
        Initialize focal loss.

        Args:
            sigmoid:  Specify if a sigmoid instead of a softmax function is applied.
                If there is only a single class, the sigmoid is automatically used.
            gamma:  Parameter that governs the relative importance of incorrect 
                predictions. If gamma equals 0.0, the focal loss is equal to the 
                cross-entropy loss.
            class_weights:  If not None, compute a weighted average of the loss 
                for the classes. 
        """
        super().__init__()
        self.sigmoid = sigmoid
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor, 
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
            weight:  Weighting factor for each item in the batch. 

        Returns:
            loss:  Focal loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # convert class weights from list to tensor if necessary
        if isinstance(self.class_weights, list):
            self.class_weights = torch.Tensor(self.class_weights)
        # check if the number of classes matches the number of class weights if provided
        if self.class_weights is not None:
            if len(self.class_weights) != logit.shape[1]:
                raise ValueError('The number of class weights and classes do not match.')
        else:
            self.class_weights = [1]*logit.shape[1]

        # get the predicted probabilities by taking
        # the sigmoid or softmax of the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
            log_y_pred = F.logsigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)
            log_y_pred = F.log_softmax(logit, dim=1)

        # flatten the data (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        log_y_pred_flat = log_y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the pixelwise cross-entropy, focal weight, and pixelwise focal loss
        pixelwise_CE = -(log_y_pred_flat * y_true_flat)
        focal_weight = (1-(y_true_flat * y_pred_flat))**self.gamma
        pixelwise_focal_loss = focal_weight * pixelwise_CE

        # calculate the class-separated focal loss
        class_separated_focal_loss = torch.mean(pixelwise_focal_loss, dim=-1)
        
        # multiply the loss per class by the class weight
        for i, class_weight in enumerate(self.class_weights):
            class_separated_focal_loss[:, i] *= class_weight
        instance_loss = torch.sum(class_separated_focal_loss, dim=1)

        # if a weight was specified, multiple each item in the batch with
        if weight is not None:
            instance_loss *= weight

        # compute the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss