"""
Utility functions for preprocessing whole slide images (WSIs).
"""

import os
import stat
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from math import ceil, floor
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import UID
import torch
import wsidicom
from natsort import natsorted
from skimage.transform import rescale
from torch.nn.functional import conv2d

from pipeline.anonymization_utils import anonymize_non_dicom


# class and function for reformatting DICOM files
class Generator():

    def __init__(self, name: str) -> None:
        self.name = name
    
    def __call__(self) -> None:
        return UID(self.name)


def reformat_dicom(path: Union[str, Path]) -> None:
    """
    Reformat DICOM whole slide image by loading and saving it using wsidicom.

    Args:
        path:  Path to whole slide image.
    """
    # initialize generator and new path variable
    path = Path(path)
    generator = Generator(path.name.replace('.dcm', '.1'))
    new_path = path.parent / (generator()+'.dcm')

    # open slide and save it again to reformat the DICOM
    try:
        with wsidicom.WsiDicom.open(path) as file:
            file.save(output_path=path.parent, uid_generator=generator)
    except wsidicom.errors.WsiDicomFileError as error:
        os.remove(new_path)
        raise error
    else:
        # remove the read-only tag, delete the old slide,
        # and rename the reformatted DICOM file to the old name
        os.chmod(path, stat.S_IWRITE)
        os.remove(path)
        os.rename(new_path, path)


# functions for anonymization
def anonymize(path: Union[str, Path]) -> None:
    """
    Anonymize whole slide image (for DICOM, NDPI, SVS, or MRXS)

    Args:
        path:  Path to whole slide image.
    """
    # check if file exists
    path = Path(path)
    if path.exists():
        # check if WSI file has the DICOM type
        if path.suffix == '.dcm':
            # define path to anonymized image 
            path_anonymized = path.as_posix().replace('.dcm', 'temp.dcm')
            # load image
            with pydicom.dcmread(path) as dicom_file:
                # anonymize tags with patient info
                if 'AccessionNumber' in dicom_file:
                    dicom_file.AccessionNumber = ''
                if 'PatientName' in dicom_file:
                    dicom_file.PatientName = ''
                if 'PatientID' in dicom_file:
                    dicom_file.PatientID = ''
                if 'PatientBirthDate' in dicom_file:
                    dicom_file.PatientBirthDate = ''
                if 'PatientSex' in dicom_file:
                    dicom_file.PatientSex = ''
                if 'BarcodeValue' in dicom_file:
                    dicom_file.BarcodeValue = ''
                if 'ContainerIdentifier' in dicom_file:
                    dicom_file.ContainerIdentifier = ''
                # save a copy of the anonymized image
                dicom_file.save_as(path_anonymized)
            # remove the initial image and rename the anonymized image
            os.remove(path)
            os.rename(path_anonymized, path)
        else:
            anonymize_non_dicom(path.as_posix())


# functions for tessellation
def get_bounding_box(array: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Returns minimum and maximum row and column index for box around 
    non-zero elements.

    Args:
        array:  Array for which the bounding box enclosing all non-zero elements
            should be found.
    
    Returns:
        rmin:  Smallest row index with non-zero element.
        rmax:  Largest row index with non-zero element.
        cmin:  Smallest column index with non-zero element.
        cmax:  Largest column index with non-zero element.
    """
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    # check if there are any non-zero elements
    if sum(rows) == 0 or sum(cols) == 0:
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def tessellate(
    segmentation: Union[np.ndarray, torch.Tensor],
    shape: tuple[int, int],
    stride: Optional[tuple[int, int]] = None,
    min_tissue_fraction: float = 0.001,
    preprocessing_function: Optional[Callable] = None,
    pen_marking_map: Optional[Union[np.ndarray, torch.Tensor]] = None,
    max_pen_marking_fraction: float = 0.0,
    exceed_image: bool = False, 
) -> dict[int,list[tuple[tuple[int,int], tuple[int,int]]]]:
    """
    Extract tile coordinates and sizes based on the binary tissue segmentation.
    
    Args:
        segmentation:  Binary tissue segmentation as (height, width, channels).
        shape:  Shape of the tile in pixels as (height, width).
        stride:  Stride for convolution (None by default, which is equal to the shape)
        min_tissue_fraction:  Threshold value for minimum fraction of tissue in a tile.
        preprocessing_function:  Optional function that preprocesses segmentation map.
        pen_marking_map:  Binary image with pen marking regions as (height, width) 
            or as (height, width, 0 or 1).
        max_pen_marking_fraction:  Maximum fraction of area covered with pen marking.
        exceed_image:  Indicates whether tile coordinates can exceed the image dimensions.
    
    Returns:
        tile_information:  Dictionary with a list per channel of tuples with 
            the tile locations (top left corner) and shapes.
    """
    # convert torch.Tensor to numpy.ndarray
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.numpy()
    # check if the segmentation shape has three axes
    if len(segmentation.shape) != 3:
        raise ValueError(
            'Argument for `segmentation` has an invalid shape (expected 3 axes).',
        )
    # check if the shape is not larger than the segmentation
    if ((shape[0] > segmentation.shape[0] or shape[1] > segmentation.shape[1])
        and not exceed_image):
        warnings.warn(
            'Argument for `shape` exceeds the height and/or width of the segmentation. '
            'No tiles were extracted.',
            stacklevel=2,
        )
        return []

    # set the stride equal to the shape if it is None
    if stride is None:
        stride = shape
    
    # perform checks if an pen marking map was provided
    if pen_marking_map is not None:
        # convert torch.Tensor to numpy.ndarray
        if isinstance(pen_marking_map, torch.Tensor):
            pen_marking_map = pen_marking_map.numpy()
        # check if the pen marking map shape has two or three (last axis is 1) axes
        if len(pen_marking_map.shape) == 3 and pen_marking_map.shape[-1] == 1:
            pen_marking_map = pen_marking_map[..., 0]
        elif len(pen_marking_map.shape) != 2:
            raise ValueError(
                'Argument for `pen_marking_map` has an invalid shape '
                '(expected 2 axes or 3 axes with only one channel).'
            )
        # check if the pen marking map shape is equal to the segmentation shape
        if (pen_marking_map.shape[0] != segmentation.shape[0] 
            or pen_marking_map.shape[1] != segmentation.shape[1]):
            raise ValueError(
                'Argument for `pen_marking_map` does not match the spatial size '
                'of the argument for `segmentation`.'
            )
    # initialize dictionary to store tile location and shapes
    tile_information = {}

    # loop over the axes (i.e., cross-sections):
    for i in range(segmentation.shape[-1]):
        # get the cross-section and optionally apply preprocessing
        cross_section = segmentation[..., i]
        if preprocessing_function is not None:
            cross_section = preprocessing_function(cross_section)
        # get bounding box for cross-section
        bounding_box = get_bounding_box(cross_section)
        if bounding_box is not None:
            # get the bounding box coordinates
            top, bottom, left, right = bounding_box
            # calculate the height and width
            height = bottom-top
            width = right-left

            # calculate the added height and width to make it divisible
            added_height = (ceil(height / shape[0])*shape[0])-height
            added_width = (ceil(width / shape[1])*shape[1])-width
            # determine the correction by splitting the height and width to be added
            height_correction = (-floor(added_height/2), ceil(added_height/2))
            width_correction = (-floor(added_width/2), ceil(added_width/2))

            # determine how far outside the corrected height and width would be
            outside_top = max(0, -(top+height_correction[0]))
            outside_bottom = max(0, (bottom+height_correction[1])-segmentation.shape[0])
            outside_left = max(0, -(left+width_correction[0]))
            outside_right = max(0, (right+width_correction[1])-segmentation.shape[1])
            padding = ((0, 0), (0, 0), (outside_top, outside_bottom), 
                       (outside_left, outside_right))

            # address the corrections for which there is overlap at the top and/or bottom
            if not exceed_image:
                if segmentation.shape[0]-height-added_height >= 0:
                    height_correction = (
                        height_correction[0]+outside_top-outside_bottom,
                        height_correction[1]+outside_top-outside_bottom,
                    )
                else:
                    height_correction = (
                        floor((shape[0]-added_height)/2), 
                        -ceil((shape[0]-added_height)/2),
                    )
                outside_top = 0
                outside_bottom = 0

                # address the corrections for which there is overlap on the left and/or right
                if segmentation.shape[1]-width-added_width >= 0:
                    width_correction = (
                        width_correction[0]+outside_left-outside_right,
                        width_correction[1]+outside_left-outside_right,
                    )
                else:
                    width_correction = (
                        floor((shape[1]-added_width)/2), 
                        -ceil((shape[1]-added_width)/2),
                    )
                outside_left = 0
                outside_right = 0

            # determine the top left coordinate as (x, y), height, and width
            top_left = (left+width_correction[0], top+height_correction[0])
            height = height-height_correction[0]+height_correction[1]
            width = width-width_correction[0]+width_correction[1]
            # correct the region if tiles can exceed the image
            if exceed_image:
                top_left = (top_left[0]+outside_left, top_left[1]+outside_top)
                height -= (outside_top+outside_bottom)
                width -= (outside_left+outside_right)

            # crop the cross-section and prepare for convolution
            crop = cross_section[
                top_left[1]:top_left[1]+height,
                top_left[0]:top_left[0]+width,
            ][None, None, ...].astype(np.float32)
            # pad the cropped image region if tiles can exceed the image
            if exceed_image:
                crop = np.pad(crop, padding, mode='constant', constant_values=0)

            # define average filter
            filter = (torch.ones(shape)/np.prod(shape))[None, None, ...]
            # convolve crop with filter
            filtered_crop = conv2d(torch.from_numpy(crop), weight=filter, bias=None, 
                                   stride=stride, padding='valid')[0, 0, ...].numpy()
            # find the region to extract tiles from
            extraction_region = np.where(filtered_crop >= min_tissue_fraction, 1, 0)

            # find the image regions to be excluded from tile extraction 
            # if an exclusion map was provided
            if pen_marking_map is not None:
                # crop the cross-section and prepare for convolution
                pen_marking_map_crop = pen_marking_map[
                    top_left[1]:top_left[1]+height,
                    top_left[0]:top_left[0]+width,
                ][None, None, ...].astype(np.float32)
                # pad the cropped image region if tiles can exceed the image
                if exceed_image:
                    pen_marking_map_crop = np.pad(pen_marking_map_crop, padding, 
                                                mode='constant', constant_values=0)
                
                # convolve crop with filter
                filtered_pen_marking_crop = conv2d(torch.from_numpy(pen_marking_map_crop),
                                                 weight=filter, bias=None, stride=stride, 
                                                 padding='valid')[0, 0, ...].numpy()
            else:
                filtered_pen_marking_crop = np.zeros_like(extraction_region)

            # find the indices of the tiles that exceed the minimum faction of tissue
            indices = np.nonzero(extraction_region)
            # loop over indices to get all (top left) tile locations
            positions = []
            locations = []
            shapes = []
            pen_marking_fractions = []
            for x, y in zip(indices[1], indices[0]):
                # exclude tiles with pen markings
                pen_marking_fraction = float(filtered_pen_marking_crop[y, x])
                if pen_marking_fraction > max_pen_marking_fraction:
                    continue
                # store tile information
                positions.append((int(x), int(y)))
                locations.append((
                    int(top_left[0]-outside_left+x*stride[1]), 
                    int(top_left[1]-outside_top+y*stride[0]),
                ))
                shapes.append(shape)
                pen_marking_fractions.append(pen_marking_fraction)
            # add information about the location and shape of the tiles 
            # to the dictionary
            tile_information[str(i)] = [
                tile for tile in zip(positions, locations, shapes, pen_marking_fractions)
            ]
    
    return tile_information


# function for visualization
def visualize_tessellation(
    images: Union[np.ndarray, torch.Tensor, list[Union[np.ndarray, torch.Tensor]]], 
    tile_information: dict[int,list[tuple[tuple[int,int], tuple[int,int]]]],
    output_path: Union[str, Path],
    line_color: Union[str, tuple[int, int, int]] = (255, 0, 0),
    line_width: int = 2,
    downscale_factor: float = 1.0,
    axis: int = 0,
) -> None:
    """
    Visualize the tile outlines on top of one or more images using cv2.

    Args:
        images:  One or more images to display tiles on top off.
        tile_information:  Dictionary with a list per channel of tuples with 
            the tile locations (top left corner) and shapes.
        output_path:  Path to save figure if not None.
        line_color:  Color of lines that delineate tiles. If equal to 'random', 
            assign a random color to each cross-section.
        line_width:  Width of lines that delineate tiles.
        downscale_factor:  Factor with which the images are downscaled before
            plotting the tiles on top.
        axis:  Axis along which the images (if more than one were given)
            are concatenated.
    """
    # put image in a list if it is not already
    if isinstance(images, (np.ndarray, torch.Tensor)):
        images = [images]
    # convert images from torch.Tensor to np.ndarray if necessary
    if isinstance(images, (list, tuple)):
        checked_images = []
        for image in images:
            if isinstance(image, torch.Tensor):
                checked_images.append(image.numpy())
            elif isinstance(image, np.ndarray):
                checked_images.append(image)
            else:
                raise TypeError('Invalid type of input argument for `images`.')
        images = checked_images
    else:
        raise TypeError('Invalid type of input argument for `images`.')
    # check if there is atleast one image and if all images match in size
    N_images = len(images)
    dimensions = images[0].shape[0:2]
    if N_images == 0:
        raise ValueError('Atleast one image should be provided as input.')
    if len({image.shape for image in images}) > 1:
        raise ValueError('Not all images have the same dimensions.')
    
    # make sure the downscale factor is always < 1.0.
    if downscale_factor > 1.0:
        downscale_factor = 1.0 / downscale_factor
    # rescale the images and convert to cv2 images
    converted_images = []
    for image in images:
        converted_images.append(
            (rescale(image, downscale_factor, channel_axis=-1)*255).astype(np.uint8),
        )
    images = converted_images

    # concatenate the images
    if N_images > 1:
        image = np.concatenate(images, axis=axis)
    else:
        image = images[0]

    # enable sampling diverse color
    if line_color in list(matplotlib.colormaps):
        sample_line_color = True 
        line_colors = matplotlib.colormaps[line_color]
    else:
        sample_line_color = False
  
    # plot tile outlines on top of the image
    if len(tile_information):
        for i, cross_section in enumerate(tile_information.values()):
            # sample random colors if enabled
            if sample_line_color:
                line_color = tuple([int(x*255) for x in line_colors(i/len(tile_information))])
            # plot tiles
            for n in range(N_images):
                for tile_info in cross_section:
                    _, (x, y), (tile_height, tile_width), _ = tile_info
                    top_left = (int((x + n*dimensions[axis]*(axis==1))*downscale_factor), 
                                int((y + n*dimensions[axis]*(axis==0))*downscale_factor))
                    bottom_right = (top_left[0] + int(tile_width*downscale_factor), 
                                    top_left[1] + int(tile_height*downscale_factor))
                    cv2.rectangle(image, top_left, bottom_right, line_color, line_width)
                # add cross-section number
                if len(cross_section):
                    coord = (top_left[0], bottom_right[1])
                    cv2.putText(image, str(i), coord, cv2.FONT_HERSHEY_DUPLEX, 3, line_color, 2, cv2.FILLED)
    
    # Save the image with squares drawn
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def combine(
    tissue_segmentation: np.ndarray, 
    pen_marking_segmentation: np.ndarray, 
    tissue_color = (1, 1, 1),
    pen_color = (0, 0, 1),
    background_color = (0, 0, 0),
) -> np.ndarray:
    """
    Combine tissue and pen marking segmentation results.

    Args:
        tissue_segmentation:  Segmentation of tissue regions as (height, width)
            or (height, width, channel).
        pen_marking_segmentation:  Segmentation of pen marking regions as (height,
            width) or (height, width, channel).
        tissue_color:  Color assigned to tissue regions.
        pen_color:  Color assigned to pen marking regions.
    
    Returns:
        combined_segmentations:  Combined tissue and pen marking segmentation as 
            (height, width, channel).
    """
    # sum over the channel dimension if available
    if len(tissue_segmentation.shape) == 3:
        tissue_segmentation = np.sum(tissue_segmentation, axis=2)
    if len(pen_marking_segmentation.shape) == 3:
        pen_marking_segmentation = np.sum(pen_marking_segmentation, axis=2)
    
    # check if the segmentation images match in size
    if tissue_segmentation.shape != pen_marking_segmentation.shape:
        raise ValueError('The tissue and pen marking segmentation images '
                         'differ in size.')
    channels = []
    colors = [tissue_color, pen_color, background_color]
    for tissue_value, pen_value, background_value in zip(*colors):
        channel = np.ones_like(tissue_segmentation, dtype=float)*background_value
        channel = np.where(tissue_segmentation > 0.5, tissue_value, channel)
        channel = np.where(pen_marking_segmentation > 0.5, pen_value, channel)
        channels.append(channel[..., None])

    # combine channels
    combined_segmentations = np.concatenate(channels, axis=2)

    return combined_segmentations

# slow implementation using matplotlib
def visualize_tessellation_plt(
    images: Union[np.ndarray, torch.Tensor, list[Union[np.ndarray, torch.Tensor]]], 
    tile_information: dict[int,list[tuple[tuple[int,int], tuple[int,int]]]],
    show_plot: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    cmap: str = 'gray',
    line_color: str = 'red',
    line_width: Union[int, float] = 1.0,
    fill_tile: bool = False,
    fill_color: Optional[str] = None,
    fill_alpha: Union[int, float] = 0.5,
    dpi: int = 72,
) -> None:
    """
    Visualize the tile outlines on top of one or more images using plt.imshow().

    Args:
        images:  One or more images to display tiles on top off.
        tile_information:  Dictionary with a list per channel of tuples with 
            the tile locations (top left corner) and shapes.
        show_plot:  Indicates whether the plot should be displayed using plt.show().
        output_path:  Path to save figure if not None.
        cmap:  Colormap for displaying 1-channel images.
        line_color:  Color of lines that delineate tiles. If equal to 'random', 
            assign a random color to each cross-section.
        line_width:  Width of lines that delineate tiles.
        fill_tile:  Indicates whether the tiles should be filled with the same color
        fill_color:  Color of tile area. If equal to 'diverse', assign a random 
            color to each cross-section.
        fill_alpha:  Opacity of tile area.
        dpi:  Specifies resolution if figure when saved as bitmap image.
    """
    # put image in a list if it is not already
    if isinstance(images, (np.ndarray, torch.Tensor)):
        images = [images]
    # convert images from torch.Tensor to np.ndarray if necessary
    if isinstance(images, (list, tuple)):
        checked_images = []
        for image in images:
            if isinstance(image, torch.Tensor):
                checked_images.append(image.numpy())
            elif isinstance(image, np.ndarray):
                checked_images.append(image)
            else:
                raise TypeError('Invalid type of input argument for `images`.')
        images = checked_images
    else:
        raise TypeError('Invalid type of input argument for `images`.')
    
    # set fill_color to line_color if no color was specified
    if fill_color is None:
        fill_color = line_color

    # enable sampling diverse color
    if line_color in list(matplotlib.colormaps):
        sample_line_color = True 
        line_colors = matplotlib.colormaps[line_color]
    else:
        sample_line_color = False
    
    if fill_color in list(matplotlib.colormaps):
        sample_fill_color = True 
        fill_colors = matplotlib.colormaps[fill_color]
    else:
        sample_fill_color = False
    
    fig, ax = plt.subplots(len(images), 1) 
    if len(images) == 1:
        ax = [ax]   
    for i, image in enumerate(images):

        # plot tile outlines on top of the image
        for j, cross_section in enumerate(tile_information.values()):
            # sample random colors if enabled
            if sample_line_color:
                line_color = line_colors(j/len(tile_information))
            if sample_fill_color:
                fill_color = fill_colors(j/len(tile_information))
            # plot tiles
            for tile_info in cross_section:
                _, (x, y), (height, width) = tile_info
                x = x-0.5
                y = y-0.5
                # plot tile filling
                if fill_tile:
                    # plot tile outlines
                    tile = plt.Rectangle((x, y), height, width, fc=fill_color, 
                                        alpha=fill_alpha) 
                    ax[i].gca().add_patch(tile)
                # plot tile lines
                ax[i].plot([x,x], [y, y+height], color=line_color, linewidth=line_width)
                ax[i].plot([x+width,x+width], [y, y+height], color=line_color, 
                           linewidth=line_width)
                ax[i].plot([x,x+width], [y, y], color=line_color, linewidth=line_width)
                ax[i].plot([x,x+width], [y+height, y+height], color=line_color, 
                           linewidth=line_width)
        # show image 
        ax[i].axis('off')
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1):   
            ax[i].imshow(image, cmap=cmap) 
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            ax[i].imshow(image)  
        else:
            raise ValueError('Invalid number of channels for image.')
    
    # save figure if a path is specified
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    if show_plot:
        plt.show()
    else:
        plt.close()


# other helper functions
def get_unique_filenames(filenames: list[str]) -> list[str]:
    """ 
    Remove duplicate filenames from list. 
    
    Args:
        filenames:  List with filenames.

    Returns:
        unique_filenames:  List with unique filenames.
    """
    unique_filenames = []
    for filename in filenames:
        if '.dcm' in filename:
            unique_filenames.append(filename.split('.')[0])
        else:
            unique_filenames.append('.'.join(filename.split('.')[:-1]))
    return natsorted(list(set(unique_filenames)))


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Replace integers by smaller type in dataframe if possible.
    
    Args:
        df:  DataFrame.

    Returns:
        df:  Optimized DataFrame.
    """
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df