import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

import torch
from torch.nn import functional as F

from torchvision import transforms as T
from torchvision.transforms import functional as F_tv
from torchvision.utils import draw_segmentation_masks

from batchflow.batchflow.opensets import PascalSegmentation


def plot_prediction(pipeline, image_number, dataset, alpha=0.6, title=""):
    """
    Plots image and predicted mask on the same figure. Also plots ground truth on the second figure.
    Colorbar is plotted on both figures for existed classes as well.

    Parameters
    ----------
    pipeline: Pipeline()
        pipeline should contain images, predictions and masks variables.

    image_number: int
        number of image in test dataset

    dataset: str
        should be either:
            'pascal' then plots image and mask according to pascal classes
            'ADE' then plot image and mask according to ADE classes

    alpha: float
        transparency factor (between 0 and 1)
    """
    label2name, label2color = _get_names_colors(dataset=dataset)

    prediction = pipeline.get_variable('predictions')[image_number].argmax(axis=1).squeeze()
    image =  np.moveaxis(pipeline.get_variable('images')[image_number], 1, -1).squeeze()
    mask = np.where(
        pipeline.get_variable('masks')[image_number] == 255,
        0,
        pipeline.get_variable('masks')[image_number]
    )
    mask = mask.squeeze()

    plot_segm(
        image=image,
        mask=prediction,
        alpha=alpha,
        label2name=label2name,
        label2color=label2color,
        title=title
    )
    plot_segm(
        image=image,
        mask=mask,
        alpha=alpha,
        label2name=label2name,
        label2color=label2color,
        title='GT'
    )


def plot_segm(image, mask, alpha, label2name, label2color, title=''):
    """
    Plots original image and segmentation mask on one plot

    Parameters
    ----------

    image: PIL.Image, np.ndarray
        original image

    mask: PIL.Image, torch.Tensor, np.ndarray
        segmentation mask

    alpha: float
        transparency factor (between 0 and 1)

    label2name: dict(int: str)
        dict of class indices and its names

    label2color: dict(int: srt)
        dict of class indices and its colors in hex code

    title: str
        title of the plot
    """
    segm_image, names, colors = _prepare_to_plot(image, mask, alpha, label2name, label2color)

    segm_image = F_tv.to_pil_image(segm_image)
    _ = plt.matshow(np.asarray(segm_image))

    cmap = ListedColormap(colors)
    bounds = np.array(range(len(colors)+1))-0.5 # without this, class names are shifted
    norm = BoundaryNorm(bounds, len(colors))
    cmappable = ScalarMappable(norm=norm, cmap=cmap)
    cax = plt.colorbar(
        cmappable,
        ticks=np.arange(0, len(names)),
        ax=plt.gca()
    )

    cax.ax.set_yticklabels(names, va='center')
    plt.title(title)

    plt.show()


def _prepare_to_plot(image, mask, alpha, label2name, label2color):
    """
    Prepares image and mask for plotting

    Parameters
    ----------
    image: PIL.Image, np.ndarray
        original image

    mask: PIL.Image, torch.Tensor, np.ndarray
        segmentation mask

    alpha: float
        transparency factor (between 0 and 1)

    label2name: dict(int: str)
        dict of class indices and its names

    label2color: dict(int: srt)
        dict of class indices and its colors in hex code

    Returns
    -------
    segmentation: torch.Tensor
        image with segmentation mask on it

    names: List[str]
        list of names for each class in segmentation mask

    colors: List[str]
        list of colors for each class in segmentation mask
    """
    if isinstance(mask, torch.Tensor):
        mask_shape = mask.size()
    else:
        mask_shape = np.array(mask).shape

    unique_classes, relabled_image = np.unique(np.array(mask), return_inverse=True)
    relabled_image = relabled_image.reshape(mask_shape)

    colors = [label2color[cls_number] for cls_number in unique_classes]
    names = [str(cls_number) + ' ' + label2name[cls_number] for cls_number in unique_classes]

    image = T.Resize(mask_shape, antialias=True)(
        torch.as_tensor(
            np.array(image).transpose(2,0,1),
        dtype=torch.uint8)
    )

    masks = F.one_hot(torch.from_numpy(relabled_image), num_classes=unique_classes.shape[0])
    masks = masks == 1

    segmentation = draw_segmentation_masks(
        image,
        masks=masks.permute(2,0,1),
        alpha=alpha,
        colors=colors
    )
    return segmentation, names, colors


def _get_names_colors(dataset):
    """
    Creates mapping for labels, names and colors

    Parameters
    ----------
    dataset: str
        should be either:
            'pascal' then plots image and mask according to pascal classes
            'ADE' then plot image and mask according to ADE classes

    Returns
    -------
    Tuple: (dict, dict)
        the first dict is label to name mapping, the second one is label to color mapping
    """
    if dataset == 'pascal':
        # TODO refactor this shit
        classes = PascalSegmentation.classes
        colors = [
            '#FFFFFF', '#B47878', '#06E6E6', '#503232',
            '#04C803', '#787850', '#8C8C8C', '#CC05FF',
            '#F6B6E6', '#04FA07', '#E005FF', '#EBFF07',
            '#96053D', '#787846', '#08FF33', '#FF0652',
            '#8FFF8C', '#CCFF04', '#FF3307', '#CC4603',
            '#0066C8'
        ]
        label2name = dict(zip(range(len(classes)), classes))
        label2color = dict(zip(range(len(colors)), colors))
    elif dataset in ['ADE', 'ade']:
        df = pd.read_csv(
            'data/ADEChallengeData2016/color_coding_semantic_segmentation_classes - Sheet1.csv'
        )
        label2name = dict(zip(df['Idx'], df['Name']))
        label2name[0] = 'background'
        label2color = dict(zip(df['Idx'], df['Color_Code(hex)']))
        label2color[0] = '#000000'
    else:
        raise ValueError('Wrong dataset name. Should be either "pascal" or "ADE" ')

    return (label2name, label2color)


def predict_image_mask(model, image, mask, device):
    """
    Predicts a segmentation mask and calculates the mIoU

    Parameters
    ----------

    model: nn.Module
        trained segmentation model

    image: PIL.Image
        original image

    mask: PIL.Image
        ground truth segmentation mask

    device: torch.device
        device to which image and model should be transfered

    Returns
    -------
    masked: torch.tensor
        predicted segmentation mask

    score: float
        mIoU score
    """
    model.eval()
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((512, 512), antialias=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
    )
    image = transform(image)
    model = model.to(device)
    image = image.to(device)

    mask = np.array(mask)[np.newaxis, :, :]
    mask.flags.writeable = True
    mask = torch.as_tensor(mask)
    mask = F_tv.resize(mask, (512, 512), interpolation=T.InterpolationMode.NEAREST)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked
