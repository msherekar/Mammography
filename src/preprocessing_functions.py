import matplotlib.pyplot as plt
import numpy as np
from skimage import io, data, img_as_float, exposure, morphology, measure
import os

# matplotlib.rcParams['font.size'] = 8

def plot_intensity_histogram(image_path, directory, image_name, bins=256, color='blue', alpha=0.7):
    """
    Plots the intensity histogram of a grayscale image.

    Parameters:
        image_path (str): Path to the image file.
        bins (int): Number of bins for the histogram (default is 256).
        color (str): Color of the histogram bars (default is 'blue').
        alpha (float): Transparency level of the bars (default is 0.7).
    """
    # Load the image as grayscale
    image = io.imread(image_path, as_gray=True)

    # Plot the intensity histogram
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.hist(image.ravel(), bins=bins, color=color, alpha=alpha)
    plt.title(f'Pixel Intensities for {image_name}', fontsize=14, fontweight='bold')
    plt.ylim(0, 5000)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(directory, f'Pixel_Intensity_{image_name}.png'))
    
def getLargestCC(im_in):
    """
    This function extracts the largest connected component 
    from a binary image, which is assumed to be the 
    main region of interest (aread on the image covered by breast).

    """

    # CC= connected compartment
    labels = measure.label(im_in)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
def resample_image(image_array, original_spacing, target_spacing):

    """
    Resamples the image to match the desired target spacing, 
    useful in medical imaging for ensuring that the pixel dimensions are consistent.
    
    """

    resize_factor = original_spacing / target_spacing
    new_size = (np.round(image_array.shape[1] * resize_factor), np.round(image_array.shape[0] * resize_factor))
    new_size = tuple(map(int, new_size))
 
    resampled_image = Image.fromarray(image_array).resize(new_size, Image.BICUBIC) #  a smooth resizing technique that preserves image quality.
    return np.array(resampled_image)

def remove_intensity_outliers(im_in):

    """
    Removes intensity outliers by capping the pixel values at specific percentiles.
    Args:
        im_in: Input image as a numpy array.
    Returns:
        outlierless_image: Image with intensity outliers removed.
    """
    PERCENTILES = [0, 0.05, 0.5, 2.5, 5, 50, 95, 97.5, 99.5, 99.95, 100]
    # Calculate percentiles
    intensity_percentiles = np.percentile(im_in, PERCENTILES)
    
    # Cap high intensities at the 99.5th percentile
    high_sat = np.where(im_in >= intensity_percentiles[-2], intensity_percentiles[-2], im_in)
    
    # Cap low intensities at the 0.05th percentile
    outlierless_image = np.where(high_sat <= intensity_percentiles[1], intensity_percentiles[1], high_sat)
    
    return outlierless_image

def normalize_image(im_in):
    """
    Normalizes the image to the range [0, 255].
    Args:
        im_in: Input image as a numpy array.
    Returns:
        normalized_image: Image normalized to the range [0, 255].
    """
    # Normalize to the range [0, 255]
    normalized_image = ((im_in - np.min(im_in)) / (np.max(im_in) - np.min(im_in))) * 255
    return normalized_image

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf