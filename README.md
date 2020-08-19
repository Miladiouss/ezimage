# ezimage

## What is ezimage?
Load and display images and access its content data with one-liners. A wrapper for `PIL` and `IPython.display`, ideal for machine learning and image processing. Easily load a PNG/JPEG image (or a list of them) from a local machine or the web and use its data in any desired format (e.g. NumPy/PyTorch/TensorFlow/etc). Display or save data arrays from any format or ordering (`CHW`/`WCH`). All with one-liners.

## Installation

```bash
pip install ezimage
```
## Example Usage

```Python
from ezimage import ezimage
img = ezimage('https://tinyurl.com/yyxvexs6')
img.CHW
img.display()
```
Watch this tutorial on [YouTube]() for a walk through.

## Description
Enables the user to easily load an image from a path, a url, or by directly providing a data array. If a list is provided, a list of ezimage instances will be retunred.
The properties enable the user to access the image data in R, G, B, A, GS (grayscale), HWC, and CHW formats.
In IPython environments such as in Jupyter, the `display` method allows the user to view images. Unlike PIL, this method allows for iterative image display.

Regardless of the ordering, format, or type of data, if directly feeding data arrays, all values must be in a valid PNG range (between 0 and 255).

## Versatile Data Type and Data Format
User is free to choose the format of the data arrays specified with `format_func`.

`format_func` is a function that converts a PIL image to user's desired format.
The output must cast to a NumPy array (i.e. no error when running `np.array(format_func(PIL_img))`). See examples below.

Note that the internal computations are done using PIL functions or NumPy arrays and `format_func` is only designated for the user interface.

NumPy uint8 Example:

```Python
format_func=lambda PIL_Image: np.array(PIL_Image, dtype=np.uint8)
```

NumPy float16 Example:

```Python
format_func=lambda PIL_Image: np.array(PIL_Image, dtype=np.float16)
```

PyTorch float32 Example:

```Python
import torch
format_func=lambda PIL_Image: torch.tensor(np.array(PIL_Image), dtype=torch.float32)
```

Example usage for a single image from the web:

```Python
img = ezimage("https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Omar_Khayyam2.JPG/220px-Omar_Khayyam2.JPG")
img.display()
```

## Example usage for reading all PNG files in a directory:

```Python
from ezimage import ezimage
from pathlib import Path
path_parent = Path("path/to/image/folder/")
pathList = list(path_parent.glob("*.png"))
imgList = [ezimage(p) for p in pathList]
for img in imgList:
    img.display(print_name=True)
```
