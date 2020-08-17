import numpy as np
from PIL.Image import open as PIL_open, fromarray
from urllib.request import urlopen
from IPython.display import display

class ezImage():
    """
    A PIL wrapper that makes image processing and machine learning easier and more fun.
    Main advantages:
    If using an IPython environment (e.g. Jupyter Notebook, Jupyter Lab, Hydrogen in Atom, Python extension in VS Code), you can display images in for loops, something that is not possible with PIL. Instead of plt.imshow, the IPython display function is used for fast and light plotting.
    Eesily access channel data as numpy a numpy, or your choice of format (such as a PyTorch or a TensorFlow tensor) with your choice of dtype.
    Get image data using HWC or CHW properties.
    """
    def __init__(self, path=None, url=None, data_HWC=None, data_CHW=None, format_func=np.array):
        """
        Enables the user to easily load an image from a path, a url, or by directly providing a data array.
        The properties enable the user to access the image data in R, G, B, A, GS (grayscale), HWC, and CHW formats.
        In IPython environments such as in Jupyter, the `display` method allows the user to view images. Unlike PIL, this method allows for iterative image display.

        Regardless of the ordering, format, or type of data, if directly feeding data arrays, all values must be in a valid PNG range (between 0 and 255).

        ## Versatile Data Type and Data Format
        User is free to choose the format of the data arrays specified with `format_func`.
        `format_func`:
            A function that converts a PIL image to user's desired format.
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
            img = ezImage(url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Omar_Khayyam2.JPG/220px-Omar_Khayyam2.JPG")
            img.display()
            ```

        Example usage for reading all PNG files in a directory:
            ```Python
            from pathlib import Path
            path_parent = Path("/home/miladiouss/Pictures/Sample/")
            pathList = list(path_parent.glob("*.png"))
            imgList = [ezImage(p) for p in pathList]
            for img in imgList:
                img.display(print_name=True)
            ```
        """
        if path is not None:
            self.path = path
            self.image = PIL_open(path)
        if url is not None:
            self.path = url
            self.image = PIL_open(urlopen(url))
        if data_HWC is not None:
            self.path = f"{np.shape(data_HWC)} data array"
            # self.image = fromarray(cm.gist_earth(np.array(data_HWC), bytes=True))
            self.image = fromarray( np.array(data_HWC, dtype=np.uint8) )
        if data_CHW is not None:
            self.path = f"{np.shape(data_CHW)} data array"
            self.image = fromarray( self.CHW_to_HWC( np.array(data_CHW, dtype=np.uint8) ) )

        if len(np.shape(self.image)) == 3:
            self.H, self.W, self.C = np.shape(self.image)
        elif len(np.shape(self.image)) == 2:
            self.H, self.W = np.shape(self.image)
        self.format_func = format_func

    @property
    def R(self):
        """Returns Red channel data (if present)."""
        return self.format_func(self.image.getchannel('R'))

    @property
    def G(self):
        """Returns Green channel data (if present)."""
        return self.format_func(self.image.getchannel('G'))

    @property
    def B(self):
        """Returns Blue channel data (if present)."""
        return self.format_func(self.image.getchannel('B'))

    @property
    def A(self, force=False):
        """Returns alpha channel data (if present)."""
        return self.format_func(self.image.getchannel('A'))

    @property
    def grayscale(self):
        """Returns grayscale PIL image"""
        return self.image.convert('LA')

    @property
    def GS(self):
        """Returns grayscale image data"""
        return self.format_func(self.grayscale.getchannel(0))

    @property
    def CHW(self):
        """
        Returns image data in Channel, Height, Width format.
        Alpha channel will NOT be included.
        """
        return self.format_func(np.array([np.array(self.R), np.array(self.G), np.array(self.B)]))

    @property
    def CHWA(self):
        """
        Returns image data in Channel, Height, Width format.
        Alpha channel will be included.
        """
        return self.format_func(np.array([np.array(self.R), np.array(self.G), np.array(self.B), np.array(self.A)]))

    @property
    def HWC(self):
        """
        Returns image data in Height, Width, Channel format.
        Alpha channel will NOT be included.
        """
        return self.format_func(self.CHW_to_HWC(self.CHW))

    @property
    def HWCA(self):
        """
        Returns image data in Height, Width, Channel format.
        Alpha channel will be included.
        """
        return self.format_func(np.array(self.CHWA).transpose(1, 2, 0))

    def HWC_to_CHW(self, array):
        return np.array(array).transpose((2, 0, 1))

    def CHW_to_HWC(self, array):
        return np.array(array).transpose((1, 2, 0))

    def display(self, print_name=False, prepend_str="\n", break_len=79, break_str='-'):
        if print_name:
            header = "".join(break_len * [break_str])
            print(f"{prepend_str}{header}\n{self.path}")
        return display(self.image)

    def __repr__(self):
        return str(self.path)


def HWC_to_CHW(array):
    return np.array(array).transpose((2, 0, 1))

def CHW_to_HWC(array):
    return np.array(array).transpose((1, 2, 0))
