from typing import Tuple
import numpy as np

def asStride(arr, sub_shape):
    '''Get a strided sub-matrices view of an ndarray.
    based on: https://numbersmithy.com/2d-and-3d-pooling-using-numpy/
    Args:
        arr (ndarray): input array of rank 4, with shape (m, hi, wi, ci).
        sub_shape (tuple): window size: (f1, f2).
    Returns:
        subs (view): strided window view.
    This is used to facilitate a vectorized 3d convolution.
    The input array <arr> has shape (m, hi, wi, ci), and is transformed
    to a strided view with shape (m, ho, wo, f, f, ci). where:
        m: number of records.
        hi, wi: height and width of input image.
        ci: channels of input image.
        f: kernel size.
    The convolution kernel has shape (f, f, ci, co).
    Then the vectorized 3d convolution can be achieved using either an einsum()
    or a tensordot():
        conv = np.einsum('myxfgc,fgcz->myxz', arr_view, kernel)
        conv = np.tensordot(arr_view, kernel, axes=([3, 4, 5], [0, 1, 2]))
    See also skimage.util.shape.view_as_windows()
    '''
    sm, sh, sw, sc = arr.strides
    m, hi, wi, ci = arr.shape
    f1, f2 = sub_shape
    view_shape = (m, 1+(hi-f1), 1+(wi-f2), f1, f2, ci)
    strides = (sm, sh, sw, sh, sw, sc)
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs

def poolingOverlap(mat, shape, method='max'):
    '''Overlapping pooling on 4D data.
    based on: https://numbersmithy.com/2d-and-3d-pooling-using-numpy/
    Args:
        mat (ndarray): input array to do pooling on the mid 2 dimensions.
            Shape of the array is (m, hi, wi, ci). Where m: number of records.
            hi, wi: height and width of input image. ci: channels of input image.
        shape (tuple): New height and width.
    Keyword Args:
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
    Returns:
        result (ndarray): pooled array.
    See also unpooling().
    '''
    m, hi, wi, ci = mat.shape
    f = (hi - shape[0] + 1, wi - shape[1] + 1)

    view = asStride(mat, f)
    if method == 'max':
        result = np.nanmax(view, axis=(3, 4))
    else:
        result = np.nanmean(view, axis=(3, 4))
    return result

class VideoProcessor(object):
    shape: Tuple[int, int]
    pooling: str
    time_window: int
    batch_size: int
    padding: str

    def __init__(self, shape = (320, 320), pooling = 'max', time_window = 10, batch_size = 128, padding = 'zero'):
        """
        Object used to adjust shape, create time windows and batch data. 
        In the future it may be extended with different masks/filters.

        Args:
            shape (tuple, optional): Width and height that video should be adjusted to. Defaults to (320, 320).
            pooling (string, optional): Method used while resizing the video. Options: 'max'; 'mean'. Defaults to 'max'.
            time_window (int, optional): Nuber of frames per example. Defaults to 10.
            batch_size (int, optional): Number of examples/time windows per batch of data. Defaults to 128.
            padding (str, optional): Version of padding that should be used on time dimension. Options: 'zero'; 'none'. Defaults to 'zero'.
        """
        self.shape = shape
        self.pooling = pooling
        self.time_window = time_window
        self.batch_size = batch_size
        self.padding = padding
    
    def process(self, vid: np.ndarray) -> np.ndarray:
        resized_vid = poolingOverlap(vid, self.shape, method = self.pooling)
        return resized_vid




