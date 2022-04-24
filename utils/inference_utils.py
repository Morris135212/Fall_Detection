import io
import cv2
from IPython.core.display import display
from PIL import Image
import IPython


def show_array(a, fmt='jpeg'):
    """
    Display array in Jupyter cell output using ipython widget.

    params:
        a (np.array): the input array
        fmt='jpeg' (string): the extension type for saving. Performance varies
                             when saving with different extension types.
    """
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    f = io.BytesIO() # get byte stream
    Image.fromarray(a).save(f, fmt) # save array to byte stream
    display(IPython.display.Image(data=f.getvalue())) # display saved array