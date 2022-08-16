# A convolutional model to extract RGB features from given image/s


## Install
Clone the project and install it:

```
git clone git@github.com:alecsharpie/conv_rgb.git
cd conv_rgb
make install # or `pip install .`
```

## Example

```python
from conv_rgb.model import ConvRGB
import numpy as np
from PIL import Image

model = ConvRGB(input_shape = (128, 128), n_cut = 3)

img = Image.open(image.png).resize((128, 128))

X = np.expand_dims(img, axis = 0)

model(X) # X should be shape (n_samples, n_width, n_height, 3 channels)
```

ConvRGB: Init parameters
 - input_shape = tuple (width, height)
 - n_cut = number of slices in each side of the grid, each cell of the grid produces 3 colour features (R, G, B)

Eg. n_cuts = 3
create a 3x3 grid of 9 colour patches
3 channels (R, G, B) for each patch
total 27 colour features

![n_cut Example](https://github.com/alecsharpie/conv_rgb/blob/master/n_cuts_example.png?raw=true)
