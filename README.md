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
 - n_cut = number of slices in each side of the grid, each cell of the grid produces 3 colour features (R, G, B) eg 3 will create a 3x3 grid and so 9 colour patches -> 27 colour features
Eg. n_cuts = 3
![n_cut Example](n_cut_example.png "Number of cuts")
