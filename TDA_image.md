## A Brief Walkthrough of Persistance Diagram and Image Applicatons to Image Modeling (Under Construction)

**Project description:** The discipline of Topology is broadly concerned with the notions of continuity, "conectedness" and "closeness" of geometric objects. Thus, when we wish to understand the global features of our data, and we suspect some kind of geometric semblance to our data as well, then we can use the concepts of topology to analyze and explore our data. In this brief walkthrough, I aim to describe a process by wich we may take a piece of data and apply topological methods to extract useful features at play.

### 1. Image Data?

Image processing and recognition are bustling fields in machine learning and data science. One particular problem of interest involves identifying the distinct components or features of an image. Given an image such as the following, it might be important for a program to become acqianted with the global features of the image as a first step. In this case, We hope to define a procedure which picks up on the gradient nature of the image and is able to express this fact in a clear manner.

<img src="images/gradient_circle_2.png?raw=true"/>

To detect the gradient of color, we would need a notion of "closeness" or "similarity" of certain pixels with others. To detect symmetry we would need to invoke some kind of geometrical parameterization that relates each pixel in terms of its position and color.

Perhaps the most basic way to determine which pixels are more similar to one another is to conisder the coordinate distance of pixels. We view the image as a n x n grid of pixels and then compute distances between pixels as a measure of what pixels should be taken together. This approach is inspired by a K-NN model.

However, since we are interested in the behavior of global features of our data, and the data is assumed to demonstrate some geometric qualities, a TDA approach would be able to detect such features along with any geometric signature.

### 2. Data Pipeline
For packages, we will need standard data handling tools such as numpy and pandas, as well matplotlib for visualizations. Since we are dealing with image parsing, Pillow can be invoked to easily convert a standard jpeg or png image file into an object that we can iterate over with itertools. Then, we finally employ our TDA packages such as ripser and persim.
```python
from PIL import Image
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from ripser import ripser
from ripser import Rips
from persim import plot_diagrams
import sys

```
Then we simply parse our image with Pillow, and proceed to structure the data into a pandas dataframe.
```python
image = Image.open("gradient_circle.png")

# create numpy array
image_array = np.array(image)
print(image_array)
# represents dimension of array as an n-tuple. Dimensions of array should match image dimensions.
print(image_array.shape) 

# creating dataframe from numpy array
data = list(itertools.chain(*image_array))
image_df = pd.DataFrame.from_records(data)

print(image_df.head(5)) # show first five data rows
print(image_df.tail(5)) # show last five data rows
print(list(image_df.columns))

```


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
