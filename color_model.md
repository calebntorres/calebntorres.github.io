## A Walkthrough of TDA Applied to Color Modeling

**Project description:** The discipline of Topology is concerned with several notions, including continuity, "conectedness" and especially "closeness." Topology also offers up a set of objects that allow us to parse through multi-dimensional data geometrically and gather meaningful information about the data. This project is a brief effort in showing how data and Topology can be united to model data. 

### 1. Why a Color Model?

Image processing and recognistion are busy fields in machine learnign and data science and the present time. One particular problem of interest involves identifying the ditinct components or features of an image in computer vision scenarios. Given an image such as the following a machine may be tasked with identiyfing all global features in a rudimentary sense. In other words, we may task a machine with identifying the foreground, background and any other significant features. 

<img src="images/background_foreground.jpg?raw=true"/>

One way to accomplish this task is to recognize that we may use colors and contrasts between different areas of the image to segment the image into distinct parts. The idea is to instruct our program to identify situations where certain components of the image are similar enough to each other, and distinct enough from yet others. In this case, our compenents are individual pixels. Perhaps the most basic way to determine which pixels belong more with each other than others is to simply look at the coordinate distance of the pixles. We view the image as a n X n grid of pixels and then compute distances between pixels as a measure of what pixels should be taken together.
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
