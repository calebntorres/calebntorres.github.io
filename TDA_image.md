## A Brief Walkthrough of Persistance Diagram and Image Applicatons to Image Modeling (Under Construction)

**Project description:** The discipline of Topology is broadly concerned with the notions of continuity, "conectedness" and "closeness" of geometric objects and their structure. When we seek to understand the global features data, and we observe that the data in question has a geometric semblance, we can apply topological concepts to determine any important features of the data. In this brief walkthrough I will demonstrate how we apply topological methods and concepts to grasp the global features of image data.

### 1. Why Image Data?

Image processing and recognition are important capabilities in machine learning. One particular problem involves identifying any distinct global components or features of any given image. It might be beneficial for a program to percieve the global features of an image as preliminary step to further processing. Through topological data analysis (TDA) We hope to define procedures which address this task. Take the following image.

<img src="images/gradient_circle_2.png?raw=true"/>

We clearly percieve certain geometric qualities in the image. Globally, the image is radially gradient and symmetric. The question is how we proceed to identify these features and ultimately produce a summary. With TDA, we can leverage the concepts of persistant homology to impose a structure on the data that will show us what features are persistant and what features are due to noise. Without discussing the theoretical foundations of TDA, we can assume that if our data has significant goemetric features, then our imposed structure will register them accordingly. 

### 2. Data Pipeline
First, we identify how we will prepare our image data. For that, we will rely heavily on Pillow to handle image parsing. Then, we will need some visualization libraries, data handling libraries and finally our TDA libraries. I will highlight each library as we go along.

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
