## A Walkthrough of TDA Applied to Color Modeling

**Project description:** The discipline of Topology is concerned with several notions, including continuity, "conectedness" and especially "closeness." Topology also offers up a set of objects that allow us to parse through multi-dimensional data geometrically and gather meaningful information about the data. This project is a brief effort in showing how data and Topology can be united to model data. 

### 1. Why a Color Model?

Image processing and recognistion are busy fields in machine learnign and data science and the present time. One particular 

### 2. Data Pipeline
For packages, we will need standard data handling tools such as numpy and pands, as well matplotlib for visualizations. Since we are dealing with image parsing, Pillow can be invoked to easily convert a standard jpeg or png image file. Then, we finally employ our TDA packages such as ripser and persim.

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
Then we simply parse our image with Pillow.
```python
image = Image.open("gradient_circle.png.png")
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
