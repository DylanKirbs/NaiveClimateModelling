"""
This program will calculate the mean temperature and precipitation for each biome and then use this data to train a Naive Bayes classifier to predict the biome of a given location based on the temperature and precipitation.

The data we will use is the classified 5m data and BIO1 (Annual Mean Temperature) and BIO12 (Annual Precipitation) from the WorldClim dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


DATA_DIR = "./data"
NAMES = [
    "",
    "Af",
    "Am",
    "Aw",
    "As",
    "BWh",
    "BWk",
    "BSh",
    "BSk",
    "Csa",
    "Csb",
    "Csc",
    "Cwa",
    "Cwb",
    "Cwc",
    "Cfa",
    "Cfb",
    "Cfc",
    "Dsa",
    "Dsb",
    "Dsc",
    "Dsd",
    "Dwa",
    "Dwb",
    "Dwc",
    "Dwd",
    "Dfa",
    "Dfb",
    "Dfc",
    "Dfd",
    "ET",
    "EF"
]

# Load the classification, BIO1, and BIO12 data
with rasterio.open(f"{DATA_DIR}/classification.tif") as classRaster:
    classification = classRaster.read(1).astype(float)
    classMeta = classRaster.meta

with rasterio.open(f"{DATA_DIR}/BIO1.tif") as bio1Raster:
    bio1 = bio1Raster.read(1).astype(float)
    bio1Meta = bio1Raster.meta

with rasterio.open(f"{DATA_DIR}/BIO12.tif") as bio12Raster:
    bio12 = bio12Raster.read(1).astype(float)
    bio12Meta = bio12Raster.meta

# Replace the no data values with NaN
classification[classification == 0] = np.nan
bio1[bio1 == bio1Meta["nodata"]] = np.nan
bio12[bio12 == bio12Meta["nodata"]] = np.nan


# Create a dataframe with the classification, BIO1, and BIO12 data
df = pd.DataFrame({
    "classification": classification.flatten(),
    "bio1": bio1.flatten(),
    "bio12": bio12.flatten()
})

# Calculate the mean temperature and precipitation for each biome as well as the standard deviation
biomes = df.groupby("classification").agg(["mean", "std"])
biomes.columns = ["MAT mean", "MAT std", "AP mean", "AP std"]

# Add a rename the index to the biome names
biomes["class"] = NAMES[1:]
biomes.index = biomes["class"]
biomes = biomes.drop("class", axis=1)

# Save the biomes dataframe to a csv file
biomes.to_csv(f"{DATA_DIR}/biomes.csv")
