"""
Use the biomes.csv file to classify any input data into a biome.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

DATA_DIR = "./data"


@dataclass
class Observation:
    MeanAnnualTemp: float
    MeanAnnualPrec: float


# Load the biomes data
biomes = pd.read_csv(f"{DATA_DIR}/biomes.csv", index_col="class")
# The columns are ["MAT mean", "MAT std", "MAP mean", "MAP std"]


def classify(observation: Observation, data: pd.DataFrame) -> pd.DataFrame:
    """
    Classify an observation into biomes using the naive Bayes classifier on iid normal distributions.

    Assuming that all biomes are equally likely and that the data is normally distributed
    we can use the mean and standard deviation of each biome to calculate the probability
    of a given data point belonging to each biome.
    $p(X \in B | X = x) = \frac{p(X = x | X \in B)}{\sum_{i=1}^{n} [p(X = x | X \in B_i)]}$
    where $B$ is a biome, $B_i$ is the $i$th biome, $X$ is a data point, and $x$ is the value
    of the data point.

    Args:
        observation (Observation): The observation to classify.
        data (pd.DataFrame): The data to use for classification.

    Returns:
        pd.DataFrame: A dataframe with the probability of the observation belonging to each biome.
    """
    # Calculate the probability of the observation belonging to each biome
    probs = data.apply(lambda row: (
        (1 / (np.sqrt(2 * np.pi) * row["MAT std"])) * np.exp(-((observation.MeanAnnualTemp - row["MAT mean"]) ** 2) / (2 * row["MAT std"] ** 2)) *
        (1 / (np.sqrt(2 * np.pi) * row["AP std"])) * np.exp(-(
            (observation.MeanAnnualPrec - row["AP mean"]) ** 2) / (2 * row["AP std"] ** 2))
    ), axis=1)
    # Normalize the probabilities
    probs = probs / probs.sum()

    probs.index = data.index

    return probs


obs = Observation(40, 10)
classes = classify(obs, biomes).round(4)
print(classes)
print(
    f"This observation is a {classes.idxmax()} with a probability of {classes.max()}")
