"""
Why do we need Encoding?

ML models work with numbers, but categorical data contains text.

Example:

    Gender
    Male
    Female
    Male
    Female

Models cannot directly work with "Male" and "Female".

Encoding converts categorical data into numerical form.
"""

#                     Label Encoder
# LabelEncoder converts each category into a unique integer.
# Mainly used for target/label (y), not input features.
# Example: pass -> 1, fail -> 0

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load CSV into Pandas DataFrame
df = pd.read_csv("./encoding_practice.csv")

# Select categorical column
gender = df["Gender"]

# Convert Pandas column to Numpy array
gender_array = np.array(gender)


# Create LabelEncoder object
le = LabelEncoder()

# Learn Categories and encode them
encoded_gender = le.fit_transform(gender_array)
print(encoded_gender) #to show encoded values

print(le.classes_) # To see category to label mapping

# Add encoded values to DataFrame
df["Gender_encoded"] = encoded_gender

# Compare original and encoded values
print(df[["Gender", "Gender_encoded"]])



#                        Experience_Level Example

# Select categorical column
experience_level = df["Experience_Level"]

# Convert Pandas column to Numpy array
experience_array = np.array(experience_level)

# Create LabelEncoder object
le = LabelEncoder()

# Learn Categories and encode them
encoded_experience = le.fit_transform(experience_array)
print(encoded_experience) #to see encoded values

print(le.classes_) # To see category to label mapping

# Add encoded values to DataFrame
df["Encoded_Experience"] = encoded_experience

# Compare original and encoded values
print(df[["Experience_Level","Encoded_Experience"]])


