
#                     Ordinal Encoder
# OrdinalEncoder converts ordered categories into numbers.
# It preserves the natural order of categories.
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Load csv into pandas dataframe
df = pd.read_csv("./encoding_practice.csv")

# select categorical column
experience_level = df[["Experience_Level"]]

# Convert into numpy array
experience_array = np.array(experience_level)

# create ordinalEncoder object    0                1             2
oe = OrdinalEncoder(categories=[["Beginner", "Intermediate", "Advanced"]])

# encode the categories
encoded_experience = oe.fit_transform(experience_array)
print(encoded_experience)

# Add encoded values to DataFrame
df["Actual_Experience"] = encoded_experience


print(df[["Experience_Level", "Actual_Experience"]])


#             Satisfaction Example

# Poor Good Average Excellent

#selection of column
satisfaction = df[["Satisfaction"]]

# convert to numpy array
satisfaction_array = np.array(satisfaction)

# ordinalEncoder object            0      1       2            3
oe = OrdinalEncoder(categories=[["Poor","Average", "Good", "Excellent"]])

# encode the column
encoded_satisfaction = oe.fit_transform(satisfaction_array)
print(encoded_satisfaction)

# assign encoded column to df
df["Encoded_Satisfaction"] = encoded_satisfaction
print(df[["Satisfaction", "Encoded_Satisfaction"]])
