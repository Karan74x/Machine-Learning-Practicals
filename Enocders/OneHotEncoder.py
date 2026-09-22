#                     OneHotEncoder
# OneHotEncoder converts nominal (unordered) categories into binary (0/1) columns.
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load csv into pandas dataframe
df = pd.read_csv("./encoding_practice.csv")


#        Gender
gender = df[["Gender"]]

# Convert gender column into np array
gender_array = np.array(gender)

# OneHotEncoder object sparse_output=False returns a standard array
ohe = OneHotEncoder(sparse_output=False, drop="first")

# Encode the gender Array
encoded_gender = ohe.fit_transform(gender_array)


# get the unique values from the gender column like (Gender_Male & Gender_Female)
gender_column_names = ohe.get_feature_names_out(["Gender"])


# Assign encoded results to a DataFrame and pass the above column names
encoded_gender_df = pd.DataFrame(encoded_gender, columns=gender_column_names)
# print(encoded_gender_df)

print(pd.concat([df[["Gender"]], encoded_gender_df], axis=1))


#                        City Example

# first take the city column out
city = df[["City"]]

# convert it into numpy array
city_array = np.array(city)

#create the OneHotEncoder Object with sparse_output = False to get normal array
ohe = OneHotEncoder(sparse_output = False,drop="first")

# Encode the city_array
encoded_city = ohe.fit_transform(city_array)
# print(encoded_city)

# take out the unique names from city column
city_names = ohe.get_feature_names_out(["City"])


#assign encoded results to Dataframe with city_names
encoded_city_df = pd.DataFrame(encoded_city, columns=city_names)
# print(encoded_city_df)

# to see side by side comparison concat old column with new column
final_result = pd.concat([df[["City"]], encoded_city_df], axis=1)
print(final_result)
