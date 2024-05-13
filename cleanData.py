# This file contains the code to clean the data and save it to a new file

'''The column indices for the crime data:
    0: 'X' : float, 
    1: 'Y' : float, 
    2: 'RowID' : int, 
    3: 'CCNumber' : string, 
    4: 'CrimeDateTime' : string, 
    5: 'CrimeCode' : string, 
    6: 'Description' : string, 
    7: 'Inside/Outside' : I or O, 
    8: 'Weapon' : string including 'NA', 
    9: 'Post' : int, 
    10: 'Gender' : M, F, or U,
    11: 'Age' : int, 
    12: 'Race' : string, 
    13: 'Ethnicity' : string,, 
    14: 'Location' : string, 
    15: 'Old_District' : string, 
    16: 'New_District' : string, 
    17: 'Neighborhood' : string, 
    18: 'Latitude' : float, 
    19: 'Longitude' : float, 
    20: 'GeoLocation' : string, 
    21: 'PremiseType' : string, 
    22: 'Total_Incidents' : int
'''

import pandas as pd
import geopandas as gpd

# Function to return a cleaned version of the crime data
def clean_crime_data(crimeData):

    # Focus on the following columns: 'CrimeDateTime', 'Description', 'Gender', 
    # 'Age', 'Race', 'Neighborhood', 'Latitude', 'Longitude', 'PremiseType'
    crimeData = crimeData.iloc[:, [4, 6, 10, 11, 12, 17, 18, 19, 21]]

    # Function to remove rows with dates that are not in 2020, 2021, or 2022
    def remove_invalid_dates(data):
        # remove rows with dates that are not in 2020, 2021, or 2022
        cleaned_data = data[data['CrimeDateTime'].str.contains('2020|2021|2022')]
        return cleaned_data
    
    # Function to combine similar descriptions
    def combine_similar_descriptions(data):
        # change 'LARCENY FROM AUTO' to 'LARCENY'
        cleaned_data = data.replace('LARCENY FROM AUTO', 'LARCENY')
        # change 'ROBBERY - CARJACKING' to 'ROBBERY'
        cleaned_data = cleaned_data.replace('ROBBERY - CARJACKING', 'ROBBERY')
        # change 'ROBBERY - COMMERCIAL' to 'ROBBERY'
        cleaned_data = cleaned_data.replace('ROBBERY - COMMERCIAL', 'ROBBERY')
        return cleaned_data
    
    # Function to clean the 'Gender' column
    def clean_gender_column(data):
        # Get unique values in the 'Gender' column
        unique_values = ['B', 'Transgende', 'N', ',', 'FB', 'O', '160', 'FW', 'FU', 'D', '60', '120', '8', 
                         'MB', 'A', '77', '17', 'FF', '165', 'FM', '042819', 'S', 'T', '50']
        
        # Create a dictionary to map unique values to 'U' except for 'M' and 'F'
        replace_dict = {value: 'U' for value in unique_values if value not in ['Male', 'Female', 'W', 'M\\']}
        
        # Replace 'Male' with 'M', 'Female' with 'F', 'W' with 'F', and 'M\' with 'M'
        replace_dict['Male'] = 'M'
        replace_dict['Female'] = 'F'
        replace_dict['W'] = 'F'
        replace_dict['M\\'] = 'M'

        # Replace the values with the dictionary
        cleaned_data = data.replace(replace_dict)

        # Fill NaN values with 'U'
        cleaned_data = cleaned_data.fillna('U')

        return cleaned_data
    
    # Function to clean the 'Age' column
    def clean_age_column(data):
        # Custom function to convert values to integers and handle 'U' values
        def convert_to_int(value):
            try:
                return int(value)
            except (ValueError, TypeError):
                return 'U'
            
        # Replace numbers 0 and below + numbers 115 and over with 'U'
        cleaned_data = data.apply(lambda x: 'U' if x <= 0 or x >= 115 else x)

        # Fill NaN values with 'U'
        cleaned_data = cleaned_data.fillna('U')
            
        # Apply the custom function to the 'Age' column
        cleaned_data = cleaned_data.apply(convert_to_int)

        return cleaned_data
    
    # Function to clean the 'Race' column
    def clean_race_column(data):
        # Fill NaN values with 'UNKNOWN'
        cleaned_data = data.fillna('UNKNOWN')
        return cleaned_data
    
    # Function to clean the neighborhood column
    def clean_neighborhood_column(data):
        # empty the Neighborhood column
        cleaned_data = data.drop(columns=['Neighborhood'])

        # create a new column 'Neighborhood' using the neighborhood shapefile
        # load the neighborhood shapefile
        neighborhoods = gpd.read_file("Neighborhood.geojson")
        # create a new column 'Neighborhood' by mapping the 'Latitude' and 'Longitude' columns to the neighborhood shapefile
        cleaned_data['Neighborhood'] = gpd.points_from_xy(data['Longitude'], data['Latitude']).map(
            lambda x: neighborhoods[neighborhoods.contains(x)]['Name'].values[0] if len(neighborhoods[neighborhoods.contains(x)]) > 0 else 'UNKNOWN')
        
        # drop the 'UNKNOWN' rows
        cleaned_data = cleaned_data[cleaned_data['Neighborhood'] != 'UNKNOWN']

        # make the neighborhood values uppercase
        cleaned_data['Neighborhood'] = cleaned_data['Neighborhood'].str.upper()

        return cleaned_data
    
    # Function to remove rows with NaN and 0 values in the 'Longitude' and 'Latitude' column
    def delete_invalid_location_rows(data):
        # delete rows with NaN values in the 'Longitude' and 'Latitude' column
        cleaned_data = data.dropna(subset=['Longitude', 'Latitude'])
        # delete rows with 0 values in the 'Longitude' and 'Latitude' column
        cleaned_data = cleaned_data[cleaned_data['Longitude'] != 0]
        cleaned_data = cleaned_data[cleaned_data['Latitude'] != 0]
        return cleaned_data
    
    # Function to clean the 'PremiseType' column
    def clean_premise_type_column(data):
        # Fill NaN values with 'UNKNOWN'
        cleaned_data = data.fillna('UNKNOWN')
        return cleaned_data
    
    # Call the functions
    crimeData = remove_invalid_dates(crimeData)
    crimeData = combine_similar_descriptions(crimeData)
    crimeData['Gender'] = clean_gender_column(crimeData['Gender'])
    crimeData['Age'] = clean_age_column(crimeData['Age'])
    crimeData['Race'] = clean_race_column(crimeData['Race'])
    crimeData = delete_invalid_location_rows(crimeData)
    crimeData = clean_neighborhood_column(crimeData)
    crimeData['PremiseType'] = clean_premise_type_column(crimeData['PremiseType'])

    return crimeData


# create a new cleaned file for the crime data
# _________________________________________________________________________
crimeData = pd.read_csv("Part_1_Crime_Data.csv")
crimeData = clean_crime_data(crimeData)
crimeData.to_csv("Cleaned_Part_1_Crime_Data.csv", index=False)
# _________________________________________________________________________