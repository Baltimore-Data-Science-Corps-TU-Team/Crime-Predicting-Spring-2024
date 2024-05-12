import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
import folium

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

st.set_page_config(page_title="Baltimore Crime", layout="wide")

# Read the data from the clean csv files

# 'CrimeDateTime', 'Description', 'Gender', 'Age', 'Race', 'Neighborhood', 'Latitude', 'Longitude', 'PremiseType'
crimeData = pd.read_csv('Cleaned_Part_1_Crime_Data.csv')

# convert the CrimeDateTime to datetime
crimeData['CrimeDateTime'] = crimeData['CrimeDateTime'].str.split("+").str[0]
crimeData['CrimeDateTime'] = pd.to_datetime(crimeData['CrimeDateTime'])

with st.sidebar:
    st.title("Baltimore Crime Data")
    st.caption("Adjust the filters for the crime data")

    # display a date input to select the date range
    selectedDate = st.date_input("Select Date Range", 
                                    [crimeData['CrimeDateTime'].min(), crimeData['CrimeDateTime'].max()],
                                    min_value=crimeData['CrimeDateTime'].min(),
                                    max_value=crimeData['CrimeDateTime'].max()
                                 )
    
    # handle the case where the tuple has a single value
    if len(selectedDate) == 1:
        selectedDate = [selectedDate[0], selectedDate[0]]

    # display a multi-select box to select the different types of crimes
    selectedCrime = st.multiselect(
        "Select Crime Descriptions",
        list(crimeData['Description'].unique()),
        placeholder="All"
    )

    # display a multi-select box to select the different genders
    selectedGender = st.multiselect(
        "Select Genders",
        list(crimeData['Gender'].unique()),
        placeholder="All"
    )

    # display a multi-select box to select the different races
    selectedRace = st.multiselect(
        "Select Races",
        list(crimeData['Race'].unique()),
        placeholder="All"
    )

    # display a multi-select box with all the neighborhoods in alphabetical order to select the different neighborhoods
    selectedNeighborhood = st.multiselect(
        "Select Neighborhoods",
        list(crimeData['Neighborhood'].unique()),
        placeholder="All"
    )

    # display a multi-select box to select the different premise types
    selectedPremiseType = st.multiselect(
        "Select Premise Types",
        list(crimeData['PremiseType'].unique()),
        placeholder="All"
    )

# filter the data based on the selected filters
filteredCrimeData = crimeData["Description"].str.contains("|".join(selectedCrime)) & \
    crimeData["Neighborhood"].str.contains("|".join(selectedNeighborhood)) & \
        crimeData["PremiseType"].str.contains("|".join(selectedPremiseType)) & \
            crimeData["Gender"].str.contains("|".join(selectedGender)) & \
                crimeData["Race"].str.contains("|".join(selectedRace)) & \
                    crimeData["CrimeDateTime"].between(pd.to_datetime(selectedDate[0]), pd.to_datetime(selectedDate[1]))

allTab, predTab = st.tabs(["All Crime", "Predictions"])

with allTab:
    st.caption("Takes into account all the filters")
    st.metric(label="Total Crime", value=crimeData[filteredCrimeData].shape[0])


with predTab:
    # display a select box to select the machine learning model
    # selectedModel = st.selectbox(
    #     "Select Machine Learning Model",
    #     ["Random Forest", "Extra Trees", "Decision Tree", "Baggin Regressor"],
    # )
    st.header("Decision Tree Model Predictions")

# ------------------------- Machine Learning -------------------------
def predictDatesModel(crimeData):
    # create a new dataframe with a 'CrimeDateTime' column that is converted to datetime and sorted from latest to oldest
    crimeDataML = pd.DataFrame()
    crimeDataML['CrimeDateTime'] = pd.to_datetime(crimeData['CrimeDateTime'])
    crimeDataML = crimeDataML.sort_values('CrimeDateTime', ascending=True)
    crimeDataML['TimeDiff'] = crimeDataML['CrimeDateTime'].diff().dt.seconds.div(60).fillna(0) # in minutes
    
     # loop through the data and calculate the time difference between each crime
    for i in range(1, 51):
        crimeDataML[f'TimeDiff-{i}-Back'] = crimeDataML['TimeDiff'].shift(i)

    # drop the rows with NaN values
    crimeDataML = crimeDataML.dropna()

    # create a linear regression model
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25 = [np.array(crimeDataML[f'TimeDiff-{i}-Back']).reshape(-1, 1) for i in range(1, 26)]
    x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50 = [np.array(crimeDataML[f'TimeDiff-{i}-Back']).reshape(-1, 1) for i in range(26, 51)]

    y = np.array(crimeDataML['TimeDiff']).reshape(-1, 1)

    X = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, 
                        x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50), axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # create a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    model.feature_names_in_ = list(crimeDataML.columns[2:])

    # create an empty dataframe to store the predicted time difference
    predCrimeTimes = pd.DataFrame(crimeDataML.iloc[-1].values.reshape(1, -1), columns=crimeDataML.columns)

    # create a loop to predict future crime dates and times
    for i in range(1, 51):
        lastRow = pd.DataFrame(predCrimeTimes.iloc[-1].values.reshape(1, -1), columns=predCrimeTimes.columns)
        nextPred = model.predict(lastRow[model.feature_names_in_])[0][0]
        nextRow = pd.DataFrame(columns=lastRow.columns)
        nextRow['CrimeDateTime'] = lastRow['CrimeDateTime']
        nextRow['TimeDiff'] = nextPred
        nextRow['CrimeDateTime'] = nextRow['CrimeDateTime'] + pd.to_timedelta(nextRow['TimeDiff'], unit='m')
        for j in range(1, 51):
            if j > 1:
                nextRow[f'TimeDiff-{j}-Back'] = lastRow[f'TimeDiff-{j-1}-Back']
            else:
                nextRow[f'TimeDiff-{j}-Back'] = lastRow['TimeDiff']

        predCrimeTimes = pd.concat([predCrimeTimes, nextRow])

    return predCrimeTimes

predictedDates = pd.DataFrame()
if (crimeData[filteredCrimeData].shape[0] > 100):
    predictedDates = predictDatesModel(crimeData[filteredCrimeData])

def locationsModel(crimeData):

    crimeDataML = crimeData[['Latitude', 'Longitude', 'CrimeDateTime']]
    crimeDataML['Year'] = crimeDataML['CrimeDateTime'].dt.year
    crimeDataML['Month'] = crimeDataML['CrimeDateTime'].dt.month
    crimeDataML['Day'] = crimeDataML['CrimeDateTime'].dt.day
    crimeDataML['Hour'] = crimeDataML['CrimeDateTime'].dt.hour
    crimeDataML['Minute'] = crimeDataML['CrimeDateTime'].dt.minute
    crimeDataML['DayOfWeek'] = crimeDataML['CrimeDateTime'].dt.dayofweek
    crimeDataML['isNight'] = np.where((crimeDataML['Hour'] >= 18) | (crimeDataML['Hour'] < 6), 1, 0)
    crimeDataML['MonthCos'] = np.cos(2 * np.pi * crimeDataML['Month'] / 12)
    crimeDataML['MonthSin'] = np.sin(2 * np.pi * crimeDataML['Month'] / 12)
    crimeDataML['DayCos'] = np.cos(2 * np.pi * crimeDataML['Day'] / 31)
    crimeDataML['DaySin'] = np.sin(2 * np.pi * crimeDataML['Day'] / 31)
    crimeDataML['HourCos'] = np.cos(2 * np.pi * crimeDataML['Hour'] / 24)
    crimeDataML['HourSin'] = np.sin(2 * np.pi * crimeDataML['Hour'] / 24)
    crimeDataML['MinuteCos'] = np.cos(2 * np.pi * crimeDataML['Minute'] / 60)
    crimeDataML['MinuteSin'] = np.sin(2 * np.pi * crimeDataML['Minute'] / 60)
    crimeDataML['DayOfWeekCos'] = np.cos(2 * np.pi * crimeDataML['DayOfWeek'] / 7)
    crimeDataML['DayOfWeekSin'] = np.sin(2 * np.pi * crimeDataML['DayOfWeek'] / 7)

    # drop the CrimeDateTime column as it is no longer needed
    crimeDataML = crimeDataML.drop(columns=['CrimeDateTime'])

    # split the data into features and target
    X = crimeDataML.drop(columns=['Latitude', 'Longitude'])
    y = crimeDataML[['Latitude', 'Longitude']]

    # order the features in the order of importance
    X = X[['isNight', 'HourCos', 'DayOfWeekCos', 'DayOfWeek', 'MinuteSin', 'DayOfWeekSin', 'DaySin', 'DayCos', 'Minute', 'Day', 'Month', 'Hour', 'MinuteCos', 'MonthCos', 'Year', 'MonthSin', 'HourSin']]

    # split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # create a model based on the selected model
    # if selectedModel == "Random Forest":
    #     model = RandomForestRegressor(random_state=13)
    # elif selectedModel == "Extra Trees":
    #     model = ExtraTreesRegressor(random_state=13)
    # elif selectedModel == "Decision Tree":
    #     model = DecisionTreeRegressor(random_state=13)
    # elif selectedModel == "Baggin Regressor":
    #     model = BaggingRegressor(random_state=13)

    model = DecisionTreeRegressor(random_state=13)

    # fit the model with the training data
    model.fit(X_train, y_train)

    return model

if (crimeData[filteredCrimeData].shape[0] > 0):
    model = locationsModel(crimeData[filteredCrimeData])
else:
    st.toast("To predict future crime locations adjust filters to have more than 100 crimes")

def predictLocationsModel(predictedDates, model):
    predictedDates = predictedDates[['CrimeDateTime']]

    predictedData = predictedDates.copy()
    predictedData['Year'] = predictedData['CrimeDateTime'].dt.year
    predictedData['Month'] = predictedData['CrimeDateTime'].dt.month
    predictedData['Day'] = predictedData['CrimeDateTime'].dt.day
    predictedData['Hour'] = predictedData['CrimeDateTime'].dt.hour
    predictedData['Minute'] = predictedData['CrimeDateTime'].dt.minute
    predictedData['DayOfWeek'] = predictedData['CrimeDateTime'].dt.dayofweek
    predictedData['isNight'] = np.where((predictedData['Hour'] >= 18) | (predictedData['Hour'] < 6), 1, 0)
    predictedData['MonthCos'] = np.cos(2 * np.pi * predictedData['Month'] / 12)
    predictedData['MonthSin'] = np.sin(2 * np.pi * predictedData['Month'] / 12)
    predictedData['DayCos'] = np.cos(2 * np.pi * predictedData['Day'] / 31)
    predictedData['DaySin'] = np.sin(2 * np.pi * predictedData['Day'] / 31)
    predictedData['HourCos'] = np.cos(2 * np.pi * predictedData['Hour'] / 24)
    predictedData['HourSin'] = np.sin(2 * np.pi * predictedData['Hour'] / 24)
    predictedData['MinuteCos'] = np.cos(2 * np.pi * predictedData['Minute'] / 60)
    predictedData['MinuteSin'] = np.sin(2 * np.pi * predictedData['Minute'] / 60)
    predictedData['DayOfWeekCos'] = np.cos(2 * np.pi * predictedData['DayOfWeek'] / 7)
    predictedData['DayOfWeekSin'] = np.sin(2 * np.pi * predictedData['DayOfWeek'] / 7)

    # drop the CrimeDateTime column as it is no longer needed
    predictedData = predictedData.drop(columns=['CrimeDateTime'])

    # order the features in the order of importance
    predictedData = predictedData[['isNight', 'HourCos', 'DayOfWeekCos', 'DayOfWeek', 'MinuteSin', 'DayOfWeekSin', 'DaySin', 'DayCos', 'Minute', 'Day', 'Month', 'Hour', 'MinuteCos', 'MonthCos', 'Year', 'MonthSin', 'HourSin']]

    # create a dataframe of the predicted latitude and longitude
    predicted = pd.DataFrame()
    predicted['latitude'] = model.predict(predictedData)[:, 0]
    predicted['longitude'] = model.predict(predictedData)[:, 1]

    with predTab:

        col1, col2 = st.columns([1, 4])
        with col1:
            st.write("Next 50 Predicted Crime Dates and Times")
            st.dataframe(predictedDates, hide_index=True, width=200)
            

        with col2:
            st.write("Next 50 Predicted Crime Locations")
            crime_pred_map = folium.Map(location=[predicted['latitude'].mean(), predicted['longitude'].mean()], zoom_start=12)

            for index, row in predicted.iterrows():
                folium.Marker(
                    [row['latitude'], row['longitude']], 
                    radius=3,
                    color='red',
                    icon=folium.Icon(color='red', icon='info-sign'),
                    tooltip=f"Crime Number: {index+1} Datetime: {predictedDates.iloc[index]['CrimeDateTime'].strftime('%Y-%m-%d %H:%M:%S')}"
                    ).add_to(crime_pred_map)

            crime_pred_map.save('crime_pred_map.html')
            st.components.v1.html(open('crime_pred_map.html', 'r').read(), height=600)

predictLocationsModel(predictedDates, model)