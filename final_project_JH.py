import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def Predict(model, features):
    st.header("The sale price will be: ")
    result = model.predict(features.reshape(1, -1))
    if sum(features) == 0:
        result = 10

    if result == 10:
        st.subheader("Input Model Features to see a prediction...")
    elif result == 1:
        st.subheader("Above the Median Sale Price")
    
    else:
        st.subheader("Below the Median Sale Price")
    

st.title("Predicting House Sale Prices in Miami")
st.subheader("John Holmquist")
st.subheader("Research Question 1: Can it be predicted based on characteristics of a house whether the sale price is above or below the median sale price for houses in Miami?")

#import the CSV File
data = pd.read_csv('miami-housing.csv')

medianValue = data['SALE_PRC'].median()

#Add PriceAboveMedian label to data frame
labels = np.zeros((len(data.SALE_PRC),), dtype=int)
labels[data['SALE_PRC'] >= medianValue]  = 1
data['PriceAboveMedian'] = labels.tolist()

#Set up training and testing set
trainSet, testSet = train_test_split(data)

#Create labels and select features set
featureSet6 = ['LND_SQFOOT', 'TOT_LVG_AREA', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'SPEC_FEAT_VAL']
trainLabels = trainSet['PriceAboveMedian'].to_numpy()
testLabels = testSet['PriceAboveMedian'].to_numpy()
trainingData = trainSet[featureSet6].to_numpy()
testingData = testSet[featureSet6].to_numpy()

displayTrain = trainSet[featureSet6]
displayTest = testSet[featureSet6]
displayTrain['PriceAboveMedian'] = trainLabels.tolist()
displayTest['PriceAboveMedian'] = testLabels.tolist()

st.header("Training Data")
st.write(displayTrain.head())
st.header("Testing Data")
st.write(displayTest.head())

#Scale the values to mean 0 and standard deviation 1
scaler = StandardScaler()
trainSet = scaler.fit_transform(trainingData)
testSet = scaler.fit_transform(testingData)

#Fit a decision tree model
treeMaxDepth = 15
model = DecisionTreeClassifier(max_depth=treeMaxDepth)
model.fit(trainSet, trainLabels)
results = model.predict(testSet)

st.header("Model Type: Decision Tree Classifier")
st.subheader("Classification Rate: " + "{:.2f}".format(accuracy_score(testLabels, results)))

lnd_sqfoot = 0
tot_lvg_area = 0
ocean_dist = 0
water_dist = 0
cntr_dist = 0
subcntr_di = 0
spec_feat_val = 0

st.header("Use The Model Yourself")
st.write("A prediction will be made each time you enter a value.")
lnd_sqfoot = st.number_input("Input the square footage of the lot: ")
tot_lvg_area = st.number_input("Input the square footage of the house: ")
ocean_dist = st.number_input("Input in feet the distance of the house from the ocean: ")
water_dist = st.number_input("Input in feet the distance of the house from the nearest body of water: ")
cntr_dist = st.number_input("Input in feet the distance of the house from the Miami central business district: ")
subcntr_di = st.number_input("Input in feet the distance of the house from the nearest subcenter: ")
spec_feat_val = st.number_input("Input in dollars the value of special features of the house: ")

features = np.array([lnd_sqfoot, tot_lvg_area, ocean_dist, water_dist, cntr_dist, subcntr_di, spec_feat_val])

Predict(model, features)