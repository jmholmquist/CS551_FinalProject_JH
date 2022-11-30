import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

st.title("Features Effects on Housing Prices")
st.subheader("Research Question: On a neighborhood block, does the median age of the houses, the total rooms,\
    the population, the number of households, and the median income predict the median price values of the homes?")

#import the CSV File
data = pd.read_csv('housing.csv')

featureFrame = data[['housing_median_age', 'total_rooms', 'population', 'households', 'median_income', 'median_house_value']]
medianValue = data['median_house_value'].median()
minValue = data['median_house_value'].min()
maxValue = data['median_house_value'].max()

st.write("The relevant features of the dataset are shown below: ")
st.write(featureFrame)

st.write("The median value of the median home values in the dataset is " + str(medianValue))

#Set up training and testing set
trainSet, testSet = train_test_split(featureFrame)
bins = [minValue-1, medianValue, maxValue+1]
labels = [0, 1]
trainLabels = pd.cut(x=trainSet['median_house_value'], bins=bins, labels=labels).to_numpy()
testLabels = pd.cut(x=testSet['median_house_value'], bins=bins, labels=labels).to_numpy()
trainSet = trainSet[['housing_median_age', 'total_rooms', 'population', 'households', 'median_income']].to_numpy()
testSet = testSet[['housing_median_age', 'total_rooms', 'population', 'households', 'median_income']].to_numpy()

#Scale the values to mean 0 and standard deviation 1
scaler = preprocessing.StandardScaler()
trainSet = scaler.fit_transform(trainSet)
testSet = scaler.fit_transform(testSet)

#Fit A Linear SVM Model
linearSVM_model = LinearSVC(C=0.001, loss="hinge")
linearSVM_model.fit(trainSet, trainLabels)
results_linearSVM = linearSVM_model.predict(testSet)
ConfusionMatrixDisplay.from_estimator(linearSVM_model, testSet, testLabels)
st.pyplot(plt)
st.write("The accuracy score is: " + str(accuracy_score(testLabels, results_linearSVM)))