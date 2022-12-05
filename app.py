import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ppscore as ps
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Couples Therapy Divorce Predictions')

@st.cache
def read_data():
	# Read Data
	df = pd.read_csv("divorce_data.csv", sep=";")
	# Drop rows that contain null
	df.dropna()
	# Divorce columns need to be inverted: 1 should mean divorced and 0 should mean not divorced
	df.Divorce = df.Divorce.map({0:1, 1:0})
	return df

@st.cache
def select_features(df):
	corr = df.corr().abs()
	matrix_df = ps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
	corr_target = corr['Divorce']
	pps_target = matrix_df['Divorce']
	pps_target = pps_target[pps_target > 0.7]
	target = pps_target[corr_target > 0.8]
	target_features = list(target.keys())
	df = df.drop(df.columns.difference(target_features), axis=1)
	corr = df.corr().abs()
	redundant_features = set()
	for i in range(len(corr.columns)):
	    for j in range(i):
	        if corr.iloc[i, j] > 0.95: # Remove redundant features with > 95% correlation
	            redundant_features.add(corr.columns[i])
	df = df.drop(redundant_features, axis=1)
	return df

@st.cache
def create_model(df):
	# Drop predicting variable Divorce into y
	X = df.drop(columns=['Divorce'],axis=1) 
	y = df.Divorce

	# Create model training sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

	# Train Logistic Regression Model
	lr = LogisticRegression(random_state=13)
	lr.fit(X_train, y_train)
	return lr

    
 # DISCLAIMER: This function is taken directly from the wine-example-app lab and slightly modified
def visualize_confidence_level(prediction_proba):
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ["Won't get Divorced","Will get Divorced"])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#dc143c', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Confidence Level Percentage", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Future of Marriage?", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

df = select_features(read_data())
model = create_model(df)

st.sidebar.header('Therapy Questions')
st.sidebar.subheader('We reduced 54 therapy questions down to 7.\n Use the sliders to determine how much you agree with each statement. (0 is Disagree, 4 is Agree)')
# Some number in the range 0-4; default 2 (neutral)
q1_filter = st.sidebar.slider('Q5: The time I spend with my spouse is special for the both of us.', 0, 4, 2)
q2_filter = st.sidebar.slider('Q9: I look forward to vacations with my spouse and enjoy the travel we do together.', 0, 4, 2)
q3_filter = st.sidebar.slider('Q17: My spouse and I have similar ideas about how we find happiness in life.', 0, 4, 2)
q4_filter = st.sidebar.slider('Q18: My spouse and I have similar ideas about how marriage should be.', 0, 4, 2)
q5_filter = st.sidebar.slider('Q29: I know my spouse very well.', 0, 4, 2)
q6_filter = st.sidebar.slider('Q36: I am humble in discussions with my spouse.', 0, 4, 2)
q7_filter = st.sidebar.slider('Q40: I know why my partner is upset before we have an argument.', 0, 4, 2)

user_features = pd.DataFrame({'Q5':[q1_filter],
	                   'Q9':[q2_filter],
	                   'Q17':[q3_filter],
	                   'Q18':[q4_filter],
	                   'Q29':[q5_filter],
	                   'Q36':[q6_filter],
	                   'Q40':[q7_filter]})

prediction = model.predict(user_features)
st.subheader('Prediction on Marriage: %s' %("Won't get Divorced" if prediction[0] == 0 else "Will get Divorced"))
prediction_proba = model.predict_proba(user_features)
visualize_confidence_level(prediction_proba)

st.subheader("Data used for model training:")
st.dataframe(df)