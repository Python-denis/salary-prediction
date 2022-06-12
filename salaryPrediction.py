import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt #non interactive
from plotly import graph_objs as go #interactive
from sklearn.linear_model import LinearRegression
import numpy as np

# loading dataset

data = pd.read_csv("data/salary_Data.csv")
#training our model
x = np.array(data['experience']).reshape(-1,1)
model = LinearRegression()
model.fit(x,np.array(data['Salary']))

st.title("Salary Prediction Based of Machine Learning")

nav = st.sidebar.radio("Navigation", ['Home', 'Predict', 'Contribute'])

if nav == 'Home':
	st.image("images/Salary-Range.png", width=500)
	#displaying a table
	if st.checkbox("Show Data Table"):
		st.table(data)
	#ploting a graph
	graph = st.selectbox("What kind of Graph ?", ["Non-Interactive","Interactive"])
	val = st.slider("Filter data by years", 0,20)
	data = data.loc[data["experience"]>= val]

	if graph =='Non-Interactive':
		plt.figure(figsize = (10,5))
		plt.scatter(data["experience"],data["Salary"])
		plt.ylim(0)
		plt.xlabel("Years of Experience")
		plt.ylabel("Salary")
		plt.tight_layout()
		st.pyplot()
		st.set_option('deprecation.showPyplotGlobalUse', False)

	if graph == 'Interactive':
		layout =go.Layout(
			xaxis = dict(range=[0,20]),
			yaxis = dict(range =[0,210000])
		)
		fig = go.Figure(data=go.Scatter(x=data["experience"], y=data["Salary"], mode='markers'),layout =layout)
		st.plotly_chart(fig)

if nav == 'Predict':
	st.header("Know Your Salary")
	val = st.number_input("Enter your Experience", 0.00, 20.00, step =0.25)
	val = np.array(val).reshape(1,-1)
	pred = model.predict(val)[0]

	if st.button("Predict"):
		st.success(f"Your predicted salary is ${round(pred)}")

if nav == 'Contribute':
	st.header('Contribute to dataset')
	exp = st.number_input("Enter your Experience",0.0,20.0)
	sal = st.number_input("Enter your Salary", 0.00, 1000000.00, step=1000.0)
	if st.button("Submit"):
		to_add = {"experience":[exp], "Salary":[sal]}
		to_add = pd.DataFrame(to_add)
		to_add.to_csv("data/salary_Data.csv",mode='a', header=False, index=False)#a means append
		st.success("Submitted")

