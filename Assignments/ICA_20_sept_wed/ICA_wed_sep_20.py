import plotly.express as px
import streamlit as st
import seaborn as sns
import pandas as pd

st.subheader("Wisconsin Breast Cancer Diagnostic dataset")
df = pd.read_csv('./data.csv')

my_dict = {"Columns":["concavity_mean","radius_mean","concave_points_mean","compactness_mean","area_mean","fractal_dimension_mean","concavity_worst", "radius_worst","concave_points_worst","compactness_worst","area_worst","fractal_dimension_worst"],"Description":["Mean concavity of the tumor cells","Mean radius of the tumor cells","Mean number of concave portions of the contour of the tumor cells","Mean compactness of the tumor cells","Mean area of the tumor cells","Mean coastline approximation of the tumor cells","Worst(Most severe) concavity of the tumor cells","Worst(Largest) radius of the tumor cells","Worst(Most Severe) number of concave portions of the contour of the tumor cells","Worst(Most Severe) compactness of the tumor cells","Worst(Largest) area of the tumor cells","Worst(Largest) coastline approximation of the tumor cells"]}
description_dataset = pd.DataFrame.from_dict(my_dict)

st.write("**Plot Mean and Worst value from dataset**")

xData = st.selectbox(
    'Choose X value for the plot',
    ('concavity_mean','radius_mean','concave points_mean','compactness_mean','area_mean','fractal_dimension_mean','diagnosis'))
yData = st.selectbox(
    'Choose Y value for the plot',
    ('concavity_worst','radius_worst','concave points_worst','compactness_worst','area_worst','fractal_dimension_worst','diagnosis'))

choosen_plot = st.selectbox('Choose the plot of your preference',('relplot','displot','catplot'))

if choosen_plot == "relplot":  
    plot = sns.relplot(data=df, x=xData, y=yData,hue="diagnosis")

else:
    if choosen_plot == "displot":
        plot = sns.catplot(data=df, x=xData, y=yData)
    else:
        plot = sns.displot(data=df, x=xData, binwidth=1)


st.pyplot(plot.fig)

st.table(description_dataset)