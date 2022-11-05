import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Q1
plt.style.use('seaborn')
st.title('Airbnb listing in New York City')

df = pd.read_csv('new-york.csv')

# create a multi select
neighbourhood_group_filter = st.sidebar.multiselect('Neighbourhood Group Filter', df.neighbourhood_group.unique(),df.neighbourhood_group.unique())

# create a input form
form = st.sidebar.form("hourse_form")
house_filter = form.text_input('room tpye (enter ALL to reset)', 'ALL')
form.form_submit_button("Apply")

if house_filter!= 'ALL':
     df=df[df.room_type == house_filter]

# 图1
df = df[df.neighbourhood_group.isin(neighbourhood_group_filter)]

f,ax = plt.subplots(figsize = (16,8))
ax = sns.scatterplot(y = df.latitude, x = df.longitude, hue = df.room_type, palette = "coolwarm")
st.pyplot(f)


#图2
fig, ax = plt.subplots(figsize = (8, 6))
sns.countplot(data = df, y = 'neighbourhood_group', hue = 'room_type', ax = ax)

title = 'Count of Rooms/Room Types by each areas'
plt.title(title)
plt.xlabel('Count of Rooms', size = 15)
plt.ylabel('Areas', size = 15)
st.pyplot(fig)

# Q2
# 箱线图
x = 'neighbourhood_group'
y = 'price'


title = 'Price per neighbourhood_group for Properties under $175'
data_filtered = df.loc[df['price'] < 175]
f, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = x, y = y, data = data_filtered, notch = True, showmeans = True,
           meanprops = {"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
plt.title(title)
plt.ioff()
f
title = 'Price per neighbourhood_group for Properties more than $175'
data_filtered = df.loc[df['price'] > 175]
f, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = x, y = y, data = data_filtered, notch = True, showmeans = True,
           meanprops = {"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
plt.title(title)
plt.ioff()
f



#柱状图
fig, ax = plt.subplots(figsize = (8, 6))
sns.barplot(data = df, x = 'neighbourhood_group', y = 'price')

plt.title("Median Price per Neighbourhood Group", size = 15)
plt.xlabel("neighbourhood_group", size = 15)
plt.ylabel("Price", size = 15)
           
st.pyplot(fig)


