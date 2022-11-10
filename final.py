import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
from pylab import *
from PIL import Image

plt.style.use('seaborn')
st.title('Airbnb listing in New York City')

df= pd.read_csv('new-york.csv')

# create a multi select
neighbourhood_group_filter = st.sidebar.multiselect('Neighbourhood Group Filter', df.neighbourhood_group.unique(),df.neighbourhood_group.unique())

# create a input form
form = st.sidebar.form("hourse_form")
house_filter = form.text_input('room tpye (enter ALL to reset)', 'ALL')
form.form_submit_button("Apply")

if house_filter!='ALL':
     df=df[df.room_type == house_filter]

#文件处理
df["last_review"] = pd.to_datetime(df.last_review)
df["reviews_per_month"] = df["reviews_per_month"].fillna(df["reviews_per_month"].mean())
df.tail()
df.last_review.fillna(method="ffill", inplace=True)
for column in df.columns:
    if df[column].isnull().sum() != 0:
        df[column] = df[column].fillna(df[column].mode()[0])


#照片
image = Image.open('2.jpg')
st.image(image, caption='New York City')

st.subheader('Basic description of data characteristics')
genre = st.radio(
    "data description",
    ('All of the variables have a left-skewed distribution', 'The correlation between variables'))

if genre =='All of the variables have a left-skewed distribution':
    sns.set_palette("muted")
    f, ax = plt.subplots(figsize=(8, 6))
    subplot(2,3,1)
    sns.distplot(df['price'])
    subplot(2,3,2)
    sns.distplot(df['minimum_nights'])
    subplot(2,3,3)
    sns.distplot(df['number_of_reviews'])
    subplot(2,3,4)
    sns.distplot(df['reviews_per_month'])
    subplot(2,3,5)
    sns.distplot(df['calculated_host_listings_count'])
    subplot(2,3,6)
    sns.distplot(df['availability_365'])
    plt.tight_layout() # avoid overlap of plotsplt.draw()
    f
else:
    f, ax = plt.subplots(figsize = (20, 10))
    title = 'Correlation matrix of numerical variables'
    sns.heatmap(df.corr(), square=True, cmap="Purples")
    plt.title(title)
    plt.ioff()
    f


st.caption('The original data set can be accessed by clicking the button below')
#download
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)

st.subheader('Question 1: Distribution characteristics of different room types in different areas')

genre = st.radio(
    "Select different representations",
    ('scatter diagram', 'histogram','pie chart'))

if genre == 'scatter diagram':
    df = df[df.neighbourhood_group.isin(neighbourhood_group_filter)]
    f,ax = plt.subplots(figsize=(16,8))
    ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.room_type,palette="Purples")
    st.pyplot(f)

elif genre == 'histogram':
    df = df[df.neighbourhood_group.isin(neighbourhood_group_filter)]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, y='neighbourhood_group', hue='room_type', ax=ax, palette='Purples' )
    title = 'Count of Rooms/Room Types by each areas'
    plt.title(title)
    plt.xlabel('Count of Rooms', size=15)
    plt.ylabel('Areas', size=15)
    st.pyplot(fig)
    sns.palplot(sns.color_palette("Blues"))
elif genre == 'pie chart':
    df= pd.read_csv('new-york.csv')
    room_type=df.room_type.value_counts()
    title = 'Percentage of Room Type'
    labels=['Entire apt','Private room','Shared room']
    explode=[0.02,0.02,0.1]
    colors=['thistle','lavender','aliceblue']
    f, ax = plt.subplots(figsize=(6, 6))
    plt.pie(room_type,autopct='%1.1f%%',shadow=True,explode=explode,labels=labels,colors=colors)
    plt.legend(loc='best')
    plt.title(title)
    f


with st.expander("See explanation"):
    st.write("""
        The number of share rooms is always small.\n
        There are more houses in Brooklyn and Manhattan.\n
        The percentage of shared room is smallest
    """)


st.subheader('Question 2: The relation of price to neigbour_hood and room_type')


#箱线图
x='neighbourhood_group'
y='price'

df = df[df.neighbourhood_group.isin(neighbourhood_group_filter)]
title = 'Price per neighbourhood_group for Properties under $175'
data_filtered = df.loc[df['price'] < 175]
f, ax = plt.subplots(1,2,figsize=(15, 6))
boxplot1=sns.boxplot(x=x, y=y, data=data_filtered, notch=True, showmeans=True,
           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"}, palette='Purples',ax=ax[0])
boxplot1.set(title=title)

title = 'Price per neighbourhood_group for Properties more than $175'
data_filtered = df.loc[df['price'] > 175]
boxplot2=sns.boxplot(x=x, y=y, data=data_filtered, notch=False, showmeans=True,
           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"}, palette='Purples',ax=ax[1])
boxplot2.set(title=title)

f

with st.expander("See explanation"):
    st.write("""
       The price is divided into two groups (below and above $175).\n
       The box diagram depicts characteristics of two sets.
    """)

#柱状图
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=df, x='neighbourhood_group', y='price', palette='Purples')

plt.title("Median Price per Neighbourhood Group", size=15)
plt.xlabel("neighbourhood_group", size=15)
plt.ylabel("Price", size=15)
           
st.pyplot(fig)
with st.expander("See explanation"):
    st.write("""
        The properties have large differences in prices.
        Prices in Manhattan are generally high.
        The most expensive houses are concentrated in this area
    """)

#气球
st.balloons()

