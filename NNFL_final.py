# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:42:33 2021

@author: Prathamesh and Abhiram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.linear_model import LinearRegression

#%%

df = pd.read_csv("C:/Users/Prathamesh/Desktop/NNFL_IA/chennai_house_price_prediction.csv")
 #includes the categorical variables too buat not necessary to use
df1=pd.read_csv("C:/Users/Prathamesh/Desktop/NNFL_IA/prediction_data.csv")
df.isnull().sum()



temp = pd.DataFrame(index=df.columns)
temp['datatypes']= df.dtypes
temp['null_count']=df.isnull().sum()
temp['unique']=df.nunique()


df['INT_SQFT'].plot.hist(bins=50)
plt.xlabel('INT_SQFT')
df['SALES_PRICE'].plot.hist(bins=50)
df['N_BEDROOM'].value_counts().plot(kind='bar')
#df['N_BEDROOM'].value_counts()/len(df)*100

df.plot.scatter('INT_SQFT','SALES_PRICE')
df['N_BEDROOM'].mode()

df.drop_duplicates(inplace=True)



Missing_Bathroom = df.loc[df['N_BATHROOM'].isnull()==True]
Missing_Bedroom = df.loc[df['N_BEDROOM'].isnull()==True]

df['N_BEDROOM'].mode()
df['N_BEDROOM'].fillna(value=(df['N_BEDROOM'].mode()[0]),inplace=True)

df.isnull().sum()

for i in range(0,len(df)):
    if pd.isnull(df['N_BATHROOM'][i])==True:
        if (df['N_BEDROOM'][i]==1.0):
            df['N_BATHROOM'][i] = 1.0
        else:
            df['N_BATHROOM'][i] = 2.0




df.loc[df['QS_OVERALL'].isnull()==True]


Quality_score = pd.DataFrame({'QS_ROOMS':df['QS_ROOMS'],'QS_BATHROOM':df['QS_BATHROOM'],
                   'QS_BEDROOM':df['QS_BEDROOM'],'QS_OVERALL':df['QS_OVERALL']})


QS_AVG = (Quality_score['QS_ROOMS']+Quality_score['QS_BATHROOM']+Quality_score['QS_BEDROOM'])/3

temp3 = pd.concat([Quality_score,QS_AVG],axis=1)
temp3.rename(columns={0:'QS_AVG'},inplace=True)

for i in range(0,len(df)):
    if pd.isnull(df['QS_OVERALL'][i])==True:
        df['QS_OVERALL'][i] = (df['QS_ROOMS'][i] + df['QS_BEDROOM'][i]+df['QS_BATHROOM'][i])/3


CorrectNames = ['AREA','SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL','STREET','MZZONE']



df['AREA'].replace({'Chrompt':'Chrompet','Chormpet':'Chrompet','Chrmpet':'Chrompet','TNagar':'T Nagar','Ana Nagar':'Anna Nagar','Ann Nagar':'Anna Nagar','Karapakam':'Karapakkam','Velchery':'Velachery','Adyr':'Adyar','KKNagar':'KK Nagar'},inplace=True)
#df['AREA'].value_counts()

df['BUILDTYPE'].replace({'Comercial':'Commercial','Other':'Others'},inplace=True)

df['SALE_COND'].replace({'Adj Land':'AdjLand','Ab Normal':'AbNormal','Partiall':'Partial','PartiaLl':'Partial'},inplace=True)
#df['SALE_COND'].value_counts()

df['PARK_FACIL'].replace({'Noo':'No'},inplace=True)
#df['PARK_FACIL'].value_counts()

df['UTILITY_AVAIL'].replace({'All Pub':'AllPub'},inplace=True)
#df['UTILITY_AVAIL'].value_counts()

df['STREET'].replace({'Pavd':'Paved','NoAccess':'No Access'},inplace=True)
#df['STREET'].value_counts()



#%%

sn.scatterplot(x='INT_SQFT',y='SALES_PRICE',data=df)

df=df.drop(['PRT_ID'],axis=1)

sn.scatterplot(x='MZZONE',y='SALES_PRICE',data=df)
df.groupby('MZZONE').SALES_PRICE.mean().plot(kind='bar')
correlation = df.corr()

sn.heatmap(df.corr(),cmap="YlGnBu",annot=True)

sn.distplot(df[df['BUILDTYPE']=='Commercial']['SALES_PRICE'],color='g',label='Commercial')
sn.distplot(df[df['PARK_FACIL']=='Yes']['SALES_PRICE'],color='r')
sn.distplot(df[df['PARK_FACIL']=='No']['SALES_PRICE'],color='r')

df['SALES_PRICE'][df['PARK_FACIL']=='No'].plot.hist(bins=100)

df.groupby('SALE_COND').SALES_PRICE.mean()


df = df.astype({'N_BATHROOM':'int64','N_BEDROOM':'int64'})

#%%
#print(df["MZZONE"].unique())
def user_input_features():
    st.title("Chennai house price prediction")
    st.header("Just fill in the below details and know the estimated house price!!!")
    
    st.subheader("ENTER LOCATION as per below numbering")
    st.write(" 4 : Karapakkam")
    st.write("1:Anna Nagar")
    st.write("0:Adyar")
    st.write("6:Velachery")
    st.write("2:Chrompet")
    st.write("3:KK Nagar")
    st.write("5:T Nagar")
    
    
    AREA = st.radio("Select Location: ", (4,1,0,6,2,3,5))
    
    
    
    
   
    INT_SQFT= st.slider("Select the area ", df['INT_SQFT'].min(), df['INT_SQFT'].max())
    st.text('Selected: {}'.format(INT_SQFT))
    
    #distance from main road
    DIST_MAINROAD= st.slider("Distance from main road ", df['DIST_MAINROAD'].min(), df['DIST_MAINROAD'].max())
    st.text('Selected: {}'.format(DIST_MAINROAD))
    
   
    N_ROOM = st.selectbox("Number of rooms ",
                     [2,3,4,5,6])
    
    
    
    N_BEDROOM= st.selectbox("Number of bedrooms",
                            [1,2,3,4])
  
    
    
    N_BATHROOM= st.selectbox("Number of bathrooms",
                             [1,2])
    
    
    st.subheader('Enter the sale condition as per below numbering')
    st.write('0:AbNormal')
    st.write('2:Family')
    st.write('4:Partial')
    st.write('1:AdjLand')
    st.write('3:Normal')
    SALE_COND = st.radio("Sale Condition: ", (0,2,4,1,3))
    

 
    
    st.subheader('Enter the requirement of Parking facility')
    st.write('1:YES')
    st.write('0:NO')
    Parking = st.radio("Select Parking facility: ", (1,0))
    

    
    
    st.subheader('Enter the required build type')
    st.write('0:Commercial')
    st.write('2:House')
    st.write('1:Others')
    Build_type = st.selectbox("Build type ",
                     [0,2,1])
    
    
    
    
    st.subheader('Enter the required Utilities')
    st.write('0:ALLPUB ')
    st.write('1:ELO')
    st.write('3:NoSewr')
    st.write('2:NoSeWa')
    utility = st.radio("Select Utility: ", (0,1,2,3))
    

   
    
    st.subheader('Enter Street ')
    st.write('2:Paved')
    st.write('0:Gravel')
    st.write('1:No Access')
    Street = st.selectbox("Select Street ",
                     [2,0,1])

    
    st.subheader("Select MZZONE")
    st.write('0:A')
    st.write('3:RH')
    st.write('4:RL')
    st.write('2:I')
    st.write('1:C')
    st.write('5:RM')
    Zone = st.radio("Select Utility: ", (0,3,4,2,1,5))
    




    Qs_rooms= st.slider("Quality rating of rooms", df['QS_ROOMS'].min(), df['QS_ROOMS'].max())
    st.text('Selected: {}'.format(Qs_rooms))

    Qs_bathroom= st.slider("Quality rating of bathrooms ", df['QS_BATHROOM'].min(), df['QS_BATHROOM'].max())
    st.text('Selected: {}'.format(Qs_bathroom))


    Qs_bedroom= st.slider("Quality rating of bedrooms ", df['QS_BEDROOM'].min(), df['QS_BEDROOM'].max())
    st.text('Selected: {}'.format(Qs_bedroom))

    Qs_overall= st.slider("Overall quality rating ", df['QS_OVERALL'].min(), df['QS_OVERALL'].max())
    st.text('Selected: {}'.format(Qs_overall))

    Commision=st.slider("Charged Commision ", df['COMMIS'].min(), df['COMMIS'].max())
    st.text('Selected: {}'.format(Commision))
    
    
    data={
        'AREA':AREA ,
        'INT_SQFT':INT_SQFT,
        'DIST_MAINROAD':DIST_MAINROAD,
        'N_ROOM':N_ROOM,
        'N_BEDROOM':N_BEDROOM,
        'N_BATHROOM':N_BATHROOM,
        'SALE_COND':SALE_COND,
        'PARK_FACIL':Parking,
        'BUILDTYPE':Build_type,
        'UTILITY_AVAIL':utility,
        'STREET':Street,
        'MZZONE':Zone,
        'QS_ROOMS':Qs_rooms,
        'QS_BEDROOM':Qs_bedroom,
        'QS_BATHROOM':Qs_bathroom,
        'QS_OVERALL':Qs_overall,
        'COMMIS':Commision
        }
    web_features=pd.DataFrame(data,index=[0])
    return web_features

#%%
prediction_data=user_input_features() #dataframe
prediction_data_scaled=MinMaxScaler()
prediction_data=prediction_data_scaled.fit_transform(prediction_data)





enc=LabelEncoder()
df.iloc[:,0]=enc.fit_transform(df.iloc[:,0])
df.iloc[:,6]=enc.fit_transform(df.iloc[:,6])
df.iloc[:,7]=enc.fit_transform(df.iloc[:,7])
df.iloc[:,8]=enc.fit_transform(df.iloc[:,8])
df.iloc[:,9]=enc.fit_transform(df.iloc[:,9])
df.iloc[:,10]=enc.fit_transform(df.iloc[:,10])
df.iloc[:,11]=enc.fit_transform(df.iloc[:,11])

#%%
st.header('THE PREDICTED HOUSE PRICE IS ')
st.write("Just a couple of seconds!!!")
X = df.drop('SALES_PRICE',axis=1).values
y = df['SALES_PRICE'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)



model=Sequential()  #initialising ANN
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))

model.add(Dense(1))



model.compile(optimizer='adam',
              loss='mae',
              
              )
model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=200,
          )
model.summary()
#%%
loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12,8))

#%%
y_pred = model.predict(prediction_data)


st.write(y_pred)
#%%








    
    
    

    
    

    
    
    

#%%
# Making user interface

    
    

    






