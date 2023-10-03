import pandas as pd
import numpy as np
import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title='The Machine Learning App',
                   layout='wide')
st.write("""
# Financial Prediction APP

This app predicts if **A person has a bank account** !

Data obtained from the [The Financial_inclusion_dataset](https://drive.google.com/file/d/1FrFTfUln67599LTm2uMTSqM8DjqpAaKL/view).
""")
st.title('Financial Inclusion prediction')
st.sidebar.header('User Input Parameters')

def user_input_features():
    with st.sidebar.header(' Choose parametres'):
        Country = st.sidebar.selectbox('Select the country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
        Year= st.sidebar.slider('Pick the year',2000,2023,2015)
        Location_type= st.sidebar.selectbox('Select the location type',['Rural','Urban'])
        Cell_access= st.sidebar.radio('Cell phone access',['Yes','No'])
        Household_size=st.sidebar.slider('Household size',0,25,10)
        Age=st.sidebar.number_input('Enter the age of the respondent',0,100,20)
        Gender=st.sidebar.radio('Gender of the respondent',['Male','Female'])
        Relationship=st.sidebar.selectbox('The relationship with the respondent',[ 'Head of Household','Spouse', 'Other relative', 'Child', 'Parent',
         'Other non-relatives'])
        Status=st.sidebar.selectbox('Marital status',['Married/Living together', 'Widowed', 'Single/Never Married',
       'Divorced/Seperated', 'Dont know'])
        Education=st.sidebar.selectbox('Education Level',['Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA'])
        Job= st.sidebar.selectbox('Job type',['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])
    data = {'country':Country,
            'year': Year,
            'location_type': Location_type,
            'cellphone_access':Cell_access,
            'household_size':Household_size,
            'age_of_respondent':Age,
            'gender_of_respondent':Gender,
            'relationship_with_head':Relationship,
            'marital_status':Status,
            'education_level':Education,
            'job_type':Job}
    features = pd.DataFrame(data, index=[0])
    return features

data=pd.read_csv(r'C:\Users\ayabe\PycharmProjects\streamlit\Financial_inclusion_dataset.csv')
data=data.drop('uniqueid',axis=1)
st.subheader('Exemple of the dataset')
st.write(data.head())
#df,model= user_input_features()
df=user_input_features()
#clf=get_model(model)
st.subheader('User Input parameters')
st.write(df)

datanew= pd.concat([data, df], ignore_index=True)


def encode_data(col):
    lb=LabelEncoder()
    encoded= lb.fit_transform(col)
    return encoded

#print(data.info())
#print(data.isna().sum())
#encoding all the categorical columns
object_col=['country','bank_account','location_type','cellphone_access','gender_of_respondent',
            'relationship_with_head','marital_status','education_level','job_type']
for colm in object_col:
    datanew[colm]=encode_data(datanew[colm])
    #print(colm)
df=datanew.iloc[-1,:]
df=pd.DataFrame(df)
df= df.T

data = datanew.drop(datanew.index[-1])
y=data['bank_account']
x=data.drop('bank_account',axis=1)
df=df.drop('bank_account',axis=1)


model=RandomForestClassifier()
params = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
#grid=GridSearchCV(model,params)
#grid.fit(x,y)
#model=grid.best_estimator_
model=RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=10)
model.fit(x,y)
st.subheader('The model using GridSearchCV is ')
st.write(model)
pred=model.predict(df)
st.subheader('The prediction is:')
if pred==0:
    prediction="No The Person doesn't have a bank account "
else: prediction="Yes The Person has a bank account "
st.write(prediction)
st.subheader('The probability of each class is :')
prob=model.predict_proba(df)
prob=pd.DataFrame(prob,columns=['The probability of NO ','The probability of YES '])
st.write(prob)
