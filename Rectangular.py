# -*- coding: utf-8 -*-
"""
Spyder Editor


This is a temporary script file.
"""

import streamlit as st
import pandas as pd 
#import shap 
#import matplotlib.pyplot as plt

st.write("""
 # **GUI for Rectangular CFSST Columns** """
        )
st.write("""
 ### This app predicts the axial capacity of rectangular CFSST columns. """       
       )

st.write('---')

df=pd.read_csv("Recatngular dataset .csv")

x=df.drop(["N_Test"],axis=1)
y=df["N_Test"]


st.sidebar.header("User Input Parameters")

def user_input_features():
    Grade_SS=st.sidebar.slider("  Grade_SS",1,23,3)
    B=st.sidebar.slider("  B",40,250,124)
    H=st.sidebar.slider("H",49,250,120)
    t=st.sidebar.slider("  t",1,12,3)
    L=st.sidebar.slider("  L",150,850,377)
    LB=st.sidebar.slider(" L/B",1.5,7.46,3.16)
    Eo=st.sidebar.slider(" Eo",18000,217000,199730)
    f=st.sidebar.slider("  f_0.2",258,598,433)
    fu=st.sidebar.slider(" fu",409,961,674)
    n=st.sidebar.slider("  n",3.0,12.4,5.97)
    fc_cyl=st.sidebar.slider("f c_cyl",21.5,114.6,48.14)
    data={"Grade_SS":Grade_SS,"B":B,"H":H,"t":t,"L":L,"L/B":LB,"Eo":Eo,"f_0.2":f,"fu":fu,"n":n,"fc_cyl":fc_cyl}
    features=pd.DataFrame(data,index=[0])
    
    return features

data_df=user_input_features()

st.subheader("Specified Input Parameters")
st.write(data_df)

#st.write("---")

#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


from xgboost import XGBRegressor
model=XGBRegressor(n_estimators=800,learning_rate=0.1)
model.fit(x,y)

prediction=model.predict(data_df)


st.subheader("Prediction of Axial Capacity of Column")
st.write(prediction)
st.write('---')



#explainer=shap.TreeExplainer(model)
#shap_values=explainer.shap_values(x)

#st.header("Feature Importance")
#plt.title("Feature Importance Based on Shap Values")
#shap.summary_plot(shap_values,x)
#st.pyplot()
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.write("---")

#plt.title("Relative Feature Importance")
#shap.summary_plot(shap_values,x,plot_type="bar")
#st.pyplot()




