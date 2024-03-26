import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='Salary and Skills', page_icon='💰')
st.title('🤖 What are you worth?')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app les you input your skills and get an estimate of your potential salary. Try adding new skills to see how that would add to your worth.')



#import the model
 with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

clf2.predict(X[0:1])

#add inout data
df_input_skills = pd.read_csv('inputskills.csv')

  
  checkbox_labels = ['SQL', 'Python', 'Excel', 'Power BI', 'Tableau', 'SAS', 'Azure', 'Snowflake', 'AWS', 'Spark', 'Looker', 'Qlik']
# Create a dictionary to store the checkbox states
checkbox_states = {}

# Create checkboxes for each label and store their states in the dictionary
for label in checkbox_labels:
    checkbox_states[label] = st.checkbox(label)



# Define a function to make predictions based on the selected checkboxes
def predict():
    # Convert the selected checkboxes to the input format required by the model
    input_data = [[1 if checkbox_states[label] else 0 for label in checkbox_labels]]

    # Convert input_data to a DataFrame with the same structure as df_input_skills
    input_df = pd.DataFrame(input_data, columns=checkbox_labels)

    # Convert boolean values to integers
    input_df = input_df.astype(int)

    # Find the matching row in df_input_skills
    matching_row = df_input_skills[df_input_skills.eq(input_df).all(axis=1)]

    # If a matching row is found, display the output
    if not matching_row.empty:
        matching_index = matching_row.index[0]
        final_output = df_output_percent.loc[matching_index]
        st.write("Job Opening Count:", final_output['count'])
        st.write("Percentage of available openings:", f"{round(final_output['percentage'],2)} %")
    else:
        st.write("No matching row found in the input data.")

# Add a button to trigger the prediction
if st.button("Predict"):
    predict()


   
   
