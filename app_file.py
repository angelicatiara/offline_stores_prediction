import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the saved model
best_model = joblib.load('store_model.pkl')

# Define preprocessing function for inference
def preprocess_data(input_data):
    # Add any necessary preprocessing steps here
    # For simplicity, assuming input data is ready to be used for prediction
    return input_data

# Streamlit UI for user input
def main():
    # Set page configuration for better UX
    st.set_page_config(
        page_title='Sociolla Offline Store Predictions',
        page_icon='\ud83c\udfe2',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Title and description
    st.title('\ud83c\udfe2 Sociolla Offline Store Prediction App')
    st.write('Fill in the store details below to predict AOV ranges, monthly rate, and monthly net revenue.')

    # Split the page into two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.header('Store Details')
        store_id = st.text_input('Store ID', value='Store001', help='Unique identifier for the store.')
        average_order_value = st.number_input('Average Order Value (AOV)', min_value=0.0, value=150000.0, step=5000.0, help='Average order value in Rupiah.')
        monthly_rate = st.number_input('Monthly Rate', min_value=0.0, value=500.0, step=10.0, help='Number of transactions per month.')

    with col2:
        st.header('Additional Metrics')
        monthly_foot_traffic = st.number_input('Monthly Foot Traffic', min_value=0, value=1000, step=50, help='Number of visitors per month.')
        location_type = st.selectbox('Location Type', options=['Mall', 'Standalone', 'Other'], index=0, help='Type of store location.')
        store_size = st.selectbox('Store Size', options=['Small', 'Medium', 'Large'], index=1, help='Physical size of the store.')

    # Prediction Button
    if st.button('Predict Metrics'):
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'Store_ID': [store_id],
            'AOV': [average_order_value],
            'Monthly_Rate': [monthly_rate],
            'Foot_Traffic': [monthly_foot_traffic],
            'Location_Type': [location_type],
            'Store_Size': [store_size]
        })

        # Preprocess the data
        processed_data = preprocess_data(input_data)

        # Make predictions
        prediction = best_model.predict(processed_data)

        # Display the results
        st.write('### Predictions:')
        st.write(f'**AOV Range:** {prediction[0][0]:,.0f} - {prediction[0][1]:,.0f} Rupiah')
        st.write(f'**Monthly Rate:** {prediction[0][2]:,.0f} Transactions')
        st.write(f'**Monthly Net Revenue:** {prediction[0][3]:,.0f} Rupiah')

if __name__ == "__main__":
    main()
