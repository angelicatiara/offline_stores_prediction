import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the saved model
best_model = joblib.load('store_model.pkl')


# Streamlit UI for user input
def main():
    st.set_page_config(
        page_title='Sociolla Offline Stores Prediction App',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    # Dropdown options for values between 0 to 5
    dropdown_options = [0, 1, 2, 3, 4, 5]

    # Inputs
    store_id = st.text_input("Store ID")

    no_of_stores_in_radius = st.selectbox("Number of Stores in Radius", options=dropdown_options)
    site_location_floors = st.selectbox("Site Location Floors", options=dropdown_options)
    footfall_avg = st.selectbox("Average Footfall", options=dropdown_options)
    category_tenancy_mix = st.selectbox("Category Tenancy Mix", options=dropdown_options)
    customer_profile = st.selectbox("Customer Profile Score", options=dropdown_options)
    store_size = st.number_input("Store Size (sq.ft)", value=0, min_value=0)  # Numeric input
    soco_members = st.selectbox("Number of SOCO Members", options=dropdown_options)
    shoppers_p12m = st.selectbox("Shoppers in the Past 12 Months", options=dropdown_options)
    population = st.selectbox("Population", options=dropdown_options)

    # Display Inputs Summary
    st.write("### Inputs Summary")
    st.write(f"Number of Stores in Radius: {no_of_stores_in_radius}")
    st.write(f"Site Location Floors: {site_location_floors}")
    st.write(f"Average Footfall: {footfall_avg}")
    st.write(f"Category Tenancy Mix: {category_tenancy_mix}")
    st.write(f"Customer Profile Score: {customer_profile}")
    st.write(f"Store Size (sq.ft): {store_size}")
    st.write(f"Number of SOCO Members: {soco_members}")
    st.write(f"Shoppers in the Past 12 Months: {shoppers_p12m}")
    st.write(f"Population: {population}")


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
