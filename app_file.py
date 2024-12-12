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
        # Prepare input data as a DataFrame with standardized column names (lowercase)
        input_data = pd.DataFrame({
            'no_of_stores_in_radius': [no_of_stores_in_radius],
            'site_location_floors': [site_location_floors],
            'footfall_avg': [footfall_avg],
            'category_tenancy_mix': [category_tenancy_mix],
            'customer_profile': [customer_profile],
            'store_size': [store_size],
            'soco_members': [soco_members],
            'shoppers_p12m': [shoppers_p12m],
            'population': [population],
        })

        # Ensure the column names are in lowercase (if that's how the model was trained)
        input_data.columns = input_data.columns.str.lower()

        # Make predictions
        prediction = best_model.predict(input_data)

        # Display the results
        st.write('### Predictions:')
        st.write(f'**AOV Range:** {prediction[0][0]:,.0f} - {prediction[0][1]:,.0f} Rupiah')
        st.write(f'**Monthly Rate:** {prediction[0][2]:,.0f} Transactions')
        st.write(f'**Monthly Net Revenue:** {prediction[0][3]:,.0f} Rupiah')

if __name__ == "__main__":
    main()

