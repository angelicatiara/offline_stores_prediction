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

        aov_50k = prediction[0][0]
        aov_50_100k = prediction[0][1]
        aov_100_200k = prediction[0][2]
        aov_200_300k = prediction[0][3]
        aov_300_500k = prediction[0][4]
        aov_500k_1mio = prediction[0][5]
        aov_1_3mio = prediction[0][6]
        aov_3_5mio = prediction[0][7]
        aov_5mio = prediction[0][8]
        monthly_rate = prediction[0][9]
        net_rev_monthly = prediction[0][10]
        aov_overall = net_rev_monthly / monthly_rate if monthly_rate != 0 else 0
        
        # Display the results
        st.write('### Predictions:')
        st.write(f'**AOV Range (orders):**')
        st.write(f'AOV <50k: {aov_50k} orders')
        st.write(f'AOV 50-100k: {aov_50_100k} orders')
        st.write(f'AOV 100-200k: {aov_100_200k} orders')
        st.write(f'AOV 200-300k: {aov_200_300k} orders')
        st.write(f'AOV 300-500k: {aov_300_500k} orders')
        st.write(f'AOV 500k-1mio: {aov_500k_1mio} orders')
        st.write(f'AOV 1-3mio: {aov_1_3mio} orders')
        st.write(f'AOV 3-5mio: {aov_3_5mio} orders')
        st.write(f'AOV >5mio: {aov_5mio} orders')

        st.write(f'**Monthly Rate (orders):** {monthly_rate}')
        st.write(f'**Monthly Net Revenue (Rupiah):** {net_rev_monthly:,.0f}')
        st.write(f'**AOV Overall:** {aov_overall:,.0f} Rupiah (calculated as net_rev_monthly / monthly_rate)')

if __name__ == "__main__":
    main()

