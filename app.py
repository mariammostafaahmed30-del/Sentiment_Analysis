import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="AI Solutions Dashboard", layout="wide")

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        # Sentiment Model
        sent_model = pickle.load(open('trained_model.sav', 'rb'))
        vect = pickle.load(open('vectorizer.sav', 'rb'))
        return sent_model, vect
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None, None

sentiment_model, vectorizer = load_models()

# --- Sidebar ---
with st.sidebar:
    st.title("🎛️ Dashboard")
    st.write("Welcome to the Data Analysis System")
    
    selected_option = st.radio(
        "Select Service:",
        ['📊 Bulk Sentiment Analysis', '💰 Price Prediction']
    )
    st.markdown("---")
    st.info("Graduation Project: Maryam")

# =========================================================
# Part 1: Bulk Sentiment Analysis
# =========================================================
if selected_option == '📊 Bulk Sentiment Analysis':
    st.header("📂 Customer Sentiment Analysis from Files")
    st.markdown("Upload a CSV file containing customer reviews for bulk analysis.")

    # 1. Upload File
    uploaded_file = st.file_uploader("Upload CSV file here", type=['csv'])

    if uploaded_file is not None:
        # Read File
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1') # or utf-8
            st.success("File uploaded successfully! ✅")
            
            # Show first 5 rows
            st.write("Data Preview:", df.head())

            # 2. Select Text Column
            text_column = st.selectbox("Select the column containing the review text:", df.columns)

            if st.button('🚀 Start Analysis'):
                with st.spinner('Analyzing customer reviews...'):
                    # Transform text
                    texts = df[text_column].astype(str)
                    transformed_texts = vectorizer.transform(texts)
                    
                    # Predict
                    predictions = sentiment_model.predict(transformed_texts)
                    
                    # Add results to DataFrame
                    df['Sentiment_Prediction'] = predictions
                    df['Sentiment_Label'] = df['Sentiment_Prediction'].apply(lambda x: 'Positive 😃' if x == 1 else 'Negative 😡')
                    
                    # 3. Show Results & Stats
                    st.divider()
                    st.subheader("📊 Analysis Report")
                    
                    # Count
                    counts = df['Sentiment_Label'].value_counts()
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Positive Reviews Count", counts.get('Positive 😃', 0))
                    col2.metric("Negative Reviews Count", counts.get('Negative 😡', 0))
                    
                    # Bar Chart
                    st.bar_chart(counts)

                    # Final Table
                    st.write("Data after analysis:")
                    st.dataframe(df[[text_column, 'Sentiment_Label']])
                    
                    # Download Button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Final Report (CSV)",
                        data=csv,
                        file_name='sentiment_analysis_results.csv',
                        mime='text/csv',
                    )

        except Exception as e:
            st.error(f"Error reading the file: {e}")

# =========================================================
# Part 2: Price Prediction
# =========================================================
elif selected_option == '💰 Price Prediction':
    st.header("🏠 Smart Price Prediction System")
    st.markdown("Enter unit specifications to get the predicted price.")

    # ⚠️ Note: Adjust inputs based on your actual price model features
    
    with st.form("price_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            input1 = st.number_input("Area (sqm)", min_value=50, value=100)
            input3 = st.number_input("Number of Bathrooms", min_value=1, value=1)
            
        with col2:
            input2 = st.number_input("Number of Rooms", min_value=1, value=3)
            input4 = st.number_input("Floor Level", min_value=0, value=1)
            
        # Submit Button
        submit_val = st.form_submit_button("Predict Price Now")
        
        if submit_val:
            try:
                price_model = pickle.load(open('price_model.sav', 'rb'))
                
                # Prepare features
                features = np.array([[input1, input2, input3, input4]])
                
                # Predict
                predicted_price = price_model.predict(features)
                
                st.success(f"💰 Predicted Price: {predicted_price[0]:,.2f} EGP")
            except:
                st.info("⚠️ (This is a demo example because the price model file is not currently loaded)")
                st.success(f"Based on inputs: Area {input1} and Rooms {input2}.. Approximate predicted price: 1,500,000 EGP (Example)")
