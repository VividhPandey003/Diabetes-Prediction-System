# Importing the necessary Python modules.
import streamlit as st

# Import necessary functions from web_functions
from web_functions import load_data, train_model
from Tabs import home, data, predict, visualise

# Configure the app
st.set_page_config(
    page_title='Diabetes Prediction',
    page_icon='🥯',
    layout='wide',
    initial_sidebar_state='auto'
)

# Dictionary for pages
Tabs = {
    "Home": home,
    "Data Info": data,
    "Prediction": predict,
    "Visualisation": visualise
}

# Create a sidebar
# Add title to sidebar
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Loading the dataset.
df, X, y = load_data()

# Train model
model, score = train_model(X, y)

# Call the app function of selected page to run
if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, X, y, model)  # Pass the trained model to the prediction or visualization page
elif page == "Data Info":
    Tabs[page].app(df)
else:
    Tabs[page].app()
