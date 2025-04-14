import streamlit as st
from pages import Predictions
from pages import Home

# set the page configuration
st.set_page_config(page_title="Transit Delay Prediction", page_icon="🚉", layout="centered", initial_sidebar_state="collapsed")  # Use "centered" layout
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# top menu navigation
menu = st.selectbox("Navigate", ["Home", "Predictions"], index=0)

# display the appropriate page based on the menu selection
if menu == "Home":
    Home.show()
elif menu == "Predictions":
    Predictions.show()