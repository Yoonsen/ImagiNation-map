import streamlit as st

st.title("Hello World Test")
st.write("If you can see this, the app is working!")

# Add a simple interactive element
if st.button("Click me"):
    st.balloons()