import streamlit as st

# Get an integer input from the user
user_integer = st.number_input("Enter an integer", value=None, step=1)

# Display the entered integer
st.write(f"You entered: {user_integer}")
