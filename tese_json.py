import streamlit as st

data = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "Machine Learning", "Data Analysis"]
}

st.json(data, height=300) 
