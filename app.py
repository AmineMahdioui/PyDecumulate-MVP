import streamlit as st

st.set_page_config(page_title="Deployment Test", page_icon="🚀", layout="centered")

st.title("🚀 Deployment Test App")
st.caption("Simple Streamlit interface for deployment validation")

name = st.text_input("Your name", placeholder="Type a name...")
value = st.slider("Pick a number", min_value=0, max_value=100, value=25)
run = st.button("Run test")

if run:
    if name.strip():
        st.success(f"Hello, {name}! ✅")
    else:
        st.warning("Please enter a name.")

    st.info(f"Selected value: {value}")

st.divider()
st.write("If you can see this app online, deployment is working.")
