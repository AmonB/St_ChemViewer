import streamlit as st


st.set_page_config(page_title="Home page", layout="wide")

# welcome page
st.title("🔬 StChemViewer for your daily work")
st.markdown("""
<div style='background-color: #e8f4fd; padding: 10px; border-radius: 10px; border-left: 0px solid #2196f3;'>
A web-ui app for molecular visualization based on Streamlit framework. Current version specializes for
Gaussian software package.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Current Features")

st.markdown("""
<div style='text-align: center; padding: 40px 20px;'>
<div style='display: flex; justify-content: center; gap: 30px; margin-bottom: 10px;'>
    <div style='text-align: center;'>
        <div style='font-size: 36px; margin-bottom: 10px;'>🔎</div>
        <h4>Molecular Visualization</h4>
    </div>
    <div style='text-align: center;'>
        <div style='font-size: 36px; margin-bottom: 10px;'>🎬</div>
        <h4>Trajectory Visualization</h4>
    </div>
    <div style='text-align: center;'>
        <div style='font-size: 36px; margin-bottom: 10px;'>📈</div>
        <h4>Optimization Monitor</h4>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Structure modification is not supported, limited by `py3Dmol`. "
            "However, it's still enough to take a look at the geometry optimization process.")
st.markdown("Combined with port forwarding, you don't need to download the files to check the results if you suffer "
            "from the poor network of your campus.")


with st.sidebar.expander("🚀 Let's start", expanded=True):
    st.markdown("""
    1. Click Analysis 
    2. Input path to Gaussian files
    3. Choose a file
    4. Click Decipher to analyze data
    5. Enjoy it! 
    """)
st.sidebar.markdown('</div>', unsafe_allow_html=True)
