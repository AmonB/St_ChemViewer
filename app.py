import streamlit as st

home_page = st.Page("pages/home_page.py", title="Home", icon="🏠")
analysis_page = st.Page("pages/gaussian.py", title="Analysis", icon="🔍")
file_browser_page = st.Page("pages/file_browser.py", title="File Browser", icon="📁")
view_file_page = st.Page("pages/file_reader.py", title="File Reader", icon="📄️")

pg = st.navigation([home_page, analysis_page, file_browser_page, view_file_page])

pg.run()

