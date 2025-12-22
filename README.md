# StChemViewer

A web-app for molecular visualization. Current version is build for parsing the output file of Gaussian software package.
It is designed to simplify your daily research and data analysis.


## StChemViewer features

- Interactive 3D molecular visualization
- Optimization plots (Energy, RMS gradient norm, Maximum internal force,
RMS internal force, Maximum internal displacement, RMS internal displacement)
- Optimization trajectory
- Vibration animation
- Thermochemistry table
- Export XYZ and energies (suitable for splitting IRC)
- With port forwarding, no need to download files from the server 

## Quickstart

- Start web-app ``streamlit run app.py``
- Click Analysis
- Input path to Gaussian files
- Choose a file
- Click Decipher

<img width="1379" height="591" alt="image" src="https://github.com/user-attachments/assets/4063a4f6-2e76-4264-ba07-b8abc05fc8ea" />
<img width="944" height="711" alt="image" src="https://github.com/user-attachments/assets/f2b193c4-e3dc-40b6-83f5-1a6dba97d998" />


## File support

- Gaussian input file (.gjf, .com)
- Gaussian output file (.log, .out)

## Requirement

- python 3.8+
- streamlit > 1.8
- py3Dmol, pandas, plotly
- numpy < 2.0.0
- Browser (Chrome, Edge, Firefox）

## Possible issues

- If you meet `_ssl import error`, you need to update openssl >= 1.1.1 and rebuild
your python ``./configure --with-openssl=/path/to/openssl``


