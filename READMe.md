# StChemViewer

---

A web-app for molecular visualization. Current version is build for parsing the output file of Gaussian software package.
It is designed to simplify your daily research and data analysis.


## StChemViewer features

---

- Interactive 3D molecular visualization
- Optimization plots (Energy, RMS gradient norm, Maximum internal force,
RMS internal force, Maximum internal displacement, RMS internal displacement)
- Optimization trajectory
- Vibration animation
- Thermochemistry table
- Export XYZ and energies (suitable for splitting IRC)
- With port forwarding, no need to download files from the server 

## Quickstart

---

- Start web-app``streamlit run app.py``
- Click Analysis
- Input path to Gaussian files
- Choose a file
- Click Decipher

![img.png](img.png)
![img_1.png](img_1.png)

## File support

---

- Gaussian input file (.gjf, .com)
- Gaussian output file (.log, .out)

## Requirement

---

- python 3.8+
- streamlit > 1.8
- py3Dmol, pandas, plotly
- numpy < 2.0.0
- Browser (Chrome, Edge, Firefox）


## Possible issues

---

- If you meet `_ssl import error`, you need to update openssl >= 1.1.1 and rebuild
your python ``./configure --with-openssl=/path/to/openssl``
