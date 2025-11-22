This repository contains a Streamlit-based stock prediction app that uses a local CSV (`sp500.csv`) as the single data source.


## Run locally


1. Create a venv and activate it


```bash
python -m venv .venv
source .venv/bin/activate # on Windows use .venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
