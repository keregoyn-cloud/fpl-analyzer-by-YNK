# FPL Scout AI

An internet-connected Streamlit app that combines

- Official Fantasy Premier League API data
- Upcoming fixture difficulty
- Form, minutes, ownership and expected points
- Live Google News RSS searches for injuries, rotation and lineup information
- A transparent squad-building heuristic under a £100m-style budget

## Run locally

```bash
cd fpl_scout
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## Deploy online

1. Put this folder in a GitHub repository.
2. Create a new app on Streamlit Community Cloud.
3. Select `app.py` as the entry point.
4. Set the Python requirements file to `requirements.txt`.

The app uses public endpoints and does not require an API key.

## Important limitation

The recommendation model is deliberately explainable rather than pretending to be a guaranteed prediction. It should be improved with historical backtesting, bookmaker odds, injury databases, press-conference transcripts and a proper integer optimizer before being used as an automated transfer decision-maker.
