import requests

url = "http://127.0.0.1:8000/predict"

tests = [
    "Free markets and low taxes drive economic growth and protect individual liberty",
    "Both parties should compromise on balanced bipartisan legislation",
    "Workers unions and wealth redistribution are essential for social justice",
    "The second amendment must be protected. Government regulation kills jobs.",
    "Climate change demands immediate government action and corporate regulation",
]

for t in tests:
    r = requests.post(url, json={"text": t})
    print(f"  {r.json()['prediction']:>6}  ->  {t[:75]}")
