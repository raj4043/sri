import requests

# Define insights data
insights = [
    "Insight 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Insight 2: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Insight 3: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
]

# Send POST request to summarize endpoint
response = requests.post('http://127.0.0.1:5000/summarize', json={'insights': insights})
print("Summarized insights:", response.json())

# Send POST request to rank endpoint
response = requests.post('http://127.0.0.1:5000/rank', json={'insights': insights})
print("Ranked insights:", response.json())
