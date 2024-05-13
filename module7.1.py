from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample insights data
insights_data = [
    "Insight 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Insight 2: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Insight 3: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
]

# Function to summarize insights
def summarize_insights(insights):
    # For demonstration, return the first two insights as summary
    return insights[:2]

# Function to rank insights based on predefined criteria
def rank_insights(insights):
    # For demonstration, rank based on length of insights
    ranked_insights = sorted(insights, key=len, reverse=True)
    return ranked_insights

# Define route to expose summarization functionality
@app.route('/summarize', methods=['POST'])
def summarize():
    insights = request.json.get('insights')
    summarized_insights = summarize_insights(insights)
    return jsonify(summarized_insights)

# Define route to expose ranking functionality
@app.route('/rank', methods=['POST'])
def rank():
    insights = request.json.get('insights')
    ranked_insights = rank_insights(insights)
    return jsonify(ranked_insights)

if __name__ == '__main__':
    app.run(debug=True)
