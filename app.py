from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Placeholder for user queries and responses
query_responses = {
    'What are the sales trends?': 'Sales have been increasing steadily.',
    'What is the status of Product A?': 'Product A is out of stock.',
    'Any updates on the marketing campaign?': 'New marketing campaign launched.',
    'default': "I'm sorry, I couldn't understand your query."
}

@app.route('/')
def chat():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    query = request.json['query']
    response = query_responses.get(query, query_responses['default'])
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
