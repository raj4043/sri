from flask import Flask, jsonify, request, render_template
import pandas as pd

app = Flask(__name__)

# Placeholder for structured data
structured_data = []

# Load structured data from CSV file
df = pd.read_csv('sample.csv')

# Convert data to JSON format
for _, row in df.iterrows():
    structured_data.append({'id': row['id'], 'name': row['name'], 'age': row['age'], 'email': row['email']})

# Basic semantic layer definition
semantic_layer = {
    'entities': ['name', 'age', 'email'],
    'relationships': {
        'user': ['name', 'age', 'email']
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query_semantic', methods=['POST'])
def process_query_semantic():
    user_query = request.json['query']
    response = interpret_query(user_query)
    return jsonify({'response': response})

def interpret_query(query):
    # Check if query contains any entity
    for entity in semantic_layer['entities']:
        if entity in query:
            return f"Query interpreted: You mentioned {entity}."

    # Check if query contains any relationship
    for relationship, attributes in semantic_layer['relationships'].items():
        for attribute in attributes:
            if attribute in query:
                return f"Query interpreted: You mentioned {attribute} related to {relationship}."

    return "Query interpreted: No relevant entities or relationships found in the query."

if __name__ == '__main__':
    app.run(debug=True)
