from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    # Retrieve user message from the form
    user_message = request.form['message']

    # Placeholder response from the chatbot
    bot_response = "This is a placeholder response."

    return bot_response


if __name__ == '__main__':
    app.run(debug=True)
