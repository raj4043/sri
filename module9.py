# app.py

from flask import Flask, render_template

app = Flask(__name__)

# Route to render the first dashboard
@app.route('/dashboard')
def dashboard():
    # Fetch dynamic data
    data = fetch_data()
    return render_template('dashboard.html', data=data)

# Route to render the second dashboard
@app.route('/dashboard2')
def dashboard2():
    # Fetch dynamic data
    data = fetch_data()
    return render_template('dashboard2.html', data=data)

# Function to fetch dynamic data


# Function to fetch dynamic data from CSV file
def fetch_data():
    try:
        with open('static/data/sample.csv', 'r') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None


if __name__ == '__main__':
    app.run(debug=True)
