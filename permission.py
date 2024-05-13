from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Placeholder for sensitive data
sensitive_data = {
    'user1': {'name': 'John Doe', 'email': 'john@example.com', 'phone': '123-456-7890'},
    'user2': {'name': 'Jane Smith', 'email': 'jane@example.com', 'phone': '987-654-3210'}
}


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Simulating basic authentication
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html', error='')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'logged_in' in session:
        # Simulate data masking
        masked_data = {key: {k: '*****' if k != 'name' else v for k, v in value.items()} for key, value in
                       sensitive_data.items()}
        return render_template('dashboard.html', data=masked_data)
    else:
        return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
