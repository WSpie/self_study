from flask import Flask, render_template, redirect, url_for, request, session, flash
from datetime import timedelta

app = Flask(__name__, template_folder='../templates')
app.secret_key = 'hello'
app.permanent_session_lifetime = timedelta(hours=1)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        session.permanent = True
        user = request.form['nm']
        session['user'] = user
        flash(f'You have been logged in, {user}.', 'info')
        return redirect(url_for('user'))
    else:
        if 'user' in session:
            return redirect(url_for('user'))
        
        return render_template('login.html')

@app.route('/user')
def user():
    if 'user' in session:
        user = session['user']
        flash(f'Already logged in, {user}.')
        return render_template('user.html', user=user)
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    if 'user' in session:
        user = session['user']
        flash(f'You have been logged out, {user}.', 'info')
        session.pop('user', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)