from os import name
from flask import Flask, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return "Home page <h1>Home page</h1>"

@app.route('/<name>')
def user(name):
    return f"Hello {name}!"

# redirect to home page if the user is admin
@app.route('/admin')
def admin():
    return redirect(url_for('home')) # 'home' is the name of the function
    # return redirect(url_for('user', name='Admin!')) # 'user' is the function and 'name' is the input variable

if __name__ == '__main__':
    app.run()