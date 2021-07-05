from flask import Flask, redirect, url_for, render_template
import os

def locate_temp(file_type):
    file_prefix = os.path.abspath(__file__).split('/')[-1].split('.')[0]
    file_path = file_prefix + f'.{file_type}'
    # print(file_path)
    return file_path

# be aware of the flask file structure
# -app
# --templates // where html must be in
# --static // where js and css
# --.py files
# --other packages
app = Flask(__name__, template_folder='../templates')


@app.route('/<name>')
def home(name):
    return render_template(locate_temp('html'),
                            content=name, lst=['ha', 'hi', 'ho'])



if __name__ == '__main__':
    app.run(debug=True)