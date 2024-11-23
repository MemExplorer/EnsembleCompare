from flask import Flask, redirect, url_for, request
from flask import render_template
app = Flask(__name__, template_folder='webpages')

if __name__ == '__main__':
    app.run(debug=True)