from flask import Flask, redirect, url_for, request
from flask import render_template
app = Flask(__name__, template_folder='webpages')


if __name__ == '__main__':
    app.run(host='192.168.100.9',port='60',debug=True)
    
    
    
    
    
    