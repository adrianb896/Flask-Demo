from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/areaUnder/')
def areaUnder():
    return render_template('areaUnder.html')

@app.route('/fs/')
def fs():
    return render_template('fs.html')

@app.route('/hist/')
def hist():
    return render_template('hist.html')

@app.route('/treeplot/')
def treeplot():
    return render_template('treeplot.html')

@app.route('/roc/')
def roc():
    return render_template('roc.html')

if __name__ == '__main__':
    app.run(debug=True)