from flask import Flask


app = Flask(__name__)

@app.route('/')
def page():
    return '<h1> hellow </h1>'


if "__main__" == __name__:
    app.run(debug=True , port= 5689)