from flask import Flask, render_template

app = Flask(__name__)

posts = [
    {
        "title" : "Java" ,
        "description" : "Java is a programing language",
        "create_at":  "November 26, 2025",
        "author" : "youssef Abbas"
    },
    {
        "title" : "Python" ,
        "description" : "Python is a programing language",
        "create_at":  "November 26, 2026",
        "author" : " BRAHIM OUHAMMOU"
    },
    {
        "title" : "C++/C" ,
        "description" : "C++/C is a programing language",
        "create_at":  "November 26, 2025",
        "author" : "youssef Abbas"
    }
]
@app.route("/")
def home():
    return  render_template('home.html' , posts=posts , title="Home")


@app.route("/about")
def about():
    return  render_template('about.html', title="About")

if __name__ == "__main__":
    app.run(debug=True)
