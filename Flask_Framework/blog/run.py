from flask import Flask, render_template , flash , redirect , url_for
from forms import RegisterForm



app = Flask(__name__)
app.config["SECRET_KEY"] = '0df5fb2d0a0cac7febd98d647b741316'



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

@app.route("/register" , methods=['GET' , 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        flash(f"{form.username.data} Registered Successfully" , "success")
        return redirect(url_for('home'))

    return render_template('register.html', form=form  , title="Register")



if __name__ == "__main__":
    app.run(debug=True)


# vedio 4 