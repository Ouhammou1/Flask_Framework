from flask_wtf import FlaskForm
from wtforms import StringField , PasswordField , SubmitField
from wtforms.validators import  DataRequired   , Length , Email , EqualTo

class RegisterForm(FlaskForm):
    username = StringField(label="Username" , validators=[DataRequired() , Length(min=2, max=10)])
    email = StringField(label="Email" , validators=[DataRequired() , Email()])
    password = StringField(label="Password" , validators=[DataRequired() , Length(min=6)])
    confirm_password = PasswordField(label="Confirm Password" , validators=[DataRequired() , EqualTo('password')])
    submit = SubmitField(label="Sign Up")

    # def validate_on_submit():

