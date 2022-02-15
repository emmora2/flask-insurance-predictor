from flask import Flask, render_template, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, SelectField, FloatField
import pandas as pd
import pickle
import os
import numpy as np
from wtforms.validators import DataRequired
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

model = pickle.load(open("regression_random_forest_model_.pkl", "rb"))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
picFolder = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = picFolder


class PredictionForm(FlaskForm):
    sex = SelectField(u'What is your Gender?', choices=[('male', 'Male'), ('female', 'Female')],
                      validators=[DataRequired()])
    smoker = SelectField(u'Do you smoke?', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    region = SelectField(u'Which region do you live in?',
                         choices=[('northeast', 'Northeast'), ('northwest', 'NorthWest'), ('southeast', 'Southeast'),
                                  ('southwest', 'SouthWest')], validators=[DataRequired()])
    age = IntegerField("What is your age?", validators=[DataRequired()])
    children = IntegerField("How many children do you have?", validators=[DataRequired()])
    bmi = FloatField("What is your BMI?", validators=[DataRequired()])
    submit = SubmitField("Submit")


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = False
    form = PredictionForm()
    if form.is_submitted():
        sex = form.sex.data
        smoker = form.smoker.data
        region = form.region.data
        age = form.age.data
        if (int(age)) < 16 or int(age) > 99:
            errors = True
            flash("Please type an age between 16 and 99.")
        children = form.children.data
        bmi = form.bmi.data
        if (int(bmi)) < 0 or int(bmi) > 50:
            errors = True
            flash("Please type a bmi between 0 and 50")
        healthCost = str(round(predictCost(predictionList(sex, smoker, region, age, children, bmi))[0], 2))
        accuracy = getModelAccuracy()
        if not errors:
            return render_template('prediction.html', healthCost=healthCost, accuracy=accuracy)
    return render_template('home.html', form=form)


@app.route('/data')
def data():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'bargraph.png')
    scatterPlot = os.path.join(app.config['UPLOAD_FOLDER'], 'scatterplot6.png')
    insurance_list = get_insurance_list()
    return render_template('data.html', pic1=pic1, insurance_list=insurance_list, scatterPlot=scatterPlot)


def validateList(age, bmi, children):
    message_list = []
    if age < 0 or age > 99:
        message_list.append('Please input a correct Age!')
    if bmi < 3 or bmi > 50:
        message_list.append('Please input an appropiate bmi!')
    if children < 0 or children > 10:
        message_list.append('Please input appropriate number of children between 1 and 10!')


# prediction model takes in an array such as, [[1,0,1,0,0,1,0,0,46,19.95,2]]
# format is sex female, sex male, smoker no, smoker yes, region northeast , region northwest,
# region southeast, region southwest
# format our categorical inputs into numerical
def predictionList(sex, smoker, region, age, bmi, children):
    dataList = []
    if sex == 'male':
        dataList.append(0)
        dataList.append(1)
    elif sex == 'female':
        dataList.append(1)
        dataList.append(0)

    if smoker == 'yes':
        dataList.append(0)
        dataList.append(1)
    elif smoker == 'no':
        dataList.append(1)
        dataList.append(0)

    if region == 'northeast':
        dataList.append(1)
        dataList.append(0)
        dataList.append(0)
        dataList.append(0)
    elif region == 'northwest':
        dataList.append(0)
        dataList.append(1)
        dataList.append(0)
        dataList.append(0)
    elif region == 'southeast':
        dataList.append(0)
        dataList.append(0)
        dataList.append(1)
        dataList.append(0)
    elif region == 'southwest':
        dataList.append(0)
        dataList.append(0)
        dataList.append(1)
        dataList.append(0)
    dataList.append(age)
    dataList.append(bmi)
    dataList.append(children)
    return dataList


# create a List of data used for our machine learning model
def get_insurance_list():
    file = open("insurance.csv", "r")
    rowList = []
    for row in file:
        newRow = row.split(",")
        rowList.append(newRow)
    # remove the first row consisting of col names
    rowList.pop(0)

    return rowList


def predictCost(list):
    newList = []
    newList.append(list)
    return model.predict(newList)


def getModelAccuracy():
    insurance = pd.read_csv("insurance.csv")
    # y = prediction , X = columns to predict on
    X = insurance.drop("charges", axis=1)
    y = insurance["charges"]
    # get categorical features that require encoding
    categorical_features = ["sex", "smoker", "region"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
    transformed_X = transformer.fit_transform(X)
    pd.DataFrame(transformed_X)
    model = RandomForestRegressor(max_depth=5, random_state=0)
    model.get_params()
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


if __name__ == '__main__':
    app.run(port=5000)
