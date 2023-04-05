import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
import warnings
from flask import Flask,jsonify,request
data=pd.read_csv('iris.csv')
data.drop(columns=['Id'],inplace=True)
X=data.drop('Species',axis=1)
Y=data['Species']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
warnings.filterwarnings('ignore')
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
app=Flask('Iris')
@app.route('/hello')
def new():
    return "It is Iris Dataset"
@app.route('/<float:SepalLengthCm>/<float:SepalWidthCm>/<float:PetalLengthCm>/<float:PetalWidthCm>/')
def test(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
    p = []
    p += [SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]
    arr = np.array([p])
    predict =logmodel.predict(arr)
    if predict == ['Iris-setosa']:
        result = {'result':'It is Iris-setosa Plant'}
    elif predict == ['Iris-versicolor']:
        result = {'result':'It is Iris-versicolor Plant'}
    else:
        result = {'result':'It is Iris-virginica Plant'}
    return jsonify(result)
app.run()

