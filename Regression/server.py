from flask import Flask,request
import pickle
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from flask_ngrok import run_with_ngrok



lasso_model=Lasso()

app = Flask(__name__)
#run_with_ngrok(app)
lasso_model=pickle.load(open('model.pkl','rb'))
print('model.pkl loaded')
prediction = lasso_model.predict([[10,10,3,3,84,26,94,5,8,51,6.7,0]])

@app.route("/admin",methods=['GET'])
def hello_world():
    X=int(request.args.get('X',''))
    prediction = lasso_model.predict([[X,10,3,3,84,26,94,5,8,51,6.7,0]])
    print(prediction[0])
    return 'result ' + str(prediction[0])

if __name__ == '__main__':
   app.run(host="localhost", port=int("50018"))

