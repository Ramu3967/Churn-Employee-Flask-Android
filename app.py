import pickle

from flask import Flask, render_template, request, jsonify

from constants import msg_exit_low, msg_exit_high

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    res = preprocess()
    if res == 0:
        res = msg_exit_low
    else:
        res = msg_exit_high
    print(res)
    return render_template('index.html', the_result=res)


@app.route('/android_predict', methods=['POST'])
def android_predict():
    req_json = request.json
    print("Values that were posted :\n", req_json)
    credit_score = req_json.get('CreditScore')
    gender = req_json.get('gender')
    age = req_json.get('Age')
    tenure = req_json.get('Tenure')
    balance = req_json.get('Balance')
    numProd = req_json.get('NumProd')
    hasCC = req_json.get('HasCC')
    activeMember = req_json.get('ActiveMember')
    estimatedSalary = req_json.get('EstimatedSalary')
    geo = req_json.get('Geography')

    test_array = [credit_score, gender, age, tenure, balance, numProd, hasCC, activeMember, estimatedSalary]
    if geo == 'fr':
        test_array.append(1)
        test_array.append(0)
        test_array.append(0)
    elif geo == 'ge':
        test_array.append(0)
        test_array.append(1)
        test_array.append(0)
    elif geo == 'sp':
        test_array.append(0)
        test_array.append(0)
        test_array.append(1)
    else:
        print("invalid country")

    prediction = model.predict([test_array])

    pred_result = str(prediction[0])
    if pred_result == "0":
        msg = msg_exit_low
    else:
        msg = msg_exit_high
    print(msg)
    return jsonify({"Prediction": msg})


def preprocess():
    test_array = list(request.form.values())
    print("values received are - ", test_array)
    last = test_array[len(test_array) - 1]
    if last == 'Submit': test_array.pop()
    country = test_array.pop()
    if country == 'fr':
        test_array.append(1)
        test_array.append(0)
        test_array.append(0)
    elif country == 'ge':
        test_array.append(0)
        test_array.append(1)
        test_array.append(0)
    elif country == 'sp':
        test_array.append(0)
        test_array.append(0)
        test_array.append(1)
    else:
        print("invalid country")
    print('data point being tested is ', test_array)
    res = model.predict([test_array])
    return res


def get(key): return request.form.get(key, 0.20202)


if __name__ == '__main__':
    app.run(debug=True)
