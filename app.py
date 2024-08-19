from flask import Flask, render_template, request, json
from model.nn import LogisticRegression, Net
import torch , pickle
import numpy as np
import torch.nn.functional as F

model = LogisticRegression()
model.load_state_dict(torch.load('weight/rg_best.pt'))
model = Net()
model.load_state_dict(torch.load('weight/best.pt'))


# with open('LG.pkl', 'rb') as f:
#     clf2 = pickle.load(f)
#     f.close()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("base.html")
@app.route("/predict", methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    age = (int(request.form['age']) - 32) / (72 - 32)
    education = (int(request.form['education']) - 1.) / (4. - 1.) 
    currentSmoker = int(request.form['currentSmoker'])
    cigsPerDay = int(request.form['cigsPerDay']) / 70.0
    BPMeds = int(request.form['BPMeds'])
    prevalentStroke = int(request.form['prevalentStroke'])
    prevalentHyp = int(request.form['prevalentHyp'])
    diabetes = int(request.form['diabetes'])
    totChol = (int(request.form['totChol']) - 113.) / (600 - 113)
    sysBP = (int(request.form['sysBP']) - 83.5) / (295 - 83.5)
    diaBP = (int(request.form['diaBP']) - 48.) / (142.5 - 48)
    BMI = (int(request.form['BMI']) - 15.54) / (56.8 - 15.54)
    heartRate = (int(request.form['heartRate']) - 44. ) / (143.-44.)
    glucose = (int(request.form['glucose'])-40) / (394-40) 
    # x = torch.asarray([[gender, age, education, 
    #                             currentSmoker, cigsPerDay, BPMeds, 
    #                             prevalentStroke, prevalentHyp, diabetes,
    #                             totChol, sysBP, diaBP,
    #                             BMI, heartRate, glucose]]).float()
    x = np.array([[gender, age, education, 
                                currentSmoker, cigsPerDay, BPMeds, 
                                prevalentStroke, prevalentHyp, diabetes,
                                totChol, sysBP, diaBP,
                                BMI, heartRate, glucose]])
    
    # y = clf2.predict(x)
    # print(y)
    # if y < 0.5:
    #     return json.dumps({'HEART DISEASE PREDICT': 'No'})
    # return json.dumps({'HEART DISEASE PREDICT': 'Yes'})

    # pred = model(torch.from_numpy(x).float())
    # print(pred)
    # is_dis = torch.where(pred > 0.5, 1, 0) 
    # if is_dis[0][0] == 0:
    #     return json.dumps({'HEART DISEASE PREDICT': 'No'})
    # return json.dumps({'HEART DISEASE PREDICT': 'Yes'})


    pred = F.softmax(model(torch.from_numpy(x).float()))
    is_dis = torch.where(pred > 0.5, 1, 0) 
    if is_dis[0][0] == 0:
        return json.dumps({'HEART DISEASE PREDICT': 'No, Heart disease: '+ str(round(float(pred[0][0])*100, 4))+' , and No heart disesas: '+ str(round(float(pred[0][1])*100, 4))})
    return json.dumps({'HEART DISEASE PREDICT': 'Yes, Heart disease: '+ str(round(float(pred[0][0])*100, 4))+' , and No heart disesas: '+ str(round(float(pred[0][1])*100, 4))})

if __name__ == "__main__":
    
    app.run(debug=True)