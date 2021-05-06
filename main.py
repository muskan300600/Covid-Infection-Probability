from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=['GET','POST'])
def hello_world():
    
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodyPain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreathe = int(myDict['diffBreathe'])
        
        
    
        input_features = [fever,bodyPain,age,runnyNose,diffBreathe]
        infection_probability= clf.predict_proba([input_features])[0][1]
        print(infection_probability)
        return render_template('show.html',inf=round(infection_probability*100))
    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True) # diplays error on the screen
    
