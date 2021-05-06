import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    
    X = data.drop('infectionProb', axis=1)
    y = data['infectionProb']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    
    infection=clf.predict([[98,0,18,0,0]])
    infection_probability= clf.predict_proba([[98,0,18,0,1]])
    
    # open the file where you have to store the data
    file = open('model.pkl','wb')
    # dump information to that file
    pickle.dump(clf,file)
    file.close()
    
    
    
    



