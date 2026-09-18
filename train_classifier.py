import pickle 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

data_dict = pickle.load(open('data.pickle', 'rb'))

X = np.asarray(data_dict['data'])
y = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,stratify=y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:")
#print(classification_report(y_test, y_pred))
#print("Confusion Matrix:")
#print(confusion_matrix(y_test, y_pred))

f=open('model.p','wb')
pickle.dump(classifier,f)
f.close()