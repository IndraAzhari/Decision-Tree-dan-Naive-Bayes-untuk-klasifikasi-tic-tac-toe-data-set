from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# load dataset ke dalam Pandas Dataframe

dat = pd.read_csv('tic_tac_toe.csv')
cols = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']
x = dat[cols]
y = dat['Class']

#Pisah, Test_size 20% dan training set 80%

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Membuat classifier Gaussian Naive Bayes

naive = GaussianNB()

# Melatih classifier dengan data training

naive = naive.fit(x_train, y_train)

# Melakukan prediksi terhadap data testing dan probabilitasnya

y_predict = naive.predict(x_test)
naive.predict_proba(x_test)

# tampilin akurasi klasifikasi

print("Akurasi klasifikasi Machine Learning : " + str(metrics.accuracy_score(y_test, y_predict))+"\n")

# tampilin confusion matrix

print(metrics.confusion_matrix(y_test, y_predict))

# tampilin precision,recall,fmeasure

print(metrics.classification_report(y_test, y_predict))
