from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier

cwd = Path.cwd()
data = pd.read_csv(cwd/'proyecto_ML'/'src'/'data'/'processed'/'train.csv')

X = data.drop(columns='target')
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(max_features=1, n_estimators=10)
rfc.fit(X_train,y_train)

print(rfc.score(X_test,y_test))

# save the model to disk

with open('new_model', 'wb') as archivo_salida:
    pickle.dump(rfc, archivo_salida)