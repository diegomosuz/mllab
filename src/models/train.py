# src/models/train.py
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--C', type=float, default=1.0)
args = parser.parse_args()
np.random.seed(args.seed)


# Cargar/crear datos
# (reutilizar generador semana 2)
from baseline import X, y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=args.seed)
clf=LogisticRegression(C=args.C,max_iter=1000).fit(X_train,y_train)
metrics={
'auc': float(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
}
print(json.dumps(metrics, indent=2))