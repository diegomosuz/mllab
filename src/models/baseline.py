# src/models/baseline.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

print('Generando...')
rng = np.random.default_rng(42)
N=5000
X = pd.DataFrame({
'age': rng.integers(18,80,N),
'tenure': rng.integers(1,60,N),
'spend': rng.normal(100,30,N).clip(0),
'is_promo': rng.integers(0,2,N)
})
print(X)
# Regla latente
p = 1/(1+np.exp(-( -3 + 0.03*X['age'] - 0.05*X['tenure'] + 0.01*X['spend'] + 0.8*X['is_promo'] )))
y = (rng.random(N) < p).astype(int)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print('Entrenando...')
clf = LogisticRegression(max_iter=1000).fit(X_train,y_train)
print("AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))