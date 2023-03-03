import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Iris veri setini yükleyelim
iris_df = pd.read_csv('iris.csv')

# X ve y verilerini ayıralım
X = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

# Modeli eğitelim
model = LogisticRegression()
model.fit(X, y)

# Oluşturduğumuz modeli model1.pkl adlı dosyaya kaydedelim
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
