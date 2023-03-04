import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split

# Iris veri setini yükleyelim
iris_df = pd.read_csv('IRIS.csv')

X_train, X_test, y_train, y_test = train_test_split(
    iris_df.iloc[:, :-1], iris_df.iloc[:, -1], test_size=0.2, random_state=42)

# Modeli eğitelim
model = LogisticRegression()
model.fit(X_train, y_train)

# Oluşturduğumuz modeli model1.pkl adlı dosyaya kaydedelim
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
