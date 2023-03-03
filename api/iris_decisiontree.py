import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Iris veri setini yükleyelim
iris_df = pd.read_csv('iris.csv')

# X ve y verilerini ayıralım
X = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

# Modeli eğitelim
model = DecisionTreeClassifier()
model.fit(X, y)

# Oluşturduğumuz modeli model1.pkl adlı dosyaya kaydedelim
with open('DecisionTree.pkl', 'wb') as f:
    pickle.dump(model, f)
