import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('titanic.csv')

df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

df['Embarked'].fillna('S', inplace=True)
df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])
df.drop('Embarked', axis=1, inplace=True)

def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return 38
        
        if row['Pclass'] == 2:
            return 29
        
        if row['Pclass'] == 3:
            return 25

    else:
        return row['Age']
    
df['Age'] = df.apply(fill_age, axis=1)

def fill_sex(sex):
    if sex == 'male':
        return 1
    
    else:
        return 0

df['Sex'] = df['Sex'].apply(fill_sex)

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X = df.drop('Survived', axis=1)
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifer = KNeighborsClassifier()
classifer.fit(X_train, Y_train)

pred = classifer.predict(X_test)

print(accuracy_score(Y_test, pred))