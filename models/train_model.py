import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

def load_dataset():

    df = pd.read_csv("resume_dataset.csv")

    df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})

    df = df.select_dtypes(include=["int64","float64"])

    X = df.drop("Attrition",axis=1)
    y = df["Attrition"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    return X_train,X_test,y_train,y_test


def train_logistic():

    X_train,X_test,y_train,y_test = load_dataset()

    model = LogisticRegression(max_iter=500,class_weight="balanced")

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    return model,pred,y_test,X_train,y_train


def train_knn():

    X_train,X_test,y_train,y_test = load_dataset()

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    return model,pred,y_test,X_train,y_train


def train_linear():

    X_train,X_test,y_train,y_test = load_dataset()

    model = LinearRegression()

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    pred = (pred>0.35).astype(int)

    return model,pred,y_test,X_train,y_train