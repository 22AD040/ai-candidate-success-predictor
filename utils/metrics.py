from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model,X_train,y_train,y_test,pred):

    accuracy = accuracy_score(y_test,pred)

    cm = confusion_matrix(y_test,pred)

    report = classification_report(y_test,pred)

    mse = mean_squared_error(y_test,pred)

    f1 = f1_score(y_test,pred)

    cv = cross_val_score(model,X_train,y_train,cv=5)

    return accuracy,cm,report,mse,f1,cv