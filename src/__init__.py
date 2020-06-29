import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

gender_submission_df = pd.read_csv("../data/final.csv")
#gender_submission_df = pd.read_csv("../data/gender_submission.csv")
test_df = pd.read_csv("../data/final.csv", na_values=["?"])
#test_df = pd.read_csv("../data/test.csv")
train_df = pd.read_csv("../data/train.csv")

#new feature the title
train_df["Title"] = train_df["Name"].str.extract("\,(\s\w+)\.")
train_predictors = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]]
train_classes = train_df[["Survived"]]


#imputer for nan in Age
simpleimputer = SimpleImputer(verbose=1, strategy="mean")
train_predictors[["Age"]] = simpleimputer.fit_transform(train_predictors[["Age"]])

#imputer for nan in Embarked
simpleimputer = SimpleImputer(verbose=1,strategy="most_frequent")
train_predictors[["Embarked"]] = simpleimputer.fit_transform(train_predictors[["Embarked"]])
train_predictors[["Title"]] = simpleimputer.fit_transform(train_predictors[["Title"]])

#encoding all categorical strin data
labelencoder = LabelEncoder()
train_predictors[["Embarked"]] = labelencoder.fit_transform(train_predictors[["Embarked"]])
train_predictors[["Sex"]] = labelencoder.fit_transform(train_predictors[["Sex"]])
train_predictors[["Title"]] = labelencoder.fit_transform(train_predictors[["Title"]])

columntransformer = ColumnTransformer(
    [
        ("PClass", StandardScaler(), [0]),
        ("Sex", OneHotEncoder(), [1]),
        ("Age", StandardScaler(), [2]),
        ("SibSp", StandardScaler(), [3]),
        ("Parch", StandardScaler(), [4]),
        ("Fare", StandardScaler(), [5]),
        ("Embarked", OneHotEncoder(), [6]),
        ("Title", StandardScaler(), [7])
     ], remainder="passthrough")

train_predictors = columntransformer.fit_transform(train_predictors)

knn_classifier = KNeighborsClassifier(n_neighbors=10, metric="minkowski", p=2)
knn_classifier.fit(train_predictors, train_classes)

randomforest_classifier = RandomForestClassifier(criterion="entropy",random_state=0,n_estimators=200)
randomforest_classifier.fit(train_predictors, train_classes)

svm_classifier = SVC(kernel="rbf", random_state=1, C=5)
svm_classifier.fit(train_predictors, train_classes)

test_df["Title"] = test_df["Name"].str.extract("\,(\s\w+)\.")
test_predictors = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]]

#imputer for nan in Age
simpleimputer = SimpleImputer(verbose=1)
test_predictors[["Age"]] = simpleimputer.fit_transform(test_predictors[["Age"]])
test_predictors[["Fare"]] = simpleimputer.fit_transform(test_predictors[["Fare"]])

#imputer for nan in Embarked
simpleimputer = SimpleImputer(verbose=1,strategy="most_frequent")
test_predictors[["Embarked"]] = simpleimputer.fit_transform(test_predictors[["Embarked"]])
test_predictors[["Title"]] = simpleimputer.fit_transform(test_predictors[["Title"]])

#encoding all categorical strin data
labelencoder = LabelEncoder()
test_predictors[["Embarked"]] = labelencoder.fit_transform(test_predictors[["Embarked"]])
test_predictors[["Sex"]] = labelencoder.fit_transform(test_predictors[["Sex"]])
test_predictors[["Title"]] = labelencoder.fit_transform(test_predictors[["Title"]])

columntransformer = ColumnTransformer(
    [
        ("PClass", StandardScaler(), [0]),
        ("Sex", OneHotEncoder(), [1]),
        ("Age", StandardScaler(), [2]),
        ("SibSp", StandardScaler(), [3]),
        ("Parch", StandardScaler(), [4]),
        ("Fare", StandardScaler(), [5]),
        ("Embarked", OneHotEncoder(), [6]),
        ("Title", StandardScaler(), [7])
     ], remainder="passthrough")

test_predictors = columntransformer.fit_transform(test_predictors)

result_knn = knn_classifier.predict(test_predictors)
result_rf = randomforest_classifier.predict(test_predictors)
result_svm = svm_classifier.predict(test_predictors)

test_classes = gender_submission_df[["Survived"]].values.tolist()

precision_knn = accuracy_score(test_classes, result_knn)
precision_randomforest = accuracy_score(test_classes, result_rf)
precision_svm = accuracy_score(test_classes, result_svm)

confusion_matrix_knn = confusion_matrix(test_classes, result_knn)
confusion_matrix_randomforest = confusion_matrix(test_classes, result_rf)
confusion_matrix_svm = confusion_matrix(test_classes, result_svm)

df_result_knn = gender_submission_df.copy()
df_result_rf = gender_submission_df.copy()
df_result_svm = gender_submission_df.copy()

df_result_knn[["Survived"]] = result_knn.reshape(-1, 1)
df_result_rf[["Survived"]] = result_rf.reshape(-1, 1)
df_result_svm[["Survived"]] = result_svm.reshape(-1, 1)



print("fim")