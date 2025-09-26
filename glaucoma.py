# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv("glaucoma.csv")

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.columns

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.describe()

# %%
df.nunique()

# %%
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []

    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].nunique() < 30:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']: # here 64 is the 64 bit Operating System
            if df[column].nunique() < 30:
                discrete_features.append(column)
            else:
                continuous_features.append(column)

    return categorical_features, non_categorical_features, discrete_features, continuous_features

# %%
categorical, non_categorical, discrete, continuous = classify_features(df)

# %%
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

# %%
# Fill missing values in 'Sleep Disorder' with the mode
df['Medical History'] = df['Medical History'].fillna(df['Medical History'].mode()[0])

# Optional: check if any missing values remain
print("Missing values after filling:", df['Medical History'].isnull().sum())


# %%
# Fill missing values in 'Sleep Disorder' with the mode
df['Medication Usage'] = df['Medication Usage'].fillna(df['Medication Usage'].mode()[0])

# Optional: check if any missing values remain
print("Missing values after filling:", df['Medication Usage'].isnull().sum())


# %%
df.isnull().sum()

# %%
df = df.drop(['Patient ID'],axis = 1)

# %%
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

# %%
# Create a copy of the DataFrame with only the selected columns
df_selected = df.copy()

# %%
# Encode categorical features using one-hot encoding
df_encoded = pd.get_dummies(df_selected, columns=['Gender', 'Visual Acuity Measurements', 'Family History', 'Medical History', 'Cataract Status', 'Angle Closure Status', 'Diagnosis'])

# %%
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
df[['Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry']]=scaler.fit_transform(df[['Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry']])

# %%


# %%


# %%
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df_encoded['Glaucoma Type'] = label_encoder.fit_transform(df_encoded['Glaucoma Type'])

# %%
df_encoded.columns

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

X = df_encoded.drop('Glaucoma Type', axis=1)
y = df_encoded['Glaucoma Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("Before Oversampling:", y_train.value_counts())

# %%
df

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# %%
logreg = LogisticRegression()

# %%
logreg.fit(X_train_resampled, y_train_resampled)

# %%
y_pred = logreg.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# %%
from sklearn import svm
svc = svm.SVC()
svc.fit(X_train_resampled, y_train_resampled)

# %%
y_pred = svc.predict(X_test)

# %%
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# %%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%
import pickle
with open('logreg_model.pkl', 'wb') as file:
    pickle.dump(logreg, file)
print("Model saved as logreg_model.pkl")


