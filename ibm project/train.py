import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def train_model():
    """
    Loads data, preprocesses, trains a RandomForestClassifier, and saves model artifacts.
    """
    print("Starting model training process...")

    # 1. Load dataset
    df = pd.read_csv('students_mental_health_survey.csv')
    print("Dataset loaded successfully.")

    # 2. Use only selected features
    selected_features = [
        'Age', 'Gender', 'CGPA', 'Depression_Score', 'Anxiety_Score', 'Sleep_Quality',
        'Physical_Activity', 'Diet_Quality', 'Social_Support', 'Financial_Stress',
        'Extracurricular_Involvement', 'Semester_Credit_Load'
    ]
    df = df[selected_features].dropna()

    # 3. Label encoding for categorical column(s) – here, only "Gender"
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    joblib.dump(encoders, 'label_encoders.joblib')
    print("Label encoders saved.")

    # 4. Compute stress_score and classify into 3 levels
    df['stress_score'] = df['Depression_Score'] + df['Anxiety_Score'] + df['Sleep_Quality']
    bins = [df['stress_score'].min() - 1, 3.33, 6.66, df['stress_score'].max()]
    labels = [0, 1, 2]
    df['stress_level'] = pd.cut(df['stress_score'], bins=bins, labels=labels).astype(int)
    print("Stress levels calculated.")

    # 5. Features & target
    X = df.drop(['stress_score', 'stress_level'], axis=1)
    y = df['stress_level']
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.joblib')
    print("Feature names saved.")

    # 6. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved.")

    # 7. SMOTE for class balance
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)
    print(f"Dataset balanced using SMOTE. New shape: {X_res.shape}")

    # 8. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # 9. Train RandomForestClassifier with GridSearchCV
    param_grid = {
        'n_estimators': [100],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [1, 2]
    }
    print("Starting GridSearchCV...")
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    joblib.dump(clf.best_estimator_, 'stress_model.joblib')
    print("Model trained and saved as 'stress_model.joblib'.")

    # 10. Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 11. Feature importance plot
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, clf.best_estimator_.feature_importances_, color='skyblue')
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("static/feature_importance.png")
    plt.close()
    print("Feature importance plot saved to static/feature_importance.png.")

if __name__ == '__main__':
    train_model()