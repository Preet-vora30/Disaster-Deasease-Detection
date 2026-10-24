import os
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score

from feature_extractors import extract_all_features
from utils import preprocess_image

def create_geotag_bins(latitudes, longitudes, bins=10):
    """
    Bin coordinates into 10 categories and label encode them.
    """
    lat_bins = pd.cut(latitudes, bins=bins, labels=False)
    lon_bins = pd.cut(longitudes, bins=bins, labels=False)
    
    geo_categories = [f"{lat}_{lon}" for lat, lon in zip(lat_bins, lon_bins)]
    le = LabelEncoder()
    encoded = le.fit_transform(geo_categories)
    return encoded.reshape(-1, 1), le

def load_data(metadata_path, datasets_dir):
    df = pd.read_csv(metadata_path)
    X_img = []
    y = df['damage_severity'].values
    
    print("Extracting features (this might take a few moments)...")
    for _, row in df.iterrows():
        post_img_path = os.path.join(datasets_dir, row['post_image'])
        _, img_array = preprocess_image(post_img_path)
        if img_array is not None:
            features = extract_all_features(img_array)
            X_img.append(features)
        else:
            X_img.append(np.zeros(129))
            
    X_img = np.array(X_img)
    
    # Process geotags
    geo_encoded, le = create_geotag_bins(df['latitude'], df['longitude'], bins=10)
    
    # Create final DataFrame to easily work with ColumnTransformer
    feature_cols = [f"img_f_{i}" for i in range(X_img.shape[1])]
    df_features = pd.DataFrame(X_img, columns=feature_cols)
    
    df_features['geo_encoded'] = geo_encoded.flatten()
    
    return df_features, y, feature_cols, le

def build_svm_pipeline(feature_cols):
    # Pipeline for image features
    img_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)), 
        ('selector', SelectKBest(score_func=mutual_info_classif, k=30))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer([
        ('img_features', img_pipeline, feature_cols),
        ('geo_pass', 'passthrough', ['geo_encoded'])
    ])
    
    # Full Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(kernel='rbf', probability=True))
    ])
    
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 0.01]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    return grid_search

def build_ensemble_pipeline(feature_cols):
    img_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)), 
        ('selector', SelectKBest(score_func=mutual_info_classif, k=30))
    ])
    
    preprocessor = ColumnTransformer([
        ('img_features', img_pipeline, feature_cols),
        ('geo_pass', 'passthrough', ['geo_encoded'])
    ])
    
    svm_clf = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(kernel='rbf', probability=True, C=10, gamma='scale'))
    ])
    
    rf_clf = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    ensemble = VotingClassifier(
        estimators=[('svm', svm_clf), ('rf', rf_clf)],
        voting='soft'
    )
    return ensemble

def train_and_save_model():
    metadata_path = "datasets/metadata.csv"
    if not os.path.exists(metadata_path):
        print("Dataset not found. Please run data_generator.py first.")
        return None, None
        
    X_df, y, feature_cols, le = load_data(metadata_path, 'datasets')
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    
    print("Training SVM pipeline with GridSearchCV...")
    grid_search = build_svm_pipeline(feature_cols)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    
    print(f"SVM Validation Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/svm_trained.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    print("Model saved to models/svm_trained.pkl")
    return best_model, (acc, f1)

if __name__ == "__main__":
    train_and_save_model()
