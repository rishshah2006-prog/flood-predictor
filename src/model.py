import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

def train_model(gdf):

    print("Training Random Forest model...")

    features = ['IFLD_RISKS', 'CFLD_RISKS', 'SOVI_SCORE', 'RESL_SCORE']
    X = gdf[features].copy()
    y = gdf['at_risk'].copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test


def save_feature_importance(model, feature_names, output_dir):

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(importance_df['feature'], importance_df['importance'], color='#378ADD')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance — Flood Risk Model')

    for bar, val in zip(bars, importance_df['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved to {path}")


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_data
    from src.feature_engineering import build_risk_score

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gdf = load_data(
        shapefile=os.path.join(base, "data/tl_2022_us_county.shp"),
        nri_csv=os.path.join(base, "data/NRI_Table_Counties.csv"),
        precip_csv=os.path.join(base, "data/normals-monthly-precipitation.csv")
    )

    gdf = build_risk_score(gdf)

    features = ['IFLD_RISKS', 'CFLD_RISKS', 'SOVI_SCORE', 'RESL_SCORE']
    model, X_test, y_test = train_model(gdf)

    save_feature_importance(
        model,
        features,
        output_dir=os.path.join(base, "outputs")
    )
    