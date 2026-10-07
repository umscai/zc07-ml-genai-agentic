import pandas as pd
from config import settings
import utils as ut
import logging
import joblib
logger = logging.getLogger(__name__)

clf_model = joblib.load(settings.MODEL_DIR/'XGB_CTGAN_model.pkl')
df_columns = pd.read_csv(settings.DATA_DIR / settings.DATASET_FILE_CLEANED, nrows=0).columns
feature_cols = [c for c in df_columns if c != "Class"]
#----------------------------------------------------------------------------------------
def get_fraud_prob_from_ml_classifier(features: dict[str,int|float]) -> float:
    """
    Get fraud_prob from trained XGBoost classifier
    Args:
        - features: features dict contains [V1 to V28, Amount, and Hour/2]
    Returns:
        fraud probability
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        X_test = pd.DataFrame([features])
        # 1. Validate that we have  30 feature columns (V1 to V28, Amount, Hour/2)
        # Convert df.columns to a list and compare directly against feature_cols
        if list(X_test.columns) != feature_cols:
            raise ValueError(
                f"Input DataFrame columns do not match expected feature columns in exact order.\n"
                f"Expected ({len(feature_cols)}): {feature_cols}\n"
                f"Received ({len(X_test.columns)}): {list(X_test.columns)}"
            )
        fraud_prob = clf_model.predict_proba(X_test)[:, 1].item()
        return fraud_prob
    except Exception as e:
        logger.error(f"Error in detect_anomaly: {e}")
        raise

