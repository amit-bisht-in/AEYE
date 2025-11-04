# =========================================================================
#
#                    model  "Ancestral EYE" 
#
# =========================================================================
#
#   ### METHODOLOGY 
#
#   This is the final, monolithic, and correct script.
#
#    My original, core hypothesis:
#   A "LightGBM" (LGBM) model trained only on the two
#   data streams we believe have real signal:
#
#   1.  The Metadata (Date, State, NDVI, Height)
#   2.  "Ancestral Eye" (the 6 GLCM Texture Features)
#
#   This is the real experiment. It is stable, it is not
#   "too complex" to converge and it is not "too simple" to learn wrong pattern.
#   It is the  test of my hypothesis.
#
# =========================================================================



# =========================================================================
#                           IMPORTS 
# =========================================================================


print("Protocol: Loading all required libraries...")
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import gc
import warnings
import joblib  # For saving the final model

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

from tqdm.auto import tqdm
tqdm.pandas()

warnings.filterwarnings('ignore')
print("Protocol: Imports Loaded.")

# =========================================================================
#                  EXPERIMENT PARAMETERS & ENVIRONMENT
# =========================================================================
class C:
    # --- 1. SOTA Protocol Parameters ---
    N_FOLDS = 5
    SEED = 42
    
    # --- 2. Path Configuration (Kaggle Environment) ---
    # (CRITICAL) You *must* verify this path is correct in your notebook
    BASE_PATH = '/kaggle/input/csiro-biomass'
    
    # --- 3. Debug Mode ---
    # We are DONE with debugging. This protocol is stable.
    # We will run the full, real experiment.
    DEBUG_MODE = False
    DEBUG_SAMPLES = 100
    
    # --- 4. GLCM (Ancestral Eye) Config ---
    IMG_SIZE = 224 # We still need to resize for GLCM consistency

print(f"Protocol: Parameters Loaded. N_FOLDS={C.N_FOLDS}, DEBUG_MODE={C.DEBUG_MODE}")

# =========================================================================
#                "ANCESTRAL EYE" (GLCM) HELPER FUNCTION 
# =========================================================================
def get_texture_features(image_path):
    """
    (Our "Ancestral Eye")
    Loads an image, resizes, converts to grayscale, and calculates
    GLCM Haralick texture features.
    """
    try:
        # Load and resize the image for consistency
        img = imread(image_path)
        img_resized = resize(img, (C.IMG_SIZE, C.IMG_SIZE), anti_aliasing=True)
        
        # Convert to grayscale
        img_gray = rgb2gray(img_resized)
        
        # Quantize to 8-bit integers (0-255)
        img_gray = (img_gray * 255).astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        
        # Extract the 6 core texture properties
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                 'correlation', 'ASM']
        features = np.array([graycoprops(glcm, prop)[0, 0] for prop in props])
        return features
    
    except Exception as e:
        # If an image is corrupted, return zeros
        return np.zeros(6)

print("Protocol: 'Ancestral Eye' (get_texture_features) function defined.")

# =========================================================================
#                MASTER PREPROCESSING FUNCTION 
# =========================================================================
def load_and_preprocess_data():
    """
    Loads all data, engineers "Ancestral Eye" and Metadata features.
    This is the *only* feature engineering step.
    """
    
   
    print("Protocol: Loading data...")
    try:
        train_df = pd.read_csv(os.path.join(C.BASE_PATH, 'train.csv'))
        test_df = pd.read_csv(os.path.join(C.BASE_PATH, 'test.csv'))
        sub_df = pd.read_csv(os.path.join(C.BASE_PATH, 'sample_submission.csv'))
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}. Check BASE_PATH in C. is correct.")
        return None, None, None, None, None

   
    train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(C.BASE_PATH, x))
    test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(C.BASE_PATH, x))

    # (Debug Slicer)
    if C.DEBUG_MODE:
        print(f"Protocol: DEBUG_MODE active. Slicing dataframes...")
        train_unique_ids = train_df['sample_id'].unique()
        n_train_samples = min(C.DEBUG_SAMPLES, len(train_unique_ids))
        debug_train_ids = np.random.choice(train_unique_ids, n_train_samples, replace=False)
        train_df = train_df[train_df['sample_id'].isin(debug_train_ids)].reset_index(drop=True)
        C.N_FOLDS = 2 # Use 2 folds for debug
        print(f"Protocol: Sliced to {len(train_df)} training rows. N_FOLDS set to {C.N_FOLDS}.")

    # ===  FEATURE ENGINEERING ===
    print("Protocol: Beginning Feature Engineering...")
    
    # Combine train/test for easy processing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # ---  "Ancestral Eye" (Texture Features) ---
    print("Protocol: Calculating 'Ancestral Eye' (GLCM) features...")
    texture_features = full_df['image_path'].progress_apply(get_texture_features)
    texture_cols = [f'glcm_{i}' for i in range(texture_features.iloc[0].shape[0])]
    texture_df = pd.DataFrame(np.stack(texture_features), columns=texture_cols)
    full_df = pd.concat([full_df, texture_df], axis=1)

    # ---  "Metadata" Features ---
    print("Protocol: Engineering Metadata features...")
    # Handle cyclical date features
    full_df['Sampling_Date'] = pd.to_datetime(full_df['Sampling_Date'], errors='coerce')
    full_df['month'] = full_df['Sampling_Date'].dt.month.fillna(6) # Impute with mid-year
    full_df['month_sin'] = np.sin(2 * np.pi * full_df['month']/12.0)
    full_df['month_cos'] = np.cos(2 * np.pi * full_df['month']/12.0)
            
    # Impute missing data
    full_df['Pre_GSHH_NDVI'] = full_df['Pre_GSHH_NDVI'].fillna(full_df['Pre_GSHH_NDVI'].median())
    full_df['Height_Ave_cm'] = full_df['Height_Ave_cm'].fillna(full_df['Height_Ave_cm'].median())
    
    # Label-Encode Categorical (LGBM handles this well)
    for col in ['State', 'Species', 'target_name']:
        full_df[col] = full_df[col].astype('category')
        
    # ===  FINAL DATASET CREATION ===
    print("Protocol: Creating final feature dataset...")
    
    # Define all features for the model
    # (NOTICE: NO 'cnn_feat_' columns. They are *gone*.)
    lgbm_features = [col for col in full_df.columns if 'glcm_' in col] \
                  + ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month_sin', 'month_cos']
                  
    categorical_features = ['State', 'Species', 'target_name']
    
    # Separate back into train and test
    train_fused = full_df[full_df['is_train'] == 1].reset_index(drop=True)
    test_fused = full_df[full_df['is_train'] == 0].reset_index(drop=True)
    
    X = train_fused[lgbm_features + categorical_features]
    y = train_fused['target']
    X_test = test_fused[lgbm_features + categorical_features]
    groups = train_fused['sample_id'] # For GroupKFold
    
    print(f"Protocol: Preprocessing complete. Training with {len(lgbm_features)} numerical + {len(categorical_features)} categorical features.")
    
    return X, y, X_test, groups, sub_df

print("Protocol: Master Preprocessing function defined.")

# =========================================================================
#                        MASTER TRAINING FUNCTION 
# =========================================================================
def run_lgbm_training(X, y, X_test, groups):
    """
    Trains the "Expert Mind" (LGBM) using "one single step"
    of N-Fold Cross-Validation.
    """
    print(f"Protocol: Beginning 'Expert Mind' (LGBM) {C.N_FOLDS}-Fold Training...")
    
    gkf = GroupKFold(n_splits=C.N_FOLDS)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print("-" * 50)
        print(f"Protocol: Commencing LGBM Fold {fold + 1}/{C.N_FOLDS}...")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        #  parameters for LGBM
        lgbm = lgb.LGBMRegressor(
            objective='regression_l1', # MAE is more robust to outliers
            metric='rmse',
            n_estimators=2000,
            learning_rate=0.01,
            n_jobs=-1,
            seed=C.SEED,
            colsample_bytree=0.7,
            subsample=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        # This is the "keep it learning" part.
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature='auto' # LGBM will auto-detect categories
        )
        
        # --- Prediction ---
        val_preds = lgbm.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        fold_test_preds = lgbm.predict(X_test) / C.N_FOLDS
        test_preds += fold_test_preds
        
        # ---  Save the model using joblib ---
        joblib.dump(lgbm, f'lgbm_model_fold_{fold}.joblib')
        
        print(f"Protocol: Fold {fold + 1} complete. Val R-Squared: {r2_score(y_val, val_preds):.4f}")
        
    return oof_preds, test_preds

print("Protocol: Master Training function defined.")

# =========================================================================
#                  FINAL SUBMISSION FUNCTION 
# =========================================================================
def create_submission(y_true, oof_preds, test_preds, sub_df):
    """
    Calculates the final OOF score and creates submission.csv
    """
    print("-" * 50)
    print("Protocol: ALL FOLDS complete. Calculating final scores...")
    
    # Calculate OOF (Out-of-Fold) R-squared
    final_r2 = r2_score(y_true, oof_preds)
    print(f"Protocol: Final OOF R-Squared (Metadata + Ancestral Eye): {final_r2:.4f}")
    
    print("Protocol: Creating final submission file...")
    # Ensure no negative biomass predictions
    test_preds[test_preds < 0] = 0
    
    sub_df['target'] = test_preds
    sub_df.to_csv('submission.csv', index=False)
    
    print("-" * 50)
    print(f"Protocol: VINDICATION COMPLETE. Final R-Squared: {final_r2:.4f}")
    print("Submission file 'submission.csv' created.")
    print("-" * 50)
    
    if final_r2 > 0:
        print("Protocol: We have a positive R-Squared. The hypothesis is valid.")
    else:
        print("Protocol: The R-Squared is negative. The hypothesis has failed.")
        
print("Protocol: Submission function defined.")

# =========================================================================
#                             PROTOCOL EXECUTION 
# =========================================================================

# (CRITICAL) must be installed for the LGBM model
#  must have "Internet" turned ON just for this pip command 
# for this !pip command to work.
try:
    import lightgbm
except ImportError:
    print("Protocol: Installing lightgbm...")
    !pip install lightgbm
    print("Protocol: lightgbm installed. Please 'Restart & Run All'.")

# must be installed for the 'Ancestral Eye' my core idea
try:
    import skimage
except ImportError:
    print("Protocol: Installing skimage...")
    !pip install scikit-image
    print("Protocol: skimage installed. Please 'Restart & Run All'.")

# --- Main Execution ---
if __name__ == "__main__":
    
    # Set global seed (again) for safety
    np.random.seed(C.SEED)
    
    # 1. Load and process all data

    X, y, X_test, groups, sub_df = load_and_preprocess_data()
    
    if X is not None:
        # 2. Run the full training protocol
        oof_preds, test_preds = run_lgbm_training(X, y, X_test, groups)
        
        # 3. Create the final submission file
        create_submission(y, oof_preds, test_preds, sub_df)
    
    else:
        print("Protocol: Halting execution due to data loading error.")

print("Protocol: End of Notebook.")

