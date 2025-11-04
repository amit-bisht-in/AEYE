# =========================================================================
#
#                               PREDICTION 
#
# =========================================================================

print("Protocol: Loading inference libraries...")
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import warnings
import joblib

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

# Silence verbose warnings
warnings.filterwarnings('ignore')

# =========================================================================
#     PARAMETERS & HELPERS 
# =========================================================================

class C:
    # --- Config ---
    N_FOLDS = 5
    IMG_SIZE = 224 # GLCM consistency
    
    # --- Model Path ---
    # Assumes models are in the *same directory* as this script
    # In Kaggle, this would be '/kaggle/your working directory'
    MODEL_PATH = '.' 

print(f"Protocol: Inference parameters loaded. N_FOLDS={C.N_FOLDS}")

def get_texture_features(image_path):
    """
    (The "Ancestral Eye")
    Loads an image, resizes, converts to grayscale, and calculates
    GLCM Haralick texture features. *Must* be identical to training.
    """
    try:
        img = imread(image_path)
        img_resized = resize(img, (C.IMG_SIZE, C.IMG_SIZE), anti_aliasing=True)
        img_gray = rgb2gray(img_resized)
        img_gray = (img_gray * 255).astype(np.uint8)
        
        glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                 'correlation', 'ASM']
        features = np.array([graycoprops(glcm, prop)[0, 0] for prop in props])
        return features
    
    except Exception as e:
        print(f"Warning: Failed to extract texture features for {image_path}: {e}")
        return np.zeros(6)

print("Protocol: 'Ancestral Eye' (get_texture_features) function defined.")

# =========================================================================
#    MASTER PREDICTION FUNCTION ###
# =========================================================================

def predict_biomass(image_path, sampling_date, ndvi, height, state, species, target_name):
    """
    Predicts biomass for a *single* image by replicating
    the full "Last Stand" protocol pipeline.
    
    Args:
        image_path (str): Path to the new image file
        sampling_date (str): e.g., "2024-03-15"
        ndvi (float): Pre_GSHH_NDVI value
        height (float): Height_Ave_cm value
        state (str): e.g., "New South Wales"
        species (str): e.g., "Species_A"
        target_name (str): The *question* (e.g., "GDM_g")
    
    Returns:
        float: The final, averaged biomass prediction, or None if failed.
    """
    
    print("-" * 50)
    print(f"Protocol: Beginning prediction for {image_path}...")
    
    # --- 1. Load the 5 "Expert Mind" models ---
    models = []
    for fold in range(C.N_FOLDS):
        model_file = os.path.join(C.MODEL_PATH, f'lgbm_model_fold_{fold}.joblib')
        try:
            model = joblib.load(model_file)
            models.append(model)
        except FileNotFoundError:
            print(f"FATAL ERROR: Could not find model file: {model_file}")
            print("Please ensure the 5 '.joblib' models are in the same directory.")
            return None
            
    print(f"Protocol: Successfully loaded all {len(models)} models.")
    
    # --- 2. Create the "Input Row" DataFrame ---
    # This *must* match the *exact* structure of the training data.
    
    input_data = {
        'image_path': [image_path],
        'Sampling_Date': [sampling_date],
        'Pre_GSHH_NDVI': [ndvi],
        'Height_Ave_cm': [height],
        'State': [state],
        'Species': [species],
        'target_name': [target_name]
    }
    df = pd.DataFrame(input_data)
    
    # --- 3. Run *all* preprocessing steps ---
    
    # 3a. "Ancestral Eye" (Texture Features)
    print("Protocol: Calculating 'Ancestral Eye' features...")
    texture_features = df['image_path'].apply(get_texture_features)
    texture_cols = [f'glcm_{i}' for i in range(texture_features.iloc[0].shape[0])]
    texture_df = pd.DataFrame(np.stack(texture_features), columns=texture_cols)
    
    # 3b. "Metadata" Features
    print("Protocol: Engineering Metadata features...")
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'], errors='coerce')
    df['month'] = df['Sampling_Date'].dt.month.fillna(6)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    # --- 4. Create Final Feature Vector ---
    print("Protocol: Fusing all 'Expert' features...")
    
    # Define *all* features our models were trained on
    lgbm_features = [f'glcm_{i}' for i in range(6)] \
                  + ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month_sin', 'month_cos']
                  
    categorical_features = ['State', 'Species', 'target_name']
    
    # Combine all our processed data
    X_pred = pd.concat([df, texture_df], axis=1)
    
    # Set categorical dtypes *exactly* as in training
    for col in categorical_features:
        X_pred[col] = X_pred[col].astype('category')
    
    # Select *only* the feature columns in the *correct order*
    X_pred_final = X_pred[lgbm_features + categorical_features]
    
    # --- 5. Predict and Average ---
    print("Protocol: Generating predictions...")
    
    fold_predictions = []
    for fold, model in enumerate(models):
        pred = model.predict(X_pred_final)[0]
        fold_predictions.append(pred)
        print(f"  > Fold {fold} prediction: {pred:.4f}")
    
    # Average the 5 "opinions"
    final_prediction = np.mean(fold_predictions)
    
    # Ensure no negative biomass
    if final_prediction < 0:
        final_prediction = 0
        
    print(f"Protocol: Final Averaged Prediction: {final_prediction:.4f}")
    print("-" * 50)
    
    return final_prediction

# =========================================================================
# ### 3. EXAMPLE USAGE ###
# =========================================================================

if __name__ == "__main__":
    
    # This is a *hypothetical* example of how you would call this script.
    # We are "making up" metadata for an image.
    
    # 1. Define the input data
    new_image_path = '/kaggle/input/csiro-image2biomass-prediction/test/ID001187975.jpg' # Just an example
    
    # We *must* provide the metadata our model needs
    image_metadata = {
        "sampling_date": "2021-09-15",
        "ndvi": 0.65,
        "height": 25.0,
        "state": "Victoria",
        "species": "Species_B"
    }
    
    # We also *must* provide the *question* we are asking
    target_question = "GDM_g" # What do we want to predict?
    
    # 2. Run the prediction
    final_biomass = predict_biomass(
        image_path=new_image_path,
        sampling_date=image_metadata["sampling_date"],
        ndvi=image_metadata["ndvi"],
        height=image_metadata["height"],
        state=image_metadata["state"],
        species=image_metadata["species"],
        target_name=target_question
    )
    
    if final_biomass is not None:
        print(f"\n--- FINAL RESULT ---")
        print(f"The predicted '{target_question}' for {os.path.basename(new_image_path)} is: {final_biomass:.2f} grams")
