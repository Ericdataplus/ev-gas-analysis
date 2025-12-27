"""
Master ML Training Script

Trains ALL machine learning models for the EV analysis project:
1. EV Adoption Models (sales, stock, market share)
2. Infrastructure Models (charging, gas stations)
3. Waste & Environmental Models (emissions, recycling)
4. Price Prediction Models (EV prices, battery costs)

Uses GPU acceleration with RTX 3060 (12GB VRAM)
Trains multiple model types and selects the best for each target.
"""

import sys
from pathlib import Path
import time

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def check_gpu():
    """Check GPU availability."""
    print("="*60)
    print("GPU CHECK")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU detected, using CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed")
        return False


def check_libraries():
    """Check available ML libraries."""
    print("\n" + "="*60)
    print("LIBRARY CHECK")
    print("="*60)
    
    libs = {}
    
    try:
        import xgboost
        libs['XGBoost'] = xgboost.__version__
        print(f"✓ XGBoost: {xgboost.__version__}")
    except ImportError:
        print("✗ XGBoost not available")
    
    try:
        import lightgbm
        libs['LightGBM'] = lightgbm.__version__
        print(f"✓ LightGBM: {lightgbm.__version__}")
    except ImportError:
        print("✗ LightGBM not available")
    
    try:
        import catboost
        libs['CatBoost'] = catboost.__version__
        print(f"✓ CatBoost: {catboost.__version__}")
    except ImportError:
        print("✗ CatBoost not available")
    
    try:
        import optuna
        libs['Optuna'] = optuna.__version__
        print(f"✓ Optuna: {optuna.__version__}")
    except ImportError:
        print("✗ Optuna not available")
    
    try:
        import sklearn
        libs['Scikit-learn'] = sklearn.__version__
        print(f"✓ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn not available")
    
    return libs


def train_ev_adoption_models():
    """Train EV adoption prediction models."""
    print("\n" + "#"*60)
    print("# EV ADOPTION MODELS")
    print("#"*60)
    
    try:
        from scripts.ml.train_ev_adoption_models import main as train_adoption
        train_adoption()
        return True
    except Exception as e:
        print(f"Error training EV adoption models: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_infrastructure_models():
    """Train infrastructure prediction models."""
    print("\n" + "#"*60)
    print("# INFRASTRUCTURE MODELS")
    print("#"*60)
    
    try:
        from scripts.ml.train_infrastructure_models import train_all_infrastructure_models
        train_all_infrastructure_models()
        return True
    except Exception as e:
        print(f"Error training infrastructure models: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_waste_models():
    """Train waste prediction models."""
    print("\n" + "#"*60)
    print("# WASTE & ENVIRONMENTAL MODELS")
    print("#"*60)
    
    try:
        from scripts.ml.train_waste_models import train_waste_models as train_waste
        train_waste()
        return True
    except Exception as e:
        print(f"Error training waste models: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_summary():
    """Generate summary of all trained models."""
    import pandas as pd
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Check for result files
    result_files = [
        ('ml_model_comparison.csv', 'EV Adoption Models'),
        ('infrastructure_model_results.csv', 'Infrastructure Models'),
        ('waste_ml_predictions.csv', 'Waste Models'),
    ]
    
    for filename, model_type in result_files:
        filepath = REPORT_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"\n{model_type}:")
            if 'r2' in df.columns:
                print(f"  Best R²: {df['r2'].max():.4f}")
                print(f"  Models trained: {len(df)}")
            elif 'prediction' in df.columns:
                print(f"  Predictions generated: {len(df)}")
        else:
            print(f"\n{model_type}: No results found")
    
    # List prediction files
    pred_files = list(REPORT_DIR.glob('*prediction*.csv'))
    print(f"\nPrediction files generated: {len(pred_files)}")
    for f in pred_files:
        print(f"  - {f.name}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("MASTER ML TRAINING SCRIPT")
    print("EV vs Gas Analysis Project")
    print("="*60)
    
    start_time = time.time()
    
    # Check environment
    has_gpu = check_gpu()
    libs = check_libraries()
    
    if not libs:
        print("\n⚠ No ML libraries available. Please install requirements.")
        return
    
    # Train all models
    results = {
        'ev_adoption': train_ev_adoption_models(),
        'infrastructure': train_infrastructure_models(),
        'waste': train_waste_models(),
    }
    
    # Generate summary
    generate_summary()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"GPU used: {has_gpu}")
    print(f"\nResults:")
    for model_type, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model_type}")
    
    print(f"\nOutput directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()
