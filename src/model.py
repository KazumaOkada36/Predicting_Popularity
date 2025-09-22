import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
from typing import Dict, Tuple
from data_preprocessing import RestaurantDataProcessor

class RestaurantPopularityPredictor:
    """
    Real Machine Learning model for restaurant popularity prediction
    Uses ensemble methods and proper ML pipeline
    """
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.is_trained = False
        self.preprocessor = RestaurantDataProcessor()
        
    def train_models(self, df: pd.DataFrame = None) -> Dict:
        """
        Train multiple ML models and select the best one
        """
        if df is None:
            # Generate training data if not provided
            print("Generating training data...")
            df = self.preprocessor.generate_training_data(1000)
        
        # Prepare features
        print("Preparing features...")
        X, y, self.feature_names = self.preprocessor.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train multiple models
        models_to_train = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  Test MAE: {test_mae:.3f}")
            print(f"  CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Select best model based on test R²
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.models = {k: v['model'] for k, v in results.items()}
        self.is_trained = True
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best R² score: {results[best_model_name]['test_r2']:.3f}")
        
        # Feature importance (for Random Forest)
        if best_model_name == 'random_forest':
            self._print_feature_importance()
        
        return results
    
    def _print_feature_importance(self):
        """Print feature importance for Random Forest model"""
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
    
    def predict_popularity(self, restaurant_data: Dict) -> Dict:
        """
        Make prediction for a single restaurant
        """
        if not self.is_trained:
            # Load pre-trained model if available
            if os.path.exists('models/best_model.pkl'):
                self.load_model()
            else:
                return {"error": "Model not trained. Please train the model first."}
        
        try:
            # Process restaurant data into features
            features = self.preprocessor.process_single_restaurant(restaurant_data)
            
            # Make prediction with best model
            popularity_score = self.best_model.predict(features)[0]
            
            # Get prediction from all models for ensemble confidence
            all_predictions = []
            for model in self.models.values():
                pred = model.predict(features)[0]
                all_predictions.append(pred)
            
            # Calculate prediction confidence based on model agreement
            pred_std = np.std(all_predictions)
            confidence = max(0.5, 1.0 - (pred_std / 2.0))  # Lower std = higher confidence
            
            # Generate growth prediction based on score
            growth_prediction = self._predict_growth_category(popularity_score)
            
            # Generate insights
            insights = self._generate_ml_insights(restaurant_data, popularity_score)
            
            return {
                'popularity_score': round(float(popularity_score), 1),
                'growth_prediction': growth_prediction,
                'confidence': round(float(confidence), 2),
                'insights': insights,
                'prediction_method': 'machine_learning',
                'model_used': 'ensemble',
                'all_model_predictions': [round(p, 1) for p in all_predictions]
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _predict_growth_category(self, score: float) -> str:
        """Convert numerical score to growth category"""
        if score >= 8.0:
            return "High Growth Potential"
        elif score >= 6.5:
            return "Moderate Growth"
        elif score >= 4.5:
            return "Stable"
        else:
            return "Needs Improvement"
    
    def _generate_ml_insights(self, restaurant_data: Dict, score: float) -> list:
        """Generate insights based on ML prediction"""
        insights = []
        
        rating = restaurant_data.get('rating', 0)
        reviews = restaurant_data.get('total_ratings', 0)
        
        # Model-based insights
        insights.append(f"ML prediction confidence: {score:.1f}/10")
        
        if score >= 8:
            insights.append("Model indicates strong market position")
        elif score >= 6:
            insights.append("Model shows moderate success potential")
        else:
            insights.append("Model suggests room for significant improvement")
        
        # Data quality insights
        if reviews < 50:
            insights.append("Limited review data may affect prediction accuracy")
        elif reviews > 200:
            insights.append("Strong review volume provides reliable prediction")
        
        return insights
    
    def save_model(self, filepath: str = 'models/best_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'best_model': self.best_model,
            'all_models': self.models,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save preprocessor
        self.preprocessor.save_preprocessor()
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/best_model.pkl'):
        """Load a pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.models = model_data['all_models']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        # Load preprocessor
        if os.path.exists('models/preprocessor.pkl'):
            self.preprocessor.load_preprocessor()
        
        print(f"Model loaded from {filepath}")
    
    def evaluate_model(self, test_df: pd.DataFrame = None) -> Dict:
        """Evaluate model performance on test data"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if test_df is None:
            # Generate test data
            test_df = self.preprocessor.generate_training_data(200)
        
        X_test, y_test, _ = self.preprocessor.prepare_features(test_df)
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }


# Training script
if __name__ == "__main__":
    print("Training Restaurant Popularity ML Model...")
    
    # Create model instance
    model = RestaurantPopularityPredictor()
    
    # Train models
    results = model.train_models()
    
    # Save the trained model
    model.save_model()
    
    # Evaluate performance
    eval_results = model.evaluate_model()
    print("\nFinal Model Performance:")
    print(f"R² Score: {eval_results['r2']:.3f}")
    print(f"MAE: {eval_results['mae']:.3f}")
    print(f"RMSE: {eval_results['rmse']:.3f}")
    
    print("\nModel training complete! Ready for predictions.")