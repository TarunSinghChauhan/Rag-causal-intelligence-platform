# Create causal uplift modeling implementation code
causal_implementation_code = '''
"""
Causal Uplift Modeling Implementation for Growth Intelligence
Treatment effect estimation and targeting optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Install causalml if not already installed
# pip install causalml

try:
    from causalml.inference.meta import TLearner, XLearner, RLearner
    from causalml.metrics import plot_gain, plot_qini, qini_score, auuc_score
    CAUSALML_AVAILABLE = True
except ImportError:
    print("CausalML not available. Install with: pip install causalml")
    CAUSALML_AVAILABLE = False

class UpliftModelingPipeline:
    """
    Complete uplift modeling pipeline for causal inference and targeting
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize uplift modeling pipeline"""
        self.random_state = random_state
        self.models = {}
        self.evaluation_results = {}
        
        # Meta-learners for uplift modeling
        if CAUSALML_AVAILABLE:
            self.meta_learners = {
                'T-Learner': TLearner(learner=RandomForestRegressor(random_state=random_state)),
                'X-Learner': XLearner(learner=RandomForestRegressor(random_state=random_state)),
                'R-Learner': RLearner(learner=RandomForestRegressor(random_state=random_state))
            }
        else:
            self.meta_learners = {}
    
    def prepare_data(self, df: pd.DataFrame, 
                    treatment_col: str, outcome_col: str, 
                    feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for uplift modeling"""
        
        # Extract features, treatment, and outcome
        X = df[feature_cols].values
        T = df[treatment_col].values
        Y = df[outcome_col].values
        
        print(f"Data prepared: {len(df)} samples, {len(feature_cols)} features")
        print(f"Treatment distribution: {np.bincount(T)}")
        print(f"Outcome mean: {Y.mean():.3f}")
        
        return X, T, Y
    
    def fit_uplift_models(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Dict:
        """Fit multiple uplift models and compare performance"""
        
        if not CAUSALML_AVAILABLE:
            print("CausalML not available. Using simplified T-Learner implementation.")
            return self._fit_simple_tlearner(X, T, Y)
        
        results = {}
        
        # Split data for evaluation
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T, Y, test_size=0.3, random_state=self.random_state, stratify=T
        )
        
        for name, model in self.meta_learners.items():
            print(f"Training {name}...")
            
            # Fit the model
            model.fit(X_train, T_train, Y_train)
            
            # Predict treatment effects
            uplift_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            qini = qini_score(Y_test, uplift_pred, T_test)
            auc = auuc_score(Y_test, uplift_pred, T_test)
            
            results[name] = {
                'model': model,
                'uplift_predictions': uplift_pred,
                'qini_score': qini,
                'auuc_score': auc,
                'test_data': (X_test, T_test, Y_test)
            }
            
            print(f"{name} - Qini: {qini:.3f}, AUUC: {auc:.3f}")
        
        self.evaluation_results = results
        return results
    
    def _fit_simple_tlearner(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Dict:
        """Simplified T-Learner implementation when CausalML unavailable"""
        
        # Split data
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T, Y, test_size=0.3, random_state=self.random_state, stratify=T
        )
        
        # Train separate models for treatment and control
        model_control = RandomForestRegressor(random_state=self.random_state)
        model_treatment = RandomForestRegressor(random_state=self.random_state)
        
        # Fit models
        control_mask = T_train == 0
        treatment_mask = T_train == 1
        
        model_control.fit(X_train[control_mask], Y_train[control_mask])
        model_treatment.fit(X_train[treatment_mask], Y_train[treatment_mask])
        
        # Predict uplift
        Y0_pred = model_control.predict(X_test)
        Y1_pred = model_treatment.predict(X_test)
        uplift_pred = Y1_pred - Y0_pred
        
        # Simple evaluation
        treatment_effect = uplift_pred.mean()
        
        return {
            'Simple T-Learner': {
                'model': (model_control, model_treatment),
                'uplift_predictions': uplift_pred,
                'treatment_effect': treatment_effect,
                'test_data': (X_test, T_test, Y_test)
            }
        }
    
    def segment_analysis(self, X: np.ndarray, uplift_predictions: np.ndarray, 
                        feature_names: List[str], n_segments: int = 5) -> pd.DataFrame:
        """Analyze uplift by customer segments"""
        
        # Create segments based on uplift predictions
        segments = pd.qcut(uplift_predictions, q=n_segments, labels=[f'Segment_{i+1}' for i in range(n_segments)])
        
        # Create analysis dataframe
        df_analysis = pd.DataFrame(X, columns=feature_names)
        df_analysis['uplift'] = uplift_predictions
        df_analysis['segment'] = segments
        
        # Calculate segment statistics
        segment_stats = df_analysis.groupby('segment').agg({
            'uplift': ['mean', 'std', 'count'],
            **{col: 'mean' for col in feature_names}
        }).round(3)
        
        return segment_stats
    
    def calculate_roi(self, uplift_predictions: np.ndarray, 
                     cost_per_treatment: float, revenue_per_conversion: float,
                     treatment_rate: float = 1.0) -> Dict:
        """Calculate ROI for different targeting strategies"""
        
        n_customers = len(uplift_predictions)
        
        # Sort customers by predicted uplift (descending)
        sorted_indices = np.argsort(uplift_predictions)[::-1]
        sorted_uplift = uplift_predictions[sorted_indices]
        
        # Calculate cumulative metrics for different percentiles
        percentiles = np.arange(10, 101, 10)
        roi_analysis = []
        
        for p in percentiles:
            # Top p% of customers
            n_treated = int(n_customers * p / 100)
            top_uplift = sorted_uplift[:n_treated]
            
            # Calculate expected outcomes
            incremental_conversions = top_uplift.sum() * treatment_rate
            total_cost = n_treated * cost_per_treatment
            total_revenue = incremental_conversions * revenue_per_conversion
            roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
            
            roi_analysis.append({
                'percentile': p,
                'customers_treated': n_treated,
                'incremental_conversions': incremental_conversions,
                'total_cost': total_cost,
                'total_revenue': total_revenue,
                'roi': roi,
                'avg_uplift': top_uplift.mean()
            })
        
        return pd.DataFrame(roi_analysis)
    
    def targeting_recommendations(self, X: np.ndarray, uplift_predictions: np.ndarray,
                                feature_names: List[str], 
                                cost_per_treatment: float = 10.0,
                                revenue_per_conversion: float = 50.0) -> Dict:
        """Generate targeting recommendations based on uplift modeling"""
        
        # Calculate ROI analysis
        roi_df = self.calculate_roi(uplift_predictions, cost_per_treatment, revenue_per_conversion)
        
        # Find optimal targeting percentile (highest ROI)
        optimal_idx = roi_df['roi'].idxmax()
        optimal_percentile = roi_df.loc[optimal_idx, 'percentile']
        optimal_roi = roi_df.loc[optimal_idx, 'roi']
        
        # Customer segments analysis
        segment_stats = self.segment_analysis(X, uplift_predictions, feature_names)
        
        # Top customers to target
        n_customers = len(uplift_predictions)
        n_target = int(n_customers * optimal_percentile / 100)
        top_indices = np.argsort(uplift_predictions)[::-1][:n_target]
        
        recommendations = {
            'optimal_targeting': {
                'percentile': optimal_percentile,
                'roi': optimal_roi,
                'customers_to_target': n_target,
                'expected_incremental_conversions': roi_df.loc[optimal_idx, 'incremental_conversions'],
                'expected_revenue': roi_df.loc[optimal_idx, 'total_revenue'],
                'total_cost': roi_df.loc[optimal_idx, 'total_cost']
            },
            'roi_by_percentile': roi_df,
            'segment_analysis': segment_stats,
            'top_customer_indices': top_indices
        }
        
        return recommendations
    
    def visualize_results(self, model_name: str = None):
        """Create visualizations for uplift modeling results"""
        
        if not self.evaluation_results:
            print("No evaluation results available. Run fit_uplift_models() first.")
            return
        
        # Use best model if not specified
        if model_name is None:
            if CAUSALML_AVAILABLE:
                model_name = max(self.evaluation_results.keys(), 
                               key=lambda k: self.evaluation_results[k]['qini_score'])
            else:
                model_name = list(self.evaluation_results.keys())[0]
        
        results = self.evaluation_results[model_name]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Uplift distribution
        axes[0, 0].hist(results['uplift_predictions'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title(f'{model_name}: Uplift Distribution')
        axes[0, 0].set_xlabel('Predicted Uplift')
        axes[0, 0].set_ylabel('Frequency')
        
        # Uplift vs actual (if available)
        if CAUSALML_AVAILABLE:
            X_test, T_test, Y_test = results['test_data']
            
            # Plot Qini curve if available
            try:
                plot_qini(Y_test, results['uplift_predictions'], T_test, ax=axes[0, 1])
                axes[0, 1].set_title(f'{model_name}: Qini Curve')
            except:
                axes[0, 1].text(0.5, 0.5, 'Qini curve not available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, 'CausalML not available\\nfor advanced plots', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Feature importance (if available)
        axes[1, 0].text(0.5, 0.5, 'Feature importance\\nwould be displayed here', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Feature Importance')
        
        # Summary statistics
        stats_text = f"""
        Model: {model_name}
        Mean Uplift: {results['uplift_predictions'].mean():.3f}
        Std Uplift: {results['uplift_predictions'].std():.3f}
        """
        
        if CAUSALML_AVAILABLE and 'qini_score' in results:
            stats_text += f"Qini Score: {results['qini_score']:.3f}\\n"
            stats_text += f"AUUC Score: {results['auuc_score']:.3f}"
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Model Performance')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

# Sample data generation for testing
def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic campaign data for testing"""
    
    np.random.seed(42)
    
    # Customer features
    age = np.random.normal(35, 12, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    previous_purchases = np.random.poisson(3, n_samples)
    engagement_score = np.random.beta(2, 5, n_samples)
    
    # Treatment assignment (randomized)
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # Outcome with heterogeneous treatment effects
    # High-income, engaged customers benefit more from treatment
    base_conversion_rate = 0.1 + 0.05 * (income > 60000) + 0.03 * (engagement_score > 0.5)
    treatment_effect = 0.05 + 0.08 * (income > 60000) * (engagement_score > 0.5)
    
    conversion_rate = base_conversion_rate + treatment * treatment_effect
    outcome = np.random.binomial(1, conversion_rate)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'previous_purchases': previous_purchases,
        'engagement_score': engagement_score,
        'treatment': treatment,
        'conversion': outcome
    })
    
    return df

if __name__ == "__main__":
    print("Causal Uplift Modeling Implementation")
    print("=" * 50)
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(10000)
    
    print(f"Sample data shape: {df.shape}")
    print(f"Treatment groups: {df['treatment'].value_counts().to_dict()}")
    print(f"Overall conversion rate: {df['conversion'].mean():.3f}")
    
    # Initialize pipeline
    pipeline = UpliftModelingPipeline()
    
    # Prepare data
    feature_cols = ['age', 'income', 'previous_purchases', 'engagement_score']
    X, T, Y = pipeline.prepare_data(df, 'treatment', 'conversion', feature_cols)
    
    # Fit models
    print("\\nFitting uplift models...")
    results = pipeline.fit_uplift_models(X, T, Y)
    
    if results:
        # Get best model results
        best_model = list(results.keys())[0]
        uplift_predictions = results[best_model]['uplift_predictions']
        
        # Generate recommendations
        print("\\nGenerating targeting recommendations...")
        recommendations = pipeline.targeting_recommendations(
            X, uplift_predictions, feature_cols,
            cost_per_treatment=10, revenue_per_conversion=50
        )
        
        print(f"\\nOptimal targeting strategy:")
        opt = recommendations['optimal_targeting']
        print(f"- Target top {opt['percentile']:.0f}% of customers")
        print(f"- Expected ROI: {opt['roi']:.2f}x")
        print(f"- Incremental conversions: {opt['expected_incremental_conversions']:.0f}")
        print(f"- Expected revenue: ${opt['expected_revenue']:.0f}")
        print(f"- Total cost: ${opt['total_cost']:.0f}")
    
    print("\\n" + "=" * 50)
    print("Causal uplift modeling implementation complete!")
    print("Key components implemented:")
    print("✓ Meta-learner models (T-Learner, X-Learner, R-Learner)")
    print("✓ Qini curve evaluation and AUUC scoring")
    print("✓ Customer segmentation analysis")
    print("✓ ROI optimization and targeting recommendations")
    print("✓ Visualization and reporting framework")
'''

# Save the code to a file
with open("causal_uplift_modeling.py", "w") as f:
    f.write(causal_implementation_code)

print("Created causal_uplift_modeling.py")
print("=" * 50)
print("Contents:")
print("- UpliftModelingPipeline class")
print("- Meta-learner implementations (T/X/R-Learner)")
print("- Qini curve and AUUC evaluation metrics")
print("- Customer segmentation and ROI analysis")
print("- Targeting recommendations with optimization")
print("- Synthetic data generation for testing")
print("- Visualization framework")