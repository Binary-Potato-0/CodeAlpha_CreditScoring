"""
Credit Scoring Model
CodeAlpha Machine Learning Internship - Task 1

This project predicts an individual's creditworthiness using past financial data
and multiple classification algorithms including Logistic Regression, Decision Trees,
and Random Forest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class CreditScoringModel:
    """
    A comprehensive credit scoring system using multiple ML algorithms
    to predict creditworthiness based on financial history
    """
    
    def __init__(self, data_path):
        """
        Initialize the credit scoring model
        
        Parameters:
        -----------
        data_path : str
            Path to the credit dataset
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("=" * 80)
        print("CREDIT SCORING MODEL - DATA LOADING")
        print("=" * 80)
        
        self.data = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Number of Records: {self.data.shape[0]}")
        print(f"Number of Features: {self.data.shape[1]}")
        
        print("\n" + "=" * 80)
        print("SAMPLE DATA (First 5 Records)")
        print("=" * 80)
        print(self.data.head())
        
        print("\n" + "=" * 80)
        print("DATASET INFORMATION")
        print("=" * 80)
        print(self.data.info())
        
        print("\n" + "=" * 80)
        print("STATISTICAL SUMMARY")
        print("=" * 80)
        print(self.data.describe())
        
        return self.data
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Check for missing values
        print("\n1. MISSING VALUES CHECK:")
        print("-" * 40)
        missing = self.data.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found!")
        else:
            print("Missing values per column:")
            print(missing[missing > 0])
        
        # Target variable distribution
        target_col = 'credit_risk'
        print(f"\n2. TARGET VARIABLE DISTRIBUTION ({target_col}):")
        print("-" * 40)
        print(self.data[target_col].value_counts())
        print(f"\nClass Balance:")
        print(self.data[target_col].value_counts(normalize=True) * 100)
        
        # Create visualizations
        self._create_eda_visualizations()
        
    def _create_eda_visualizations(self):
        """Create comprehensive EDA visualizations"""
        
        # 1. Target Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        target_counts = self.data['credit_risk'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(target_counts.index, target_counts.values, color=colors, alpha=0.8)
        axes[0].set_xlabel('Credit Risk', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Credit Risk Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Good Credit', 'Bad Credit'])
        
        # Add value labels on bars
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(target_counts.values, labels=['Good Credit', 'Bad Credit'], 
                    colors=colors, autopct='%1.1f%%', startangle=90,
                    explode=(0.05, 0.05), shadow=True)
        axes[1].set_title('Credit Risk Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/claude/CodeAlpha_CreditScoring/target_distribution.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Target distribution visualization saved!")
        
        # 2. Numerical Features Distribution
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'credit_risk']
        
        if len(numerical_cols) > 0:
            n_cols = 3
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for idx, col in enumerate(numerical_cols[:len(axes)]):
                self.data.boxplot(column=col, by='credit_risk', ax=axes[idx])
                axes[idx].set_title(f'{col} by Credit Risk', fontweight='bold')
                axes[idx].set_xlabel('Credit Risk')
                axes[idx].set_ylabel(col)
                plt.sca(axes[idx])
                plt.xticks([1, 2], ['Good', 'Bad'])
            
            # Hide extra subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Numerical Features Distribution by Credit Risk', 
                        fontsize=16, fontweight='bold', y=1.001)
            plt.tight_layout()
            plt.savefig('/home/claude/CodeAlpha_CreditScoring/numerical_features.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Numerical features visualization saved!")
        
        # 3. Correlation Matrix
        numerical_data = self.data.select_dtypes(include=[np.number])
        if len(numerical_data.columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation = numerical_data.corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, linewidths=0.5, 
                       cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('/home/claude/CodeAlpha_CreditScoring/correlation_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Correlation matrix saved!")
    
    def feature_engineering(self):
        """Perform feature engineering and preprocessing"""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING & PREPROCESSING")
        print("=" * 80)
        
        # Handle missing values
        print("\n1. Handling Missing Values...")
        self.data = self.data.dropna()
        print(f"✓ Records after removing missing values: {len(self.data)}")
        
        # Encode categorical variables
        print("\n2. Encoding Categorical Variables...")
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'credit_risk']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le
            print(f"   ✓ Encoded: {col}")
        
        # Separate features and target
        X = self.data.drop('credit_risk', axis=1)
        y = self.data['credit_risk']
        
        self.feature_names = X.columns.tolist()
        
        print(f"\n3. Dataset Split:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        
        return X, y
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split and scale the data"""
        print("\n" + "=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)
        
        X, y = self.feature_engineering()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} records")
        print(f"Testing set: {self.X_test.shape[0]} records")
        print(f"\nTraining set class distribution:")
        print(self.y_train.value_counts())
        
        # Scale features
        print("\nScaling features using StandardScaler...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Data preparation completed!")
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\n{'='*40}")
            print(f"Training: {name}")
            print(f"{'='*40}")
            
            # Train
            model.fit(self.X_train_scaled, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='accuracy')
            print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Training accuracy
            train_pred = model.predict(self.X_train_scaled)
            train_acc = accuracy_score(self.y_train, train_pred)
            print(f"Training Accuracy: {train_acc:.4f}")
            
            print(f"✓ {name} training completed!")
        
        print("\n" + "=" * 80)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 80)
    
    def evaluate_models(self):
        """Evaluate all trained models with comprehensive metrics"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"{name.upper()}")
            print(f"{'='*60}")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print metrics
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
            print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
            print(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
            
            # Classification report
            print(f"\n📋 CLASSIFICATION REPORT:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['Good Credit', 'Bad Credit']))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\n📈 CONFUSION MATRIX:")
            print(f"                 Predicted")
            print(f"                 Good  Bad")
            print(f"Actual Good    [{cm[0][0]:4d} {cm[0][1]:4d}]")
            print(f"Actual Bad     [{cm[1][0]:4d} {cm[1][1]:4d}]")
        
        return self.results
    
    def visualize_results(self):
        """Create comprehensive visualizations of model performance"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # 1. Model Performance Comparison
        self._plot_model_comparison()
        
        # 2. ROC Curves
        self._plot_roc_curves()
        
        # 3. Confusion Matrices
        self._plot_confusion_matrices()
        
        # 4. Feature Importance
        self._plot_feature_importance()
        
        print("\n✓ All visualizations generated successfully!")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()]
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Credit Scoring Model - Performance Comparison', 
                    fontsize=18, fontweight='bold', y=1.00)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(metrics_df['Model'], metrics_df[metric], 
                         color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_title(f'{metric}', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_ylim([0, 1.1])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
        
        # Hide last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/claude/CodeAlpha_CreditScoring/model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Model comparison plot saved!")
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(12, 8))
        
        for name in self.results.keys():
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_pred_proba'])
            roc_auc = self.results[name]['roc_auc']
            plt.plot(fpr, tpr, linewidth=2.5, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('ROC Curves - Credit Scoring Models', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/claude/CodeAlpha_CreditScoring/roc_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ ROC curves saved!")
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Confusion Matrices - Credit Scoring Models', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        for idx, (name, result) in enumerate(self.results.items()):
            ax = axes[idx // 2, idx % 2]
            cm = confusion_matrix(self.y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'},
                       xticklabels=['Good Credit', 'Bad Credit'],
                       yticklabels=['Good Credit', 'Bad Credit'])
            ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/claude/CodeAlpha_CreditScoring/confusion_matrices.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confusion matrices saved!")
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        tree_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for idx, model_name in enumerate(tree_models):
            if model_name in self.models:
                model = self.models[model_name]
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                axes[idx].barh(importance_df['feature'], importance_df['importance'], 
                              color=colors[idx], alpha=0.8, edgecolor='black')
                axes[idx].set_xlabel('Importance', fontsize=11, fontweight='bold')
                axes[idx].set_title(f'{model_name}\nTop 10 Features', 
                                   fontsize=12, fontweight='bold')
                axes[idx].invert_yaxis()
                axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/CodeAlpha_CreditScoring/feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved!")
    
    def get_best_model(self):
        """Determine and return the best performing model"""
        print("\n" + "=" * 80)
        print("BEST MODEL SELECTION")
        print("=" * 80)
        
        # Find best model based on F1-score (balance of precision and recall)
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['f1_score'])[0]
        best_model = self.models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"{'='*60}")
        print(f"   Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
        print(f"   Recall:    {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
        print(f"   F1-Score:  {best_metrics['f1_score']:.4f} ({best_metrics['f1_score']*100:.2f}%)")
        print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f} ({best_metrics['roc_auc']*100:.2f}%)")
        
        return best_model_name, best_model
    
    def predict_creditworthiness(self, applicant_data):
        """
        Predict creditworthiness for a new applicant
        
        Parameters:
        -----------
        applicant_data : dict or array-like
            New applicant's financial data
        """
        best_model_name, best_model = self.get_best_model()
        
        # Convert dict to array if necessary
        if isinstance(applicant_data, dict):
            applicant_array = [applicant_data[feat] for feat in self.feature_names]
        else:
            applicant_array = applicant_data
        
        # Scale the data
        applicant_scaled = self.scaler.transform([applicant_array])
        
        # Make prediction
        prediction = best_model.predict(applicant_scaled)[0]
        probability = best_model.predict_proba(applicant_scaled)[0]
        
        print("\n" + "=" * 80)
        print("CREDIT SCORING PREDICTION")
        print("=" * 80)
        print(f"\nModel Used: {best_model_name}")
        print(f"\n{'='*60}")
        
        if prediction == 0:
            print(f"✅ CREDIT APPROVED - Good Credit Risk")
            print(f"   Confidence: {probability[0] * 100:.2f}%")
        else:
            print(f"❌ CREDIT DENIED - Bad Credit Risk")
            print(f"   Confidence: {probability[1] * 100:.2f}%")
        
        print(f"\n   Probability of Good Credit: {probability[0] * 100:.2f}%")
        print(f"   Probability of Bad Credit:  {probability[1] * 100:.2f}%")
        print(f"{'='*60}")
        
        return prediction, probability


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("CREDIT SCORING MODEL")
    print("CodeAlpha Machine Learning Internship - Task 1")
    print("=" * 80)
    
    # Initialize model
    model = CreditScoringModel('/home/claude/CodeAlpha_CreditScoring/credit_data.csv')
    
    # Load and explore data
    model.load_data()
    model.exploratory_data_analysis()
    
    # Prepare data
    model.prepare_data(test_size=0.2, random_state=42)
    
    # Train models
    model.train_models()
    
    # Evaluate models
    model.evaluate_models()
    
    # Visualize results
    model.visualize_results()
    
    # Get best model
    best_model_name, best_model = model.get_best_model()
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   ✓ target_distribution.png")
    print("   ✓ numerical_features.png")
    print("   ✓ correlation_matrix.png")
    print("   ✓ model_comparison.png")
    print("   ✓ roc_curves.png")
    print("   ✓ confusion_matrices.png")
    print("   ✓ feature_importance.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
