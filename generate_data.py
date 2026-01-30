"""
Credit Data Generator
Generates a realistic credit dataset for training the credit scoring model
"""

import pandas as pd
import numpy as np

def generate_credit_dataset(n_samples=1000, random_state=42):
    """
    Generate a realistic credit dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    """
    np.random.seed(random_state)
    
    print("=" * 80)
    print("GENERATING CREDIT DATASET")
    print("=" * 80)
    
    # Generate features
    data = {
        # Demographics
        'age': np.random.randint(18, 75, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        
        # Credit History
        'credit_history_length': np.random.randint(0, 30, n_samples),
        'num_credit_lines': np.random.randint(0, 15, n_samples),
        'num_open_accounts': np.random.randint(0, 10, n_samples),
        
        # Financial Behavior
        'total_debt': np.random.randint(0, 100000, n_samples),
        'debt_to_income_ratio': np.random.uniform(0, 2, n_samples).round(2),
        'credit_utilization': np.random.uniform(0, 1, n_samples).round(2),
        
        # Payment History
        'num_late_payments': np.random.randint(0, 20, n_samples),
        'num_delinquencies': np.random.randint(0, 10, n_samples),
        'bankruptcies': np.random.randint(0, 3, n_samples),
        
        # Loan Details
        'loan_amount': np.random.randint(1000, 50000, n_samples),
        'loan_purpose': np.random.choice(['home', 'auto', 'personal', 'education', 'business'], n_samples),
        
        # Other
        'homeownership': np.random.choice(['rent', 'own', 'mortgage'], n_samples),
        'num_dependents': np.random.randint(0, 6, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable based on features (realistic risk assessment)
    # Higher risk if:
    # - High debt to income ratio
    # - Many late payments
    # - Low income
    # - High credit utilization
    # - Recent bankruptcies
    
    risk_score = (
        (df['debt_to_income_ratio'] * 0.3) +
        (df['num_late_payments'] / 20 * 0.25) +
        (df['num_delinquencies'] / 10 * 0.2) +
        (df['bankruptcies'] / 3 * 0.15) +
        (df['credit_utilization'] * 0.1) +
        ((150000 - df['income']) / 150000 * 0.1) -
        (df['credit_history_length'] / 30 * 0.1)
    )
    
    # Add some randomness
    risk_score += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary (0 = good credit, 1 = bad credit)
    threshold = risk_score.quantile(0.3)  # 30% bad credit (realistic)
    df['credit_risk'] = (risk_score > threshold).astype(int)
    
    # Save to CSV
    output_path = '/home/claude/CodeAlpha_CreditScoring/credit_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"   Location: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {df.shape[1] - 1}")
    print(f"   Samples: {df.shape[0]}")
    
    print("\n" + "=" * 80)
    print("FEATURE DESCRIPTIONS")
    print("=" * 80)
    
    feature_descriptions = {
        'age': 'Age of the applicant (18-74 years)',
        'income': 'Annual income in dollars',
        'employment_length': 'Years of employment',
        'credit_history_length': 'Length of credit history in years',
        'num_credit_lines': 'Number of credit lines',
        'num_open_accounts': 'Number of open credit accounts',
        'total_debt': 'Total debt amount in dollars',
        'debt_to_income_ratio': 'Ratio of debt to income',
        'credit_utilization': 'Credit utilization ratio (0-1)',
        'num_late_payments': 'Number of late payments',
        'num_delinquencies': 'Number of delinquencies',
        'bankruptcies': 'Number of bankruptcies',
        'loan_amount': 'Requested loan amount',
        'loan_purpose': 'Purpose of the loan',
        'homeownership': 'Home ownership status',
        'num_dependents': 'Number of dependents',
        'credit_risk': 'Target: 0 = Good Credit, 1 = Bad Credit'
    }
    
    for feature, description in feature_descriptions.items():
        print(f"   {feature:25s} : {description}")
    
    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION")
    print("=" * 80)
    print(f"\nCredit Risk Distribution:")
    print(df['credit_risk'].value_counts().sort_index())
    print(f"\nPercentage:")
    print(df['credit_risk'].value_counts(normalize=True).sort_index() * 100)
    
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First 5 Records)")
    print("=" * 80)
    print(df.head())
    
    return df


if __name__ == "__main__":
    generate_credit_dataset(n_samples=1000, random_state=42)
