# Credit Scoring Model

**CodeAlpha Machine Learning Internship - Task 1**

---

##  Project Overview

This project implements a **Credit Scoring Model** that predicts an individual's creditworthiness using past financial data. The system employs multiple machine learning classification algorithms to assess credit risk and make lending decisions.

###  Objective

Predict whether a loan applicant is a good or bad credit risk based on their financial history and personal information.

---

##  Machine Learning Algorithms

The project implements and compares four classification algorithms:

1. **Logistic Regression** - Linear model for baseline performance
2. **Decision Tree** - Non-linear decision-making model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Advanced ensemble method

---

##  Dataset

### Features (16 attributes):

**Demographics:**
- `age` - Age of the applicant
- `income` - Annual income
- `employment_length` - Years of employment
- `homeownership` - Home ownership status (rent/own/mortgage)
- `num_dependents` - Number of dependents

**Credit History:**
- `credit_history_length` - Length of credit history
- `num_credit_lines` - Number of credit lines
- `num_open_accounts` - Number of open accounts

**Financial Behavior:**
- `total_debt` - Total debt amount
- `debt_to_income_ratio` - DTI ratio
- `credit_utilization` - Credit utilization ratio

**Payment History:**
- `num_late_payments` - Number of late payments
- `num_delinquencies` - Number of delinquencies
- `bankruptcies` - Number of bankruptcies

**Loan Information:**
- `loan_amount` - Requested loan amount
- `loan_purpose` - Purpose of loan (home/auto/personal/education/business)

### Target Variable:
- `credit_risk` - 0 = Good Credit (Approve), 1 = Bad Credit (Deny)

---

##  Technologies Used

- **Python 3.8+**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization

---

##  Project Structure

```
CodeAlpha_CreditScoring/
│
├── credit_scoring.py          # Main implementation
├── generate_data.py           # Dataset generation script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── credit_data.csv            # Dataset 
│
└── Visualizations/
    ├── target_distribution.png
    ├── numerical_features.png
    ├── correlation_matrix.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

---

##  Installation & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python generate_data.py
```

### 3. Run Credit Scoring Model

```bash
python credit_scoring.py
```

---

##  Code Usage

### Basic Implementation

```python
from credit_scoring import CreditScoringModel

# Initialize model
model = CreditScoringModel('credit_data.csv')

# Load and analyze data
model.load_data()
model.exploratory_data_analysis()

# Prepare data
model.prepare_data(test_size=0.2)

# Train models
model.train_models()

# Evaluate models
model.evaluate_models()

# Visualize results
model.visualize_results()

# Get best model
best_model_name, best_model = model.get_best_model()
```

### Making Predictions

```python
# New applicant data
applicant = {
    'age': 35,
    'income': 75000,
    'employment_length': 10,
    'credit_history_length': 12,
    'num_credit_lines': 5,
    'num_open_accounts': 3,
    'total_debt': 25000,
    'debt_to_income_ratio': 0.33,
    'credit_utilization': 0.3,
    'num_late_payments': 2,
    'num_delinquencies': 0,
    'bankruptcies': 0,
    'loan_amount': 15000,
    'loan_purpose': 'auto',
    'homeownership': 'own',
    'num_dependents': 2
}

# Predict creditworthiness
prediction, probability = model.predict_creditworthiness(applicant)
```

---

##  Evaluation Metrics

The models are evaluated using five key metrics:

1. **Accuracy** - Overall correctness
2. **Precision** - Accuracy of positive predictions (% of predicted bad credits that are actually bad)
3. **Recall** - True positive rate (% of actual bad credits correctly identified)
4. **F1-Score** - Harmonic mean of precision and recall
5. **ROC-AUC** - Area under the ROC curve (discrimination ability)

### Why These Metrics Matter in Credit Scoring:

- **High Precision**: Minimizes false positives (denying good customers)
- **High Recall**: Minimizes false negatives (approving bad customers)
- **F1-Score**: Balances both concerns
- **ROC-AUC**: Overall model discrimination ability

---

##  Visualizations

The project generates 7 comprehensive visualizations:

1. **Target Distribution** - Credit risk class balance
2. **Numerical Features** - Feature distributions by credit risk
3. **Correlation Matrix** - Feature correlations
4. **Model Comparison** - Performance metrics across models
5. **ROC Curves** - Model discrimination ability
6. **Confusion Matrices** - Prediction accuracy breakdown
7. **Feature Importance** - Most influential features

---

##  Key Insights

### Important Features for Credit Scoring:

Based on feature importance analysis, the most predictive features typically include:

1. **Debt-to-Income Ratio** - Key indicator of repayment ability
2. **Number of Late Payments** - Direct indicator of payment behavior
3. **Credit Utilization** - Shows credit management
4. **Income** - Repayment capacity
5. **Number of Delinquencies** - Past credit problems
6. **Credit History Length** - Credit experience
7. **Total Debt** - Overall debt burden

---

##  Expected Performance

All models typically achieve:
- **Accuracy**: 75-85%
- **Precision**: 70-85%
- **Recall**: 70-90%
- **F1-Score**: 72-85%
- **ROC-AUC**: 0.80-0.90

*Actual results may vary based on data distribution and random seed.*

---

##  Business Impact

### Risk Mitigation:
- Reduces default rates by identifying high-risk applicants
- Automates credit decisions consistently
- Provides a transparent decision-making process

### Operational Efficiency:
- Faster loan approval process
- Reduced manual review workload
- Scalable to large applicant volumes

### Financial Benefits:
- Lower default losses
- Increased approval of creditworthy applicants
- Better portfolio risk management

---

## 🔮 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Ensemble stacking methods
- [ ] Hyperparameter optimization (GridSearchCV)
- [ ] Real-time API for credit scoring
- [ ] Web dashboard for visualization
- [ ] Explainable AI (SHAP values)
- [ ] Multi-class risk categories
- [ ] Time-series analysis of credit behavior
- [ ] Integration with credit bureaus
- [ ] Mobile application

---

##  Project Deliverables

 **Code Implementation**
- Clean, well-documented Python code
- Object-oriented design
- Modular and reusable functions

 **Data Analysis**
- Exploratory data analysis
- Feature engineering
- Statistical insights

 **Model Development**
- Multiple algorithm implementation
- Cross-validation
- Hyperparameter consideration

 **Model Evaluation**
- Comprehensive metrics
- Comparative analysis
- Business interpretation

 **Visualization**
- Professional plots
- Clear insights
- Publication-ready quality

 **Documentation**
- Detailed README
- Code comments
- Usage examples

---

##  Learning Outcomes

### Technical Skills:
- Classification algorithms
- Feature engineering
- Model evaluation techniques
- Cross-validation
- Ensemble methods
- Data visualization
- Python programming

### Domain Knowledge:
- Credit risk assessment
- Financial metrics (DTI, utilization)
- Lending industry practices
- Risk management

### Professional Skills:
- Project documentation
- Code organization
- Technical communication
- Problem-solving

---


##  Contact


Email: mostafa.eldeeb912@gmail.com 

LinkedIn: https://www.linkedin.com/in/mostafa--eldeeb/  

GitHub: https://github.com/Binary-Potato-0

**CodeAlpha**  
Website: www.codealpha.tech  
Email: services@codealpha.tech  
WhatsApp: +91 8052293611

---

##  License

This project is part of CodeAlpha Machine Learning Internship program.

---

## 🙏 Acknowledgments

- **CodeAlpha** for the internship opportunity
- **UCI Machine Learning Repository** for dataset inspiration
- **Scikit-learn** community for excellent tools and documentation
- All mentors and peers for guidance

---

##  References

1. [Credit Scoring Best Practices](https://www.investopedia.com/terms/c/credit_score.asp)
2. [Machine Learning for Credit Risk](https://arxiv.org/abs/1811.00102)
3. [Scikit-learn Classification](https://scikit-learn.org/stable/supervised_learning.html)
4. [Fair Lending Practices](https://www.consumerfinance.gov/compliance/compliance-resources/fair-lending/)

---

**⭐ If you find this project helpful, please star the repository!**

---

*Developed with ❤️ for CodeAlpha Machine Learning Internship*  
