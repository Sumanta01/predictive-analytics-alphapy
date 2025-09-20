# 🏠 House Price Prediction with AlphaPy AutoML

An optimized automated machine learning pipeline for predicting house prices using the AlphaPy framework with clean, efficient code structure.

## 📊 Project Overview

This project implements a streamlined AlphaPy AutoML solution that:
- Leverages AlphaPy's powerful AutoML framework
- Tests multiple regression algorithms automatically
- Provides comprehensive feature importance analysis
- Handles data preprocessing seamlessly
- Generates professional model comparison reports

### 🎯 Key Results
- **Best Model**: Gradient Boosting with **92.6% R² accuracy**
- **Prediction Error**: Average $70,367 RMSE
- **Top Feature**: `sqft_living` (72.8% importance)
- **Optimized Code**: Clean 248-line implementation

## �‍💻 Author

**Sumanta Swain**
- 🔗 GitHub: [@Sumanta01](https://github.com/Sumanta01)
- 📧 Email: [Contact via GitHub](https://github.com/Sumanta01)
- 💼 Role: Data Scientist & ML Engineer
- 🎯 Specialization: AutoML, Predictive Analytics, Python Development

*Passionate about building efficient, production-ready machine learning solutions with clean code architecture.*

## �🚀 Features

- **AlphaPy AutoML**: Professional AutoML framework integration
- **Multi-Algorithm Testing**: Random Forest, Gradient Boosting, Linear Regression
- **Automatic Preprocessing**: Handles missing values and categorical variables
- **Feature Importance**: Identifies key price drivers
- **Clean Architecture**: Optimized, maintainable code structure
- **Minimal Dependencies**: Streamlined package requirements

## 📁 Project Structure

```
house_price_prediction_alphapy/
├── README.md                 # Project documentation
├── requirements.txt          # Minimal Python dependencies (4 packages)
├── data/
│   └── house_prices.csv     # Dataset (2000 houses, 22 features)
├── src/
│   └── run_pipeline.py      # Optimized AlphaPy AutoML pipeline (248 lines)
└── Scripts/                 # Virtual environment (auto-generated)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Windows PowerShell (for virtual environment activation)

### Setup Instructions

1. **Clone/Download the project**

   ```bash
   cd "C:\Your\Desired\Path"
   ```

2. **Create and activate virtual environment**

   ```powershell
   # Navigate to project directory
   cd house_price_prediction_alphapy
   
   # Activate virtual environment
   .\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**

   ```bash
   cd src
   python run_pipeline.py
   ```

## 📈 Usage

### Basic Usage

```python
# Navigate to src directory and run
python run_pipeline.py
```

### Expected Output

```text
✅ AlphaPy available
🏡 House Price Prediction with AlphaPy AutoML
============================================================
📊 Loading house price dataset...
✅ Dataset loaded: 2000 rows, 22 columns
Target variable: price
Features: 21
Missing values: 0
🔧 Preprocessing data...
✅ Preprocessing complete. Features: 21
🚀 Starting AlphaPy AutoML pipeline...
Training algorithms with AlphaPy framework...
  ⚡ Training Random Forest...
  ⚡ Training Gradient Boosting...
  ⚡ Training Linear Regression...

🎯 AlphaPy AutoML Results
====================================================
     Algorithm  Test_R2  Test_RMSE
Gradient Boosting   0.9260   70366.99
Linear Regression   0.9108   77216.29
    Random Forest   0.8940   84194.69

🏆 Best Performing Model: Gradient Boosting
📊 Test R² Score: 0.9260
💰 Test RMSE: $70,366.99
🎉 AlphaPy AutoML Pipeline Completed Successfully!
```

## 📊 Dataset Information

### Input Features (21 total)
- **sqft_living**: Living space square footage (most important)
- **grade**: Construction quality rating
- **sqft_lot**: Lot size
- **house_age**: Age of the house
- **condition**: Overall condition
- **waterfront**: Waterfront property (yes/no)
- **view**: Quality of view
- And 14 additional features...

### Target Variable
- **price**: House sale price (in dollars)

### Data Quality
- **Size**: 2,000 house records
- **Features**: 22 columns (21 predictors + 1 target)
- **Preprocessing**: Automatic handling of missing values and categorical encoding

## 🧠 Model Performance

| Algorithm | Test R² | RMSE ($) | Performance |
|-----------|---------|----------|-------------|
| **Gradient Boosting** | **0.9260** | **70,367** | **Best** |
| Linear Regression | 0.9108 | 77,216 | Excellent |
| Random Forest | 0.8940 | 84,195 | Very Good |

### 🎯 Model Interpretation

- **R² Score**: Percentage of price variance explained by the model
- **RMSE**: Average prediction error in dollars
- **92.6% Accuracy**: Model explains 92.6% of house price variance

## 🔍 Feature Importance Analysis

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | sqft_living | 72.8% | Living space square footage |
| 2 | grade | 13.6% | Construction quality rating |
| 3 | sqft_lot | 3.2% | Lot size |
| 4 | house_age | 3.1% | Age of the house |
| 5 | condition | 2.5% | Overall condition |

## 🔧 Technical Details

### Dependencies (Minimal & Optimized)

- **pandas>=2.0.0**: Data manipulation and analysis
- **numpy>=1.20.0**: Numerical computations
- **scikit-learn>=1.3.0**: Machine learning algorithms
- **alphapy-pro>=3.0.0**: AutoML framework

### Algorithms Used

1. **Random Forest**: Ensemble of decision trees
2. **Gradient Boosting**: Sequential boosting algorithm (best performer)
3. **Linear Regression**: Baseline linear model

### Validation Strategy

- **Train/Test Split**: 80/20 split
- **Random State**: 42 (for reproducibility)
- **Data Preprocessing**: Automatic handling of missing values and categorical encoding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure virtual environment is activated
   .\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Missing Data File**

   ```text
   Error: Data file not found at 'data/house_prices.csv'
   Solution: Ensure your dataset is in the correct location
   ```

3. **AlphaPy Import Issues**

   ```bash
   # Install AlphaPy-Pro specifically
   pip install alphapy-pro>=3.0.0
   ```

## 📞 Support

For questions or issues:

- Check the troubleshooting section above
- Review the error messages for specific guidance
- Ensure all dependencies are properly installed
- Verify virtual environment activation

## 🎉 Acknowledgments

- **AlphaPy Team** for the excellent AutoML framework
- **Scikit-learn** for the foundational machine learning library  
- **Pandas & NumPy** for data manipulation capabilities
- **Python Community** for the amazing ecosystem

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

### 💡 About This Project

This optimized AlphaPy implementation demonstrates:
- **Clean Code Architecture**: Production-ready, maintainable codebase
- **AutoML Excellence**: Leveraging AlphaPy's powerful automation
- **Performance Optimization**: 92.6% accuracy with minimal dependencies
- **Best Practices**: Following industry standards for ML project structure

**Developed by Sumanta Swain - Building efficient ML solutions with clean architecture** ✨