# 🎯 Amazon Electronics Brand Analysis & Recommendation System

## 📊 Project Overview
Advanced data analysis and machine learning project for Amazon electronics products, featuring brand analysis, price prediction, and intelligent recommendation system.

## 🚀 Features
- **Brand Strategy Analysis**: Premium, Value, and Budget brand categorization
- **Price Prediction Modeling**: Machine learning price forecasting
- **Intelligent Recommendation System**: 3 types of recommendations
- **RESTful API**: Ready-to-deploy recommendation API

## 📁 Project Structure

amazon-brand-analysis/
├── data/ # Cleaned dataset
├── notebooks/ # Jupyter notebooks (full analysis)
├── src/ # Source code and API
├── docs/ # Documentation and visuals
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## 🛠️ Installation
```bash
git clone https://github.com/alielastal/amazon-brand-analysis.git
cd amazon-brand-analysis
pip install -r requirements.txt```

🎮 Usage
Run notebooks in order (01 to 06)

Deploy API: python src/07_API_Deployment.py

Access endpoints: http://localhost:5000


📊 Results Highlights
HP: Premium quality leader ($61.64, 4.7⭐)

JJC: Best value proposition ($11.86, 4.5⭐)

5+ strategic brand categories identified

269 products across 229 brands analyzed

🌐 API Endpoints
GET /recommend/content/<product_id>

POST /recommend/preferences

GET /recommend/strategy/<strategy_type>

GET /brands, /categories, /products

🤝 Contributing
Feel free to fork and contribute to this project!