# IBelieveICanFlyPy

A machine learning project for flight delay prediction using various classification algorithms, including Naive Bayes and LightGBM models.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Documentation](#documentation)
- [License](#license)

## 🎯 Project Overview

IBelieveICanFlyPy is a comprehensive machine learning engineering project designed to predict flight delays. The project implements multiple classification algorithms, provides benchmarking capabilities, and includes a complete ETL pipeline for data processing.

## 📁 Project Structure

```
IBelieveICanFlyPy/
├── data/
│   ├── interim/              # Processed datasets
│   │   └── cleaned_flights.csv
│   ├── models/               # Trained models and results
│   │   ├── benchmarks/       # Performance metrics
│   │   │   ├── lightgbm_feature_importance_nb.csv
│   │   │   ├── lightgbm_random.json
│   │   │   └── naive_bayes.json
│   │   ├── naive_bayes_flight_delay_metadata.json
│   │   └── naive_bayes_flight_delay.pkl
│   └── raw/                  # Original datasets
│       ├── airlines.csv
│       ├── airports.csv
│       └── flights.csv
├── docs/                     # Documentation
│   ├── dicionario_dados_flights.pdf
│   └── Tech Challenge Fase 3 - Machine Learning Engineering.pdf
├── setup/                    # Setup scripts
│   ├── clean_flights.py
│   ├── format_code.sh
│   └── naive_bayes_flight.py
└── src/                      # Source code
    ├── etl/
    │   ├── data_cleaning.py
    │   └── feature_engineering.py
    ├── models/
    │   ├── lightgbm_benchmark.py
    │   ├── naive_bayes_benchmarks.py
    │   └── naive_bayes_pipeline.py
    ├── notebooks/
    │   └── exploratory_data_analysis.ipynb
    └── utils/
        └── config.py
```

## ✨ Features

- **Multiple ML Models**: Naive Bayes and LightGBM implementations
- **Complete ETL Pipeline**: Data cleaning and feature engineering modules
- **Benchmarking System**: Compare model performance and feature importance
- **Exploratory Analysis**: Jupyter notebook for data exploration
- **Modular Architecture**: Well-organized code structure for maintainability

## 🚀 Installation

### Prerequisites

- Python 3.x
- pip or Poetry for package management

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd IBelieveICanFlyPy
```

2. **Download the data files**

Download the required dataset files from Google Drive:
```
https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj
```

Place the downloaded files in the `data/raw/` directory:
- `airlines.csv`
- `airports.csv`
- `flights.csv`

3. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies**

Using pip:
```bash
pip install -r requirements.txt
```

Using Poetry:
```bash
poetry install
```

## 💻 Usage

### Data Preparation

Clean and prepare the raw flight data:
```bash
python setup/clean_flights.py
```

### Model Training

Train the Naive Bayes model:
```bash
python setup/naive_bayes_flight.py
```

### Code Formatting

Format the codebase:
```bash
bash setup/format_code.sh
```

### Running Benchmarks

Compare model performance:
```bash
python src/models/naive_bayes_benchmarks.py
python src/models/lightgbm_benchmark.py
```

### Exploratory Data Analysis

Launch Jupyter notebook for data exploration:
```bash
jupyter notebook src/notebooks/exploratory_data_analysis.ipynb
```

## 🤖 Models

### Naive Bayes Classifier
Primary model for flight delay prediction. The trained model and metadata are stored in `data/models/`.

**Features:**
- Fast training and prediction
- Probabilistic predictions
- Works well with categorical features

### LightGBM Benchmark
Gradient boosting model used for comparison and benchmarking.

**Features:**
- Feature importance analysis
- High performance on structured data
- Efficient memory usage

### Model Outputs
- Performance metrics (JSON format)
- Feature importance rankings
- Trained model artifacts (.pkl files)

## 📊 Data

### Raw Data
- `airlines.csv`: Airline information
- `airports.csv`: Airport details and locations
- `flights.csv`: Historical flight records

### Interim Data
- `cleaned_flights.csv`: Processed and cleaned flight data ready for modeling

### Data Processing
The ETL pipeline includes:
- Data cleaning and validation
- Feature engineering
- Missing value handling
- Data type conversions

## 📖 Documentation

Project documentation is available in the `docs/` directory:

- **Data Dictionary** (`dicionario_dados_flights.pdf`): Detailed description of data fields
- **Technical Specification** (`Tech Challenge Fase 3 - Machine Learning Engineering.pdf`): Project requirements and guidelines

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

Contributions are welcome! Please ensure code is properly formatted using the provided formatting script before submitting.

## 📧 Contact

For questions, suggestions, or issues, please feel free to reach out:

- 🐛 **Issues**: [Open an issue](https://github.com/username/IBelieveICanFlyPy/issues) in this repository
- 💼 **LinkedIn**: [Lucas Mendes Barbosa](https://www.linkedin.com/in/lucas-mendes-barbosa/)
- 📧 **Email**: lucas.mendestech@gmail.com
- 🎵 **Portfolio**: [musicmoodai.com.br](https://musicmoodai.com.br/)


---

**Note**: Make sure all dependencies are installed and the virtual environment is activated before running any scripts.