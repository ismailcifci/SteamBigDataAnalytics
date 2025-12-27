# Steam Games Dataset Analysis - Big Data Project

A comprehensive big data analysis project using Apache Spark to analyze Steam games dataset through MapReduce operations, machine learning techniques, and advanced SQL queries.

## Project Overview

This project performs multi-dimensional analysis on a large-scale Steam games dataset containing over 89,000 games. The analysis encompasses:

- **MapReduce Analysis**: Publisher quality assessment using distributed processing
- **Machine Learning - Classification**: Game success prediction using Logistic Regression
- **Machine Learning - Clustering**: Player engagement pattern discovery using K-Means
- **SQL Analysis**: Genre engagement patterns and pricing strategy relationships

## Dataset

### Source Data
- **File**: `archive1/games_march2025_cleaned.csv`
- **Size**: ~447 MB
- **Total Games**: 89,618
- **Columns**: 47 features including game metadata, reviews, engagement metrics, pricing, and categorical information

### Key Features
- Game metadata (name, appid, release date)
- Publisher and developer information
- Review statistics (positive, negative, total reviews, percentages)
- Engagement metrics (average/median playtime, peak concurrent users)
- Pricing information
- Game categories and genres
- Boolean flags (multiplayer, achievements, trading cards, etc.)

### Processed Datasets
- `archive1/games_march2025_ml_ready.csv`: Preprocessed dataset ready for ML
- `archive1/train_ml_ready.parquet`: Training set (9,576 games)
- `archive1/test_ml_ready.parquet`: Test set (2,313 games)

## Project Structure

```
ST2/
├── archive1/                          # Data directory
│   ├── games_march2025_cleaned.csv    # Original cleaned dataset
│   ├── games_march2025_ml_ready.csv/  # ML-ready CSV (partitioned)
│   ├── train_ml_ready.parquet/        # Training set
│   └── test_ml_ready.parquet/         # Test set
│
├── preprocessing/                     # Data preprocessing notebooks
│   ├── preprocessing_1.ipynb          # Initial data cleaning
│   └── preprocessing_2.ipynb          # ML data preparation
│
├── MapReduce.ipynb                    # Publisher quality analysis
├── ML_logisticr.ipynb                 # Game success classification
├── ML_kmeans.ipynb                     # Engagement pattern clustering
├── SQL_query1.ipynb                    # Genre engagement analysis
├── SQL_query2.ipynb                    # Price strategy analysis
│
└── README.md                           # This file
```

## Prerequisites

### Software Requirements
- **Python**: 3.14.0 or higher
- **Apache Spark**: 4.1.0
- **PySpark**: Included with Spark installation
- **Jupyter Notebook**: For running analysis notebooks

### Python Libraries
- `pyspark` - Spark Python API
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualization
- `numpy` - Numerical computing
- `pandas` - Data manipulation (for visualization)

### System Requirements
- Minimum 8GB RAM recommended
- Sufficient disk space for dataset and intermediate files
- Java 8 or higher (required for Spark)

## Installation

### 1. Install Apache Spark

Download and install Apache Spark 4.1.0 from the official website:
```
https://spark.apache.org/downloads.html
```

### 2. Set Up Python Environment

Create a virtual environment (optional but recommended):
```bash
python -m venv pyspark_env
source pyspark_env/bin/activate  # On Windows: pyspark_env\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install pyspark matplotlib seaborn numpy pandas jupyter findspark
```

### 4. Configure Spark

Set the `SPARK_HOME` environment variable to your Spark installation directory:
```bash
export SPARK_HOME=/path/to/spark
export PATH=$PATH:$SPARK_HOME/bin
```

## Usage

### Running the Analysis

The project is organized as a series of Jupyter notebooks that should be executed in order:

#### Step 1: Data Preprocessing

1. **Initial Cleaning** (`preprocessing/preprocessing_1.ipynb`)
   - Removes unnecessary columns
   - Converts boolean columns to 0/1
   - Filters games with <500 reviews
   - Outputs: `games_march2025_ml_ready.csv`

2. **ML Data Preparation** (`preprocessing/preprocessing_2.ipynb`)
   - Handles null values
   - Encodes categorical variables
   - Scales numerical features
   - Splits into train/test sets
   - Outputs: `train_ml_ready.parquet` and `test_ml_ready.parquet`

#### Step 2: MapReduce Analysis

**Publisher Quality Analysis** (`MapReduce.ipynb`)
- Analyzes which publishers consistently release higher-quality games
- Uses MapReduce paradigm with PySpark RDDs
- Calculates average net Steam scores (positive - negative reviews) per publisher
- Generates rankings and statistics

**Key Results:**
- Analyzes 35,273 publishers (all games)
- Identifies top publishers by average net score
- Provides insights into publisher quality distribution

#### Step 3: Machine Learning - Classification

**Game Success Prediction** (`ML_logisticr.ipynb`)
- Predicts if a game will be "well-received" using Logistic Regression
- Uses weighted score: `(pct_pos_total / 100.0) × log10(num_reviews_total + 1)`
- Trains on 9,576 games, tests on 2,313 games

**Model Performance:**
- AUC-ROC: 0.9222 (92.22%)
- Accuracy: 0.8478 (84.78%)
- F1-Score: 0.8472 (84.72%)
- Precision: 0.8467 (84.67%)
- Recall: 0.8478 (84.78%)

**Outputs:**
- Trained model saved to `models/game_success_lr_model/`
- Evaluation metrics and visualizations
- Feature importance analysis

#### Step 4: Machine Learning - Clustering

**Engagement Pattern Clustering** (`ML_kmeans.ipynb`)
- Identifies distinct game types based on player engagement
- Uses K-Means clustering with 5 features
- Applies log scaling and standardization

**Features Used:**
- Price
- Average playtime (forever)
- Median playtime (forever)
- Total reviews
- Peak concurrent users

**Clustering Results:**
- 5 clusters identified
- Cluster distribution:
  - Cluster 0: 656 games (14.5%) - Low-engagement titles
  - Cluster 1: 1,810 games (39.9%) - Small-scale games
  - Cluster 2: 872 games (19.2%) - High-commitment niche games
  - Cluster 3: 830 games (18.3%) - Mass-market popular games
  - Cluster 4: 366 games (8.1%) - Large-scale premium titles
- WSSSE: 8,453.15

#### Step 5: SQL Analysis

**Genre Engagement Analysis** (`SQL_query1.ipynb`)
- Analyzes player engagement patterns across different game genres
- Uses advanced SQL with CTEs, LATERAL VIEW, and window functions
- Calculates statistical metrics (mean, median, percentiles, standard deviation)

**Key Findings:**
- MMO games show highest median engagement (14.17 hours)
- RPG and Strategy games show high engagement
- Significant variation in engagement across genres

**Price Strategy Analysis** (`SQL_query2.ipynb`)
- Analyzes relationship between pricing and player engagement
- Categorizes games into price buckets
- Examines engagement and popularity by price category

**Key Findings:**
- Non-linear relationship between price and engagement
- $20-40 price range shows optimal balance
- Premium games require exceptional engagement to justify pricing

## Key Results Summary

### Publisher Quality
- Highly fragmented market (most publishers have 1-2 games)
- Top publishers identified by consistent quality metrics
- Average net score: 901.57 (all games), 874.07 (paid games)

### Game Success Prediction
- Strong predictive model with 92% AUC-ROC
- Review quality and volume are key predictors
- Model provides actionable insights for game development

### Engagement Clustering
- Five distinct market segments identified
- Clear patterns in engagement depth, popularity, and pricing
- Useful for strategic planning and market positioning

### Genre Analysis
- Significant inequality in engagement across genres
- MMO, RPG, and Strategy show highest engagement
- Genre choice impacts expected player behavior

### Pricing Strategy
- Optimal price range: $20-40 for balanced engagement
- Premium games require exceptional value
- Budget games achieve volume but struggle with deep engagement

## Technical Details

### Spark Configuration
- **Master**: `local[*]` (uses all available cores)
- **App Name**: Varies by notebook
- **Log Level**: WARN (reduces output noise)

### Data Processing
- **File Formats**: CSV (input), Parquet (processed data)
- **Partitioning**: Automatic based on data size
- **Caching**: Used strategically for iterative operations

### Machine Learning
- **Classification**: Logistic Regression with L2 regularization
- **Clustering**: K-Means with elbow method for optimal k
- **Feature Engineering**: Log scaling, standardization, categorical encoding
- **Evaluation**: Multiple metrics (AUC-ROC, Accuracy, F1, Precision, Recall)

### SQL Techniques
- Common Table Expressions (CTEs)
- LATERAL VIEW for array/string expansion
- Window functions (PERCENTILE_APPROX, AVG, STDDEV)
- String manipulation (TRIM, TRANSLATE, SPLIT)

## Output Files

### Models
- `models/game_success_lr_model/`: Trained Logistic Regression model

### Analysis Results
- `output/publisher_steam_score_analysis_min5games/`: Publisher rankings

### Visualizations
- Generated plots saved within notebooks
- Cluster distribution charts
- Model evaluation metrics
- Feature importance visualizations

## Notes

- All notebooks include detailed markdown explanations
- Code is commented for clarity
- Results are interpreted and discussed in each notebook
- Preprocessing steps are critical - run in order
- Some operations may take time depending on system resources

## Future Enhancements

- Temporal analysis (release date, seasonal patterns)
- External data integration (marketing spend, competition)
- Genre-specific prediction models
- Publisher strategy analysis over time
- Real-time prediction API
- Advanced clustering techniques (DBSCAN, hierarchical)

## License

This project is for educational and research purposes.

---

**Last Updated**: December 2025
**Spark Version**: 4.1.0
**Python Version**: 3.14.0

