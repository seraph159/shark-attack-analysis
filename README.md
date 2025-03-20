# Shark Attack Analysis

## Overview
This project performs exploratory data analysis (EDA) on global shark attack incidents data to identify patterns, trends, and insights about shark attacks worldwide.

## Data Source
The dataset used in this analysis is sourced from Kaggle:
[Global Shark Attack Incidents](https://www.kaggle.com/datasets/thedevastator/global-shark-attack-incidents)

The dataset contains historical records of shark attacks globally, including information about the date, location, victim details, activity during the attack, shark species, and fatality outcome.

## Project Structure
- `GSAF5.xls.csv` - Original dataset from Kaggle
- `shark_attacks_cleaned.csv` - Cleaned dataset after preprocessing
- `shark_attack_eda.py` - Python script for exploratory data analysis

## Analysis Performed
The exploratory data analysis includes:

1. **Data Understanding**
   - Dataset structure examination
   - Missing values identification
   - Variable exploration

2. **Data Cleaning**
   - Handling missing values
   - Fixing data types
   - Removing unnecessary columns

3. **Exploratory Analysis**
   - Univariate analysis (distributions, central tendencies)
   - Bivariate analysis (relationships between variables)
   - Multivariate analysis (interactions between multiple variables)
   - Domain-specific analysis:
     - Temporal analysis (trends over years)
     - Geographical analysis (countries with most attacks)
     - Activity analysis (activities leading to shark attacks)
     - Fatality analysis (factors contributing to fatal attacks)
     - Species analysis (shark species most involved in attacks)

4. **Visualization**
   - Various plots and charts to visualize key insights

5. **Statistical Analysis**
   - Correlation analysis between variables

### Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

### Running the Analysis
1. Clone this repository
2. Ensure you have the required Python packages installed
3. Run the analysis script:
   ```
   python shark_attack_eda.py
   ```
4. Check the generated visualizations in the `plots` directory

## License
See the LICENSE file for details.

## Acknowledgements
Data provided by [The Devastator](https://www.kaggle.com/thedevastator) on Kaggle.