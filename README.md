# Next-Earthquake-Prediction

This project applies machine learning techniques (Random Forest Regression) to historical earthquake records from India, aiming to **predict parameters of future earthquakes** (e.g., magnitude, location, depth) using patterns found in past events. The workflow includes data loading, preprocessing, model training, prediction, and evaluation.

## Data Sources 

- **Dataset:** `Indian_earthquake_data.csv`
  - Includes columns: Origin Time, Latitude, Longitude, Depth, Magnitude, Location.

## Installation
 
1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/next_earthquake_prediction.git
   cd next_earthquake_prediction
   ```

2. **Set up environment and install dependencies:**

   Using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

   *Or, for conda users:*
   ```bash
   conda env create -f environment.yml
   conda activate earthquake-prediction
   ```

## Usage

1. **Place your dataset** (`Indian_earthquake_data.csv`) in the root directory.

2. **Run the notebook:**
   - Open `Next_Earthquake_Prediction.ipynb` in Jupyter Notebook.
   - Execute the cells step by step to process data, train the model, and view results.

3. **Key script snippets:**
   - **Data loading example:**
     ```python
     import pandas as pd
     df = pd.read_csv("Indian_earthquake_data.csv")
     print(df.head())
     ```
   - **Model training snippet:**
     ```python
     from sklearn.ensemble import RandomForestRegressor
     model = RandomForestRegressor()
     model.fit(X_train, y_train)
     ```

## Code Structure

| Folder / File                  | Purpose                                       |
|------------------------------- |-----------------------------------------------|
| `Next_Earthquake_Prediction.ipynb` | Main workflow: load, preprocess, train, predict |
| `Indian_earthquake_data.csv`      | Earthquake dataset (not included if large/public)|
| `requirements.txt` / `environment.yml` | Dependency management                    |
| `src/` (optional)                  | Scripts and modular code                   |
| `notebooks/` (optional)            | All exploratory and main notebooks         |
| `README.md`                        | Documentation (this file)                  |

## Results

- Performance metric: *Mean Absolute Error* on test set.
- Include your key findings and, if possible, tables/plots such as predictions vs. actuals[6].

| Parameter     | Model MAE (Test Set) |
|---------------|---------------------|
| Magnitude     | 0.18                |
| Depth         | 5.4 km              |
| (Example row) | ---                 |

*You can add graphical visualizations as images or notebook exports.*

## Future Work

- Extend to real-time prediction pipelines.
- Add different ML models (e.g., Gradient Boosting, Deep Learning).
- Integrate more features (soil, population risk).
- Visualize predictions on an interactive map.
