# EasyData
Easy-to-use utilities for data loading and visualization with pandas and matplotlib.

A collection of pyplot wrapper functions that simplify common data analysis tasks, developed for research at Swarthmore College and improved for my personal use.

## Features

### Data Loading (`easyData.py`)
- **`load_jsons()`**: Load and combine multiple JSON files into a single DataFrame
- **`load_csvs()`**: Load and combine multiple CSV files into a single DataFrame
- Automatic error handling and progress logging
- File source tracking for data provenance
- Support for both JSON Lines and regular JSON formats

### Data Visualization (`easyPlots.py`)
- **`create_frequency_histogram()`**: Multi-category frequency histograms with extensive customization
- **`create_boxplot()`**: Category comparison boxplots with statistical overlays
- **`create_time_series_boxplot()`**: Time-based data analysis with flexible grouping
- **`compare_datasets_histogram()`**: Cross-dataset comparisons

## Sample Usage

```
# Load data
df = load_jsons('/path/to/data/*.json')

# Create visualizations
create_frequency_histogram(df, 'age', 'gender', bins=20)
create_boxplot(df, 'salary', 'department', show_outliers=True)
```