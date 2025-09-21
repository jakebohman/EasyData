"""
EasyPlots: Modular plotting functions for data visualization.

This module provides convenient wrapper functions for creating common plot types
with sensible defaults and easy customization options.

@author Jake Bohman
@date 7/23/2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Optional, Dict, Union, Tuple, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Default color scheme
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Common figure size presets
FIGURE_SIZES = {
    'small': (6, 4),
    'medium': (10, 6),
    'large': (12, 8),
    'square': (8, 8)
}

# Public API - functions available when using "from easyPlots import *"
__all__ = [
    'create_frequency_histogram',
    'create_boxplot', 
    'compare_datasets_histogram',
    'create_time_series_boxplot',
    'validate_dataframe_columns',
    'get_figure_size',
    'setup_color_scheme',
    # Legacy aliases for backwards compatibility
    'create_freqhist',
    'create_compare_freqhist',
    # Constants
    'DEFAULT_COLORS',
    'FIGURE_SIZES'
]

def create_frequency_histogram(
    data: pd.DataFrame,
    x_column: str,
    category_column: str,
    bins: int = 10,
    categories: Optional[List[str]] = None,
    category_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Frequency",
    x_range: Optional[Tuple[float, float]] = None,
    y_limit: float = 1.0,
    log_scale: Tuple[bool, bool] = (False, False),  # (x_log, y_log)
    figure_size: Union[str, Tuple[int, int]] = 'medium',
    show_legend: bool = True,
    colors: Optional[List[str]] = None
) -> None:
    """
    Create a frequency histogram with multiple categories.
    
    Args:
        data: DataFrame containing the data to plot
        x_column: Column name for the x-axis values
        category_column: Column name for categorical grouping
        bins: Number of histogram bins (default: 10)
        categories: List of categories to include (default: all unique values)
        category_labels: Dictionary mapping category values to display labels
        title: Plot title (default: auto-generated)
        xlabel: X-axis label (default: column name)
        ylabel: Y-axis label (default: "Frequency")
        x_range: Tuple of (min, max) for x-axis range (default: auto)
        y_limit: Maximum y-axis value (default: 1.0)
        log_scale: Tuple of (x_log, y_log) boolean flags for log scaling
        figure_size: Either preset name ('small', 'medium', 'large') or (width, height) tuple
        show_legend: Whether to display legend (default: True)
        colors: Custom color list (default: uses DEFAULT_COLORS)
        
    Raises:
        ValueError: If required columns don't exist in the DataFrame
        
    Example:
        >>> create_frequency_histogram(df, 'age', 'gender', bins=20, 
        ...                           title='Age Distribution by Gender')
    """
    # Validate inputs
    if x_column not in data.columns:
        raise ValueError(f"Column '{x_column}' not found in DataFrame")
    if category_column not in data.columns:
        raise ValueError(f"Column '{category_column}' not found in DataFrame")
    
    # Set up figure size
    if isinstance(figure_size, str):
        fig_size = FIGURE_SIZES.get(figure_size, FIGURE_SIZES['medium'])
    else:
        fig_size = figure_size
    
    plt.figure(figsize=fig_size)
    
    # Determine categories to plot
    if categories is None:
        categories = data[category_column].unique()
    
    # Set up colors
    plot_colors = colors if colors else DEFAULT_COLORS
    
    # Determine x-axis range
    if x_range is None:
        x_min, x_max = data[x_column].min(), data[x_column].max()
        x_range = (x_min, x_max)
    
    # Plot histogram for each category
    labels = []
    for i, category in enumerate(categories):
        category_data = data[data[category_column] == category]
        
        if len(category_data) == 0:
            logger.warning(f"No data found for category: {category}")
            continue
            
        # Calculate weights for frequency normalization
        weights = np.ones(len(category_data)) / len(category_data)
        
        # Determine label
        label = category_labels.get(category, category) if category_labels else category
        labels.append(label)
        
        # Plot histogram
        color = plot_colors[i % len(plot_colors)]
        plt.hist(
            category_data[x_column], 
            weights=weights, 
            bins=bins, 
            range=x_range,
            histtype='step',
            label=label,
            color=color,
            linewidth=2
        )
    
    # Apply log scales if requested
    x_log, y_log = log_scale
    if y_log:
        plt.yscale('log')
    if x_log:
        plt.xscale('log')
    
    # Set labels and title
    plt.title(title if title else f'Frequency Distribution of {x_column}')
    plt.xlabel(xlabel if xlabel else x_column.replace('_', ' ').title())
    plt.ylabel(ylabel)
    
    # Set limits
    plt.ylim(0, y_limit)
    if x_range:
        plt.xlim(x_range)
    
    # Add legend if requested
    if show_legend and labels:
        plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

# Frequency histogram but with multiple dataframes. Same as above but with lists of data, category, x_axis, category_list
def compare_datasets_histogram(
    datasets: List[pd.DataFrame],
    x_columns: List[str],
    category_columns: List[str],
    category_values: List[str],
    dataset_labels: List[str],
    bins: int = 10,
    x_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Frequency",
    y_limit: float = 1.0,
    log_scale: Tuple[bool, bool] = (False, False),
    figure_size: Union[str, Tuple[int, int]] = 'medium',
    colors: Optional[List[str]] = None
) -> None:
    """
    Compare histograms across multiple datasets.
    
    Args:
        datasets: List of DataFrames to compare
        x_columns: List of x-column names (one per dataset)
        category_columns: List of category column names (one per dataset)
        category_values: List of category values to filter by (one per dataset)
        dataset_labels: List of labels for each dataset
        bins: Number of histogram bins
        x_range: Tuple of (min, max) for x-axis range
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        y_limit: Maximum y-axis value
        log_scale: Tuple of (x_log, y_log) boolean flags
        figure_size: Either preset name or (width, height) tuple
        colors: Custom color list
    """
    if not (len(datasets) == len(x_columns) == len(category_columns) == 
            len(category_values) == len(dataset_labels)):
        raise ValueError("All input lists must have the same length")
    
    # Set up figure
    if isinstance(figure_size, str):
        fig_size = FIGURE_SIZES.get(figure_size, FIGURE_SIZES['medium'])
    else:
        fig_size = figure_size
    
    plt.figure(figsize=fig_size)
    
    # Set up colors
    plot_colors = colors if colors else DEFAULT_COLORS
    
    # Determine x-range if not provided
    if x_range is None:
        all_values = []
        for i, (df, x_col, cat_col, cat_val) in enumerate(
            zip(datasets, x_columns, category_columns, category_values)
        ):
            filtered_data = df[df[cat_col] == cat_val][x_col].dropna()
            all_values.extend(filtered_data.values)
        
        if all_values:
            x_range = (min(all_values), max(all_values))
        else:
            x_range = (0, 1)
    
    # Plot histogram for each dataset
    for i, (df, x_col, cat_col, cat_val, label) in enumerate(
        zip(datasets, x_columns, category_columns, category_values, dataset_labels)
    ):
        filtered_data = df[df[cat_col] == cat_val][x_col].dropna()
        
        if len(filtered_data) == 0:
            logger.warning(f"No data found for {label}")
            continue
        
        # Calculate weights for frequency normalization
        weights = np.ones(len(filtered_data)) / len(filtered_data)
        
        # Plot histogram
        color = plot_colors[i % len(plot_colors)]
        plt.hist(
            filtered_data,
            weights=weights,
            bins=bins,
            range=x_range,
            histtype='step',
            label=label,
            color=color,
            linewidth=2
        )
    
    # Apply log scales if requested
    x_log, y_log = log_scale
    if y_log:
        plt.yscale('log')
    if x_log:
        plt.xscale('log')
    
    # Set labels and title
    plt.title(title if title else 'Dataset Comparison')
    plt.xlabel(xlabel if xlabel else 'Value')
    plt.ylabel(ylabel)
    
    # Set limits
    plt.ylim(0, y_limit)
    if x_range:
        plt.xlim(x_range)
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def create_boxplot(
    data: pd.DataFrame,
    y_column: str,
    category_column: str,
    categories: Optional[List[str]] = None,
    category_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    y_range: Optional[Tuple[float, float]] = None,
    figure_size: Union[str, Tuple[int, int]] = 'square',
    x_tick_rotation: int = 0,
    show_means: bool = True,
    show_outliers: bool = False,
    add_reference_line: bool = False,
    reference_line_value: float = 0,
    colors: Optional[List[str]] = None
) -> None:
    """
    Create a boxplot comparing categories.
    
    Args:
        data: DataFrame containing the data to plot
        y_column: Column name for the y-axis values
        category_column: Column name for categorical grouping
        categories: List of categories to include (default: all unique values)
        category_labels: Dictionary mapping category values to display labels
        title: Plot title (default: auto-generated)
        xlabel: X-axis label (default: category column name)
        ylabel: Y-axis label (default: y column name)
        y_range: Tuple of (min, max) for y-axis range (default: auto)
        figure_size: Either preset name or (width, height) tuple
        x_tick_rotation: Rotation angle for x-axis labels (default: 0)
        show_means: Whether to show mean markers (default: True)
        show_outliers: Whether to show outlier points (default: False)
        add_reference_line: Whether to add horizontal reference line (default: False)
        reference_line_value: Y-value for reference line (default: 0)
        colors: Custom color list for boxes (default: uses DEFAULT_COLORS)
        
    Example:
        >>> create_boxplot(df, 'salary', 'department', 
        ...                title='Salary Distribution by Department')
    """
    # Validate inputs
    if y_column not in data.columns:
        raise ValueError(f"Column '{y_column}' not found in DataFrame")
    if category_column not in data.columns:
        raise ValueError(f"Column '{category_column}' not found in DataFrame")
    
    # Set up figure size
    if isinstance(figure_size, str):
        fig_size = FIGURE_SIZES.get(figure_size, FIGURE_SIZES['square'])
    else:
        fig_size = figure_size
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Determine categories to plot
    if categories is None:
        categories = sorted(data[category_column].unique())
    
    # Prepare data for boxplot
    data_to_plot = []
    xlabels = []
    
    for category in categories:
        category_data = data[data[category_column] == category][y_column].dropna()
        
        if len(category_data) == 0:
            logger.warning(f"No data found for category: {category}")
            continue
            
        data_to_plot.append(category_data)
        
        # Determine label
        label = category_labels.get(category, category) if category_labels else category
        xlabels.append(label)
    
    if not data_to_plot:
        raise ValueError("No valid data found for any categories")
    
    # Create boxplot
    bp = ax.boxplot(
        data_to_plot, 
        patch_artist=True, 
        notch=True,
        showfliers=show_outliers,
        showmeans=show_means,
        labels=xlabels
    )
    
    # Apply colors if provided
    if colors:
        plot_colors = colors
    else:
        plot_colors = DEFAULT_COLORS
        
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(plot_colors[i % len(plot_colors)])
        box.set_alpha(0.7)
    
    # Set labels and title
    plt.title(title if title else f'{y_column} by {category_column}')
    plt.xlabel(xlabel if xlabel else category_column.replace('_', ' ').title())
    plt.ylabel(ylabel if ylabel else y_column.replace('_', ' ').title())
    
    # Set y-axis limits
    if y_range:
        plt.ylim(y_range)
    
    # Rotate x-axis labels if needed
    if x_tick_rotation != 0:
        plt.xticks(rotation=x_tick_rotation)
    
    # Add reference line if requested
    if add_reference_line:
        plt.axhline(reference_line_value, linestyle='--', color='gray', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def create_time_series_boxplot(
    data: pd.DataFrame,
    y_column: str,
    datetime_column: str,
    time_grouping: str = 'month',
    filter_column: Optional[str] = None,
    filter_value: Optional[Any] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    y_range: Optional[Tuple[float, float]] = None,
    figure_size: Union[str, Tuple[int, int]] = 'medium',
    x_tick_rotation: int = 0,
    show_means: bool = True,
    show_outliers: bool = False,
    date_format: str = '%Y-%m-%d'
) -> None:
    """
    Create a boxplot showing time-series data grouped by time periods.
    
    Args:
        data: DataFrame containing the data
        y_column: Column name for the y-axis values
        datetime_column: Column name containing datetime information
        time_grouping: How to group time data ('hour', 'day', 'month', 'year')
        filter_column: Optional column to filter data by
        filter_value: Value to filter by if filter_column is specified
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        y_range: Tuple of (min, max) for y-axis range
        figure_size: Either preset name or (width, height) tuple
        x_tick_rotation: Rotation angle for x-axis labels
        show_means: Whether to show mean markers
        show_outliers: Whether to show outlier points
        date_format: Format string for parsing dates
        
    Example:
        >>> create_time_series_boxplot(df, 'price', 'date', 
        ...                           time_grouping='month', 
        ...                           title='Price by Month')
    """
    # Validate inputs
    if y_column not in data.columns:
        raise ValueError(f"Column '{y_column}' not found in DataFrame")
    if datetime_column not in data.columns:
        raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
    
    # Filter data if requested
    plot_data = data.copy()
    if filter_column and filter_value:
        if filter_column not in data.columns:
            raise ValueError(f"Filter column '{filter_column}' not found in DataFrame")
        plot_data = plot_data[plot_data[filter_column] == filter_value]
        
    if len(plot_data) == 0:
        raise ValueError("No data remains after filtering")
    
    # Convert datetime column to datetime if it's not already
    try:
        plot_data[datetime_column] = pd.to_datetime(plot_data[datetime_column], format=date_format)
    except:
        # Try automatic parsing
        plot_data[datetime_column] = pd.to_datetime(plot_data[datetime_column])
    
    # Group data by time period
    if time_grouping == 'hour':
        plot_data['time_group'] = plot_data[datetime_column].dt.hour
        xlabel = xlabel or 'Hour'
    elif time_grouping == 'day':
        plot_data['time_group'] = plot_data[datetime_column].dt.day
        xlabel = xlabel or 'Day'
    elif time_grouping == 'month':
        plot_data['time_group'] = plot_data[datetime_column].dt.month
        xlabel = xlabel or 'Month'
    elif time_grouping == 'year':
        plot_data['time_group'] = plot_data[datetime_column].dt.year
        xlabel = xlabel or 'Year'
    else:
        raise ValueError("time_grouping must be one of: 'hour', 'day', 'month', 'year'")
    
    # Set up figure
    if isinstance(figure_size, str):
        fig_size = FIGURE_SIZES.get(figure_size, FIGURE_SIZES['medium'])
    else:
        fig_size = figure_size
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Prepare data for boxplot
    unique_groups = sorted(plot_data['time_group'].unique())
    data_to_plot = []
    
    for group in unique_groups:
        group_data = plot_data[plot_data['time_group'] == group][y_column].dropna()
        data_to_plot.append(group_data)
    
    # Create boxplot
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        notch=True,
        showfliers=show_outliers,
        showmeans=show_means,
        labels=unique_groups
    )
    
    # Style the boxes
    for i, box in enumerate(bp['boxes']):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        box.set_facecolor(color)
        box.set_alpha(0.7)
    
    # Set labels and title
    plt.title(title if title else f'{y_column} by {time_grouping.title()}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else y_column.replace('_', ' ').title())
    
    # Set y-axis limits
    if y_range:
        plt.ylim(y_range)
    
    # Rotate x-axis labels if needed
    if x_tick_rotation != 0:
        plt.xticks(rotation=x_tick_rotation)
    
    plt.tight_layout()
    plt.show()


# Utility functions for data validation and preparation
def validate_dataframe_columns(data: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate that all required columns exist in the DataFrame."""
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def get_figure_size(size_spec: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert size specification to figure size tuple."""
    if isinstance(size_spec, str):
        return FIGURE_SIZES.get(size_spec, FIGURE_SIZES['medium'])
    return size_spec


def setup_color_scheme(colors: Optional[List[str]], n_colors: int) -> List[str]:
    """Set up color scheme for plots."""
    if colors:
        return colors
    # Repeat default colors if we need more than available
    return (DEFAULT_COLORS * ((n_colors // len(DEFAULT_COLORS)) + 1))[:n_colors]


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================
# These aliases maintain compatibility with existing code that uses old function names

def create_freqhist(*args, **kwargs):
    """
    Deprecated: Use create_frequency_histogram() instead.
    This function is maintained for backwards compatibility.
    """
    logger.warning("create_freqhist() is deprecated. Use create_frequency_histogram() instead.")
    
    # Map old parameters to new ones
    if len(args) >= 3:
        data, category, x_axis = args[:3]
        new_kwargs = {
            'data': data,
            'x_column': x_axis,
            'category_column': category
        }
        
        # Map common old kwargs to new ones
        old_to_new_mapping = {
            'category_list': 'categories',
            'x_axis': 'x_column',
            'category': 'category_column',
            'xlim': None,  # Will be handled specially
            'xlower': None,  # Will be handled specially
            'ylim': 'y_limit',
            'fsize': 'figure_size',
            'ylog': None,  # Will be handled specially
            'xlog': None   # Will be handled specially
        }
        
        # Handle special cases
        xlower = kwargs.get('xlower', 0)
        xlim = kwargs.get('xlim', 50)
        if xlower != 0 or xlim != 50:
            new_kwargs['x_range'] = (xlower, xlim)
            
        ylog = kwargs.get('ylog', False)
        xlog = kwargs.get('xlog', False)
        if ylog or xlog:
            new_kwargs['log_scale'] = (xlog, ylog)
            
        fsize = kwargs.get('fsize', [10, 6])
        if fsize != [10, 6]:
            new_kwargs['figure_size'] = tuple(fsize)
            
        # Map other parameters
        for old_key, new_key in old_to_new_mapping.items():
            if old_key in kwargs and new_key:
                new_kwargs[new_key] = kwargs[old_key]
        
        # Handle remaining parameters
        for key in ['bins', 'title', 'xlabel', 'ylabel', 'category_labels']:
            if key in kwargs:
                new_kwargs[key] = kwargs[key]
                
        return create_frequency_histogram(**new_kwargs)
    else:
        raise ValueError("create_freqhist requires at least 3 positional arguments")


# Legacy function name alias
create_compare_freqhist = compare_datasets_histogram