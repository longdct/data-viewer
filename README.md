# HuggingFace Dataset Viewer

A simple web application built with Flask for viewing HuggingFace datasets with pagination support.

## Features

- Load and view datasets directly from HuggingFace Hub
- Support for dataset subsets and different splits (train/test/validation)
- Pagination with configurable rows per page (10, 20, or 50)
- **Column Visibility Controls:** Hide/show specific columns with checkbox controls
- **Data Filtering:** Filter rows by column values (SQL WHERE clause functionality)
- **Improved Text Display:** Proper rendering of escape characters (\\n, \\t) in table cells
- Clean, responsive interface using Bootstrap
- Built with Python/Flask for easy extensibility to other data types
- Automatic conversion from HuggingFace datasets to pandas DataFrames

## Setup and Installation

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the application:**
   ```bash
   uv run python app.py
   ```

3. **Access the application:**
   Open your web browser and navigate to `http://localhost:5000`

## Usage

1. **Load HuggingFace Dataset:** Enter a dataset path (e.g., "imdb", "squad", "glue/cola")
2. **Optional Configuration:** 
   - Specify a subset name if the dataset has multiple configurations
   - Choose the split (train, test, validation, dev)
3. **View Data:** After loading, the data will be displayed in a paginated table
4. **Column Visibility:** Use the "Column Visibility Controls" to show/hide specific columns:
   - Check/uncheck individual columns
   - Use "Select All" or "Deselect All" for bulk operations
   - Click "Apply" to update the table
5. **Filter Data:** Use the "Filter Data (WHERE clause)" to filter rows:
   - Select a column from the dropdown
   - Enter a value to filter by
   - Click "Apply Filter" to show only matching rows
   - Click "Clear Filter" to remove all filters
6. **Navigate Pages:** Use the pagination controls at the bottom to navigate through pages
7. **Change Page Size:** Use the dropdown to change how many rows are displayed per page (10, 20, or 50)
8. **Load New Dataset:** Click "Load New Dataset" to load a different dataset
9. **Clear Data:** Click "Clear Data" to remove the current dataset from memory

### Popular Datasets to Try:
- **imdb** - Movie reviews for sentiment analysis
- **squad** - Reading comprehension dataset  
- **glue/cola** - Grammar acceptability classification
- **wikitext/wikitext-2-raw-v1** - Language modeling dataset
- **amazon_reviews_multi/en** - Amazon product reviews

## Project Structure

```
data_viewer/
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   ├── base.html       # Base template with navigation
│   ├── index.html      # Home page with dataset loading
│   └── view.html       # Data viewing page with pagination
├── requirements.txt    # Python dependencies
├── sample_data.csv     # Sample CSV file for reference
└── README.md          # This file
```

## Extending the Application

This application is designed to be easily extensible for other Python data types:

- **Add new data sources:** Extend beyond HuggingFace to support local files, databases, APIs
- **Support other formats:** Add support for Arrow, Parquet, JSON datasets
- **Add data processing:** Integrate with NumPy, SciPy, or other scientific Python libraries
- **Add visualizations:** Integrate matplotlib, plotly, or other visualization libraries
- **Add filtering/search:** Implement data filtering and search capabilities
- **Database support:** Add SQLAlchemy for database connectivity

## Technical Notes

- Uses HuggingFace `datasets` library for data loading
- Automatic conversion to pandas DataFrames for easy manipulation
- Session-based dataset storage (datasets are kept in memory during the session)
- Bootstrap 5 for responsive UI components
- Error handling for invalid dataset paths or loading failures
- Support for dataset subsets and different splits
