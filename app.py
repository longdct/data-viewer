import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datasets import load_dataset
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables to store current dataset data
current_data = None
current_dataset_name = None

@app.route('/')
def index():
    return render_template('index.html', dataset_name=current_dataset_name)

@app.route('/load_dataset', methods=['POST'])
def load_hf_dataset():
    global current_data, current_dataset_name
    
    dataset_path = request.form.get('dataset_path', '').strip()
    subset_name = request.form.get('subset_name', '').strip()
    split_name = request.form.get('split_name', 'train').strip()
    
    if not dataset_path:
        flash('Please enter a HuggingFace dataset path')
        return redirect(url_for('index'))
    
    try:
        # Load HuggingFace dataset
        if subset_name:
            dataset = load_dataset(dataset_path, subset_name, split=split_name)
        else:
            dataset = load_dataset(dataset_path, split=split_name)
        
        # Convert to pandas DataFrame
        current_data = dataset.to_pandas()
        current_dataset_name = f"{dataset_path}" + (f"/{subset_name}" if subset_name else "") + f" ({split_name})"
        
        flash(f'Successfully loaded dataset: {current_dataset_name}')
        flash(f'Dataset shape: {current_data.shape[0]} rows, {current_data.shape[1]} columns')
        return redirect(url_for('view_data'))
        
    except Exception as e:
        flash(f'Error loading HuggingFace dataset: {str(e)}')
        flash('Please check the dataset path and try again. Example: "imdb", "squad", "glue/cola"')
        return redirect(url_for('index'))

@app.route('/view')
def view_data():
    global current_data, current_dataset_name
    
    if current_data is None:
        flash('No dataset loaded. Please load a HuggingFace dataset first.')
        return redirect(url_for('index'))
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Get column visibility parameters
    hidden_columns = request.args.getlist('hidden_columns')
    
    # Get filtering parameters
    filter_column = request.args.get('filter_column', '')
    filter_value = request.args.get('filter_value', '')
    
    # Validate per_page options
    if per_page not in [10, 20, 50]:
        per_page = 10
    
    # Start with original data
    display_data = current_data.copy()
    
    # Apply column filtering (WHERE clause functionality)
    if filter_column and filter_value and filter_column in display_data.columns:
        # Convert filter value to appropriate type for comparison
        try:
            # Try to match the column's dtype
            if display_data[filter_column].dtype == 'object':
                # For string columns, do case-insensitive partial matching
                display_data = display_data[display_data[filter_column].astype(str).str.contains(filter_value, case=False, na=False)]
            else:
                # For numeric columns, try exact matching
                try:
                    numeric_filter = pd.to_numeric(filter_value)
                    display_data = display_data[display_data[filter_column] == numeric_filter]
                except:
                    # If conversion fails, convert column to string and do partial matching
                    display_data = display_data[display_data[filter_column].astype(str).str.contains(filter_value, case=False, na=False)]
        except Exception as e:
            flash(f'Error applying filter: {str(e)}')
    
    # Apply column hiding
    if hidden_columns:
        visible_columns = [col for col in display_data.columns if col not in hidden_columns]
        display_data = display_data[visible_columns]
    
    # Calculate pagination based on filtered data
    total_rows = len(display_data)
    total_pages = math.ceil(total_rows / per_page) if total_rows > 0 else 1
    
    # Ensure page is within valid range
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages if total_pages > 0 else 1
    
    # Get data for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_data = display_data.iloc[start_idx:end_idx]
    
    # Process escape characters for better display
    def format_cell_content(val):
        if pd.isna(val):
            return val
        if isinstance(val, str):
            # Replace escape characters with visible representations
            val = val.replace('\\n', '<br>')  # Line breaks
            val = val.replace('\\t', '&nbsp;&nbsp;&nbsp;&nbsp;')  # Tabs as spaces
            val = val.replace('\n', '<br>')  # Actual newlines
            val = val.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')  # Actual tabs
        return val
    
    # Apply formatting to all cells
    formatted_data = page_data.copy()
    for col in formatted_data.columns:
        formatted_data[col] = formatted_data[col].apply(format_cell_content)
    
    # Convert to HTML table format
    table_html = formatted_data.to_html(classes='table table-striped table-bordered', 
                                       table_id='dataset-table', 
                                       escape=False,
                                       index=True)
    
    return render_template('view.html', 
                         table_html=table_html,
                         dataset_name=current_dataset_name,
                         page=page,
                         per_page=per_page,
                         total_pages=total_pages,
                         total_rows=total_rows,
                         original_total_rows=len(current_data),
                         start_row=start_idx + 1,
                         end_row=min(end_idx, total_rows),
                         all_columns=list(current_data.columns),
                         hidden_columns=hidden_columns,
                         filter_column=filter_column,
                         filter_value=filter_value)

@app.route('/api/search', methods=['GET', 'POST'])
def search_data():
    """
    Search API that performs exact matching on a specified field.
    
    GET Usage: GET /api/search?field=column_name&value=search_value&dataset_path=huggingface_dataset
    POST Usage: POST /api/search with JSON body containing field, value, dataset_path, etc.
    
    Parameters:
    - field: Column name to search in (required)
    - value: Value to search for (required)
    - dataset_path: HuggingFace dataset path (optional)
    - subset_name: Dataset subset name (optional)
    - split_name: Dataset split (defaults to 'train')
    
    Returns: JSON array of matching records
    """
    global current_data, current_dataset_name
    
    # Get parameters based on request method
    if request.method == 'POST':
        # For POST requests, get parameters from JSON body
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'error': 'Invalid JSON',
                    'message': 'POST request must contain valid JSON in the request body'
                }), 400
            
            dataset_path = data.get('dataset_path', '').strip()
            subset_name = data.get('subset_name', '').strip()
            split_name = data.get('split_name', 'train').strip()
            field = data.get('field', '').strip()
            value = data.get('value', '').strip()
        except Exception as e:
            return jsonify({
                'error': 'Invalid request format',
                'message': f'Error parsing JSON: {str(e)}'
            }), 400
    else:
        # For GET requests, get parameters from query string
        dataset_path = request.args.get('dataset_path', '').strip()
        subset_name = request.args.get('subset_name', '').strip()
        split_name = request.args.get('split_name', 'train').strip()
        field = request.args.get('field', '').strip()
        value = request.args.get('value', '').strip()
    
    # Determine which dataset to use
    search_data = None
    dataset_info = None
    
    if dataset_path:
        # Load HuggingFace dataset on-demand
        try:
            if subset_name:
                dataset = load_dataset(dataset_path, subset_name, split=split_name)
            else:
                dataset = load_dataset(dataset_path, split=split_name)
            
            search_data = dataset.to_pandas()
            dataset_info = f"{dataset_path}" + (f"/{subset_name}" if subset_name else "") + f" ({split_name})"
            
        except Exception as e:
            return jsonify({
                'error': 'Failed to load HuggingFace dataset',
                'message': f'Error loading dataset "{dataset_path}": {str(e)}',
                'suggestion': 'Please check the dataset path, subset name, and split name. Examples: "imdb", "squad", "glue/cola"'
            }), 400
    else:
        # Use currently loaded dataset
        if current_data is None:
            return jsonify({
                'error': 'No dataset available',
                'message': 'Please either load a dataset first or provide a dataset_path parameter'
            }), 400
        search_data = current_data
        dataset_info = current_dataset_name
    
    # Validate parameters
    if not field:
        return jsonify({
            'error': 'Missing field parameter',
            'message': 'Please specify a field to search in using the "field" parameter'
        }), 400
    
    if not value:
        return jsonify({
            'error': 'Missing value parameter',
            'message': 'Please specify a value to search for using the "value" parameter'
        }), 400
    
    # Check if field exists in dataset
    if field not in search_data.columns:
        return jsonify({
            'error': 'Field not found',
            'message': f'Field "{field}" does not exist in the dataset',
            'available_fields': list(search_data.columns),
            'dataset': dataset_info
        }), 400
    
    try:
        # Perform exact matching
        # Convert both the column and search value to string for exact comparison
        # This handles different data types consistently
        mask = search_data[field].astype(str) == value
        matching_records = search_data[mask]
        
        # Convert to list of dictionaries for JSON response
        results = matching_records.to_dict('records')
        
        return jsonify({
            'success': True,
            'dataset': dataset_info,
            'field': field,
            'value': value,
            'count': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Search failed',
            'message': f'An error occurred while searching: {str(e)}',
            'dataset': dataset_info
        }), 500

@app.route('/clear')
def clear_data():
    global current_data, current_dataset_name
    current_data = None
    current_dataset_name = None
    flash('Dataset cleared successfully')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2706)
