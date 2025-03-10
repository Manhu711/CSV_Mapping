import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import io
import plotly.express as px
import json
from datetime import datetime
import uuid

# Page configuration
st.set_page_config(
    page_title="Advanced CSV Data Mapper",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .file-list {
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    .file-container {
        max-height: 300px;
        overflow-y: auto;
        padding-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load the model once and store in session state
@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Initialize session state for file management
if 'source_files_dict' not in st.session_state:
    st.session_state.source_files_dict = {}  # {file_id: {"name": name, "data": dataframe}}

if 'destination_files_dict' not in st.session_state:
    st.session_state.destination_files_dict = {}  # {file_id: {"name": name, "data": dataframe, "standard": data_standard}}

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()  # Set of processed file_ids

if 'current_source_id' not in st.session_state:
    st.session_state.current_source_id = None

if 'current_destination_id' not in st.session_state:
    st.session_state.current_destination_id = None

if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Define a default confidence threshold (since we removed the slider)
CONFIDENCE_THRESHOLD = 50  # 50% confidence threshold for automatic mapping

# Function to infer data type from pandas dtype
def infer_data_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif pd.api.types.is_datetime64_dtype(dtype):
        return "datetime"
    else:
        return "string"

# Function to extract data standard from destination file
def extract_data_standard_from_file(df):
    data_standard = {}
    
    for column in df.columns:
        # Infer data type from column values
        dtype = df[column].dtype
        data_type = infer_data_type(dtype)
        
        # Generate a description based on column name and data
        sample_values = df[column].dropna().head(3).tolist()
        sample_str = ", ".join([str(val) for val in sample_values])
        if len(sample_str) > 50:
            sample_str = sample_str[:47] + "..."
        
        description = f"Field containing {column} data. Sample values: {sample_str}"
        
        # Add to data standard
        data_standard[column] = {
            "type": data_type,
            "description": description
        }
    
    return data_standard

# Function to upload source files
def upload_source_file():
    uploaded_files = st.file_uploader("Upload Source Files", type=['csv'], accept_multiple_files=True, key="source_uploader")
    if uploaded_files:
        # Track existing filenames
        existing_filenames = {info["name"] for info in st.session_state.source_files_dict.values()}
        new_files_added = False
        
        for uploaded_file in uploaded_files:
            # Skip if file with same name already exists, but don't show warning
            if uploaded_file.name in existing_filenames:
                continue
                
            file_id = str(uuid.uuid4())
            df = pd.read_csv(uploaded_file)
            st.session_state.source_files_dict[file_id] = {
                "name": uploaded_file.name,
                "data": df,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            existing_filenames.add(uploaded_file.name)
            new_files_added = True
        
        if new_files_added:
            st.success("Source files uploaded successfully!")

# Function to upload destination files
def upload_destination_file():
    uploaded_files = st.file_uploader("Upload Destination Files", type=['csv'], accept_multiple_files=True, key="destination_uploader")
    if uploaded_files:
        # Track existing filenames
        existing_filenames = {info["name"] for info in st.session_state.destination_files_dict.values()}
        new_files_added = False
        
        for uploaded_file in uploaded_files:
            # Skip if file with same name already exists, but don't show warning
            if uploaded_file.name in existing_filenames:
                continue
                
            file_id = str(uuid.uuid4())
            df = pd.read_csv(uploaded_file)
            data_standard = extract_data_standard_from_file(df)
            st.session_state.destination_files_dict[file_id] = {
                "name": uploaded_file.name,
                "data": df,
                "standard": data_standard,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            existing_filenames.add(uploaded_file.name)
            new_files_added = True
        
        if new_files_added:
            st.success("Destination files uploaded successfully!")

# Function to display file selection interface
def display_file_selection():
    st.markdown("### Select Files for Mapping")
    
    # Select source file - show all files, including processed ones
    source_options = {file_id: info["name"] for file_id, info in st.session_state.source_files_dict.items()}
    if source_options:
        st.session_state.current_source_id = st.selectbox("Select Source File", options=list(source_options.keys()), 
                                                         format_func=lambda x: f"{source_options[x]} {'(Processed)' if x in st.session_state.processed_files else ''}")
    else:
        st.warning("No available source files. Please upload more.")

    # Select destination file
    destination_options = {file_id: info["name"] for file_id, info in st.session_state.destination_files_dict.items()}
    if destination_options:
        st.session_state.current_destination_id = st.selectbox("Select Destination File", options=list(destination_options.keys()), format_func=lambda x: destination_options[x])
    else:
        st.warning("No available destination files. Please upload more.")

def get_best_match(column_name, column_dtype, standard_fields, data_standard):
    """Find the best matching standard field using word embeddings, cosine similarity, and data type compatibility"""
    # Get embeddings
    column_embedding = st.session_state.model.encode([column_name.lower()])
    field_embeddings = st.session_state.model.encode([field.lower() for field in standard_fields])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(column_embedding, field_embeddings)[0]
    
    # Get data type of the column
    column_type = infer_data_type(column_dtype)
    
    # Adjust similarity scores based on data type compatibility
    adjusted_similarities = similarities.copy()
    for i, field in enumerate(standard_fields):
        field_type = data_standard[field]["type"]
        
        # If data types match, boost similarity
        if column_type == field_type:
            adjusted_similarities[i] += 0.2  # Boost by 20%
        # If data types are incompatible, reduce similarity
        elif (column_type == "numeric" and field_type == "string") or \
             (column_type == "string" and field_type == "numeric") or \
             (column_type == "string" and field_type == "datetime"):
            adjusted_similarities[i] -= 0.1  # Reduce by 10%
    
    # Ensure scores are within [0, 1] range
    adjusted_similarities = np.clip(adjusted_similarities, 0, 1)
    
    # Get best match
    best_match_idx = np.argmax(adjusted_similarities)
    best_match_score = adjusted_similarities[best_match_idx] * 100  # Convert to percentage
    
    return list(standard_fields)[best_match_idx], best_match_score

def validate_data_type(series, expected_type):
    """Validate if the data matches the expected type"""
    try:
        if expected_type == "datetime":
            pd.to_datetime(series)
            return True
        elif expected_type == "numeric":
            pd.to_numeric(series)
            return True
        elif expected_type == "string":
            return True
        return False
    except:
        return False

def generate_mapped_dataframe(df, mapping_data):
    """Generate new DataFrame with mapped column names"""
    new_df = df.copy()
    
    # Create mapping dictionary, keeping only the first occurrence of each mapped field
    seen_fields = set()
    column_mapping = {}
    duplicate_mappings = []
    
    for row in mapping_data:
        orig_col = row["Column of Uploaded CSV"]
        mapped_field = row["Mapped Field of Data Standard"]
        
        # Only include columns that exist in the DataFrame and aren't unmapped
        if mapped_field != "Unmapped" and orig_col in df.columns:
            if mapped_field not in seen_fields:
                column_mapping[orig_col] = mapped_field
                seen_fields.add(mapped_field)
            else:
                duplicate_mappings.append((orig_col, mapped_field))
    
    # Show warning for duplicate mappings
    if duplicate_mappings:
        warning_msg = "Warning: Multiple columns mapped to the same standard field. Only the first mapping will be used:\n"
        for orig_col, mapped_field in duplicate_mappings:
            warning_msg += f"\n- '{orig_col}' also mapped to '{mapped_field}' (ignored)"
        st.warning(warning_msg)
    
    # Check if we have any valid columns to map
    if not column_mapping:
        st.error("No valid mappings found. Please map at least one column.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Only keep columns that are mapped and exist in the DataFrame
    valid_columns = [col for col in column_mapping.keys() if col in df.columns]
    
    if not valid_columns:
        st.error("None of the mapped columns exist in the source file.")
        return pd.DataFrame()  # Return empty DataFrame
    
    new_df = new_df[valid_columns]
    
    # Rename columns according to mapping
    new_df = new_df.rename(columns=column_mapping)
    return new_df

def analyze_data_quality(df, mapping_data, data_standard):
    """Analyze data quality based on expected types"""
    quality_report = []
    
    for row in mapping_data:
        col_name = row["Column of Uploaded CSV"]
        mapped_field = row["Mapped Field of Data Standard"]
        
        if mapped_field != "Unmapped":
            expected_type = data_standard[mapped_field]["type"]
            is_valid = validate_data_type(df[col_name], expected_type)
            missing_pct = (df[col_name].isna().sum() / len(df)) * 100
            
            quality_report.append({
                "Column": col_name,
                "Mapped Field": mapped_field,
                "Expected Type": expected_type,
                "Type Valid": is_valid,
                "Missing Values (%)": round(missing_pct, 2),
                "Unique Values": df[col_name].nunique()
            })
    
    return pd.DataFrame(quality_report)

def visualize_data_quality(quality_df):
    """Create visualizations for data quality"""
    # Missing values chart
    fig1 = px.bar(
        quality_df, 
        x="Column", 
        y="Missing Values (%)", 
        color="Mapped Field",
        title="Missing Values by Column",
        labels={"Missing Values (%)": "Percentage of Missing Values"},
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Type validity chart
    type_validity = quality_df.groupby("Expected Type")["Type Valid"].value_counts().unstack().fillna(0)
    fig2 = px.bar(
        type_validity, 
        title="Data Type Validity by Expected Type",
        labels={"value": "Count", "Expected Type": "Expected Data Type"},
        height=400,
        barmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)

def save_mapping_template(mapping_data):
    """Save the current mapping as a template"""
    template = {}
    for row in mapping_data:
        if row["Mapped Field of Data Standard"] != "Unmapped":
            template[row["Column of Uploaded CSV"]] = row["Mapped Field of Data Standard"]
    return template

def load_mapping_template(template, columns, df, data_standard):
    """Load a mapping template and apply it to the current dataset"""
    mapping_data = []
    for col in columns:
        mapped_field = template.get(col, "Unmapped")
        sample_data = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No data"
        similarity_score = 100 if mapped_field != "Unmapped" else 0
        mapping_data.append({
            "Column of Uploaded CSV": col,
            "Sample Data": str(sample_data),
            "Mapped Field of Data Standard": mapped_field,
            "Confidence Score": similarity_score
        })
    return mapping_data

def add_custom_field(data_standard):
    """Add a custom field to the data standard"""
    with st.expander("Add Custom Field to Data Standard"):
        col1, col2 = st.columns(2)
        
        with col1:
            field_name = st.text_input("Field Name", placeholder="Enter field name")
        
        with col2:
            field_type = st.selectbox("Field Type", options=["string", "numeric", "datetime"])
        
        field_description = st.text_area("Field Description", placeholder="Enter field description")
        
        if st.button("Add Field") and field_name and field_description:
            # Check if field already exists
            if field_name in data_standard:
                st.warning(f"Field '{field_name}' already exists in the data standard.")
            else:
                # Add new field to data standard
                data_standard[field_name] = {
                    "type": field_type,
                    "description": field_description
                }
                st.success(f"Field '{field_name}' added to data standard.")
    
    return data_standard

def main():
    st.title("CSV Data Field Mapping")

    # Check if we need to reset for the next mapping
    if st.session_state.get('reset_for_next', False):
        # Display a success message about the last processed file
        if 'last_processed_file' in st.session_state:
            st.success(f"File '{st.session_state.last_processed_file['name']}' was successfully processed at {st.session_state.last_processed_file['time']} with {st.session_state.last_processed_file['mapped_columns']} mapped columns.")
        
        # Clear mapping data and analysis state for next mapping
        st.session_state.mapping_data = None
        st.session_state.show_analysis = False
        st.session_state.current_source_id = None  # Reset source file selection
        st.session_state.reset_for_next = False
        
        # Add a button to continue to next mapping
        if st.button("Continue to Next Mapping", type="primary"):
            st.rerun()

    # Sidebar for file management
    with st.sidebar:
        st.markdown("## File Management")
        
        # Create tabs for source and destination file management
        file_tabs = st.tabs(["Source Files", "Destination Files"])
        
        with file_tabs[0]:
            # Upload source files
            upload_source_file()
        
        with file_tabs[1]:
            # Upload destination files
            upload_destination_file()
        
        # Reset button
        st.markdown("---")
        if st.button("Reset All Data", type="primary"):
            for key in ['source_files_dict', 'destination_files_dict', 'processed_files', 
                       'current_source_id', 'current_destination_id', 'mapping_data', 
                       'show_analysis', 'last_processed_file', 'reset_for_next']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("All data has been reset!")
            st.rerun()
    
    # Initialize session state for editing
    if 'editing_row' not in st.session_state:
        st.session_state.editing_row = None
    if 'mapping_data' not in st.session_state:
        st.session_state.mapping_data = None
    if 'mapping_templates' not in st.session_state:
        st.session_state.mapping_templates = {}
    
    # Main content area
    st.markdown("## File Selection")
    
    # Display file selection interface with improved layout
    col1, col2 = st.columns(2)

    with col1:
        # Select source file
        source_options = {file_id: info["name"] for file_id, info in st.session_state.source_files_dict.items()}
        if source_options:
            st.session_state.current_source_id = st.selectbox(
                "Select Source File", 
                options=list(source_options.keys()), 
                format_func=lambda x: f"{source_options[x]} {'(Processed)' if x in st.session_state.processed_files else ''}"
            )
        else:
            st.warning("No available source files. Please upload more.")

    with col2:
        # Select destination file
        destination_options = {file_id: info["name"] for file_id, info in st.session_state.destination_files_dict.items()}
        if destination_options:
            st.session_state.current_destination_id = st.selectbox(
                "Select Destination File", 
                options=list(destination_options.keys()), 
                format_func=lambda x: destination_options[x]
            )
        else:
            st.warning("No available destination files. Please upload more.")

    # Only proceed if both source and destination are selected
    if st.session_state.current_source_id and st.session_state.current_destination_id:
        # Get the current source file
        source_id = st.session_state.current_source_id
        df = st.session_state.source_files_dict[source_id]["data"]
        
        # Get the current destination file's data standard
        destination_id = st.session_state.current_destination_id
        data_standard = st.session_state.destination_files_dict[destination_id]["standard"]
        dest_df = st.session_state.destination_files_dict[destination_id]["data"]
        
        # Store data standard in session state for custom field addition
        st.session_state.data_standard = data_standard
        
        # Add a button to start the mapping process
        if st.button("Begin Analysis and Mapping", type="primary", use_container_width=True):
            st.session_state.show_analysis = True
        
        # Only show analysis and mapping if the button has been clicked
        if st.session_state.get('show_analysis', False):
            # Display source and destination previews side by side
            st.markdown('<h2 class="sub-header">Data Preview</h2>', unsafe_allow_html=True)
            
            # Create two columns for side-by-side preview
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                # Source file preview
                st.markdown('<h3 class="sub-header">Source Dataset</h3>', unsafe_allow_html=True)
                
                # Basic metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Rows", df.shape[0])
                with metric_col2:
                    st.metric("Columns", df.shape[1])
                with metric_col3:
                    st.metric("Missing Values", df.isna().sum().sum())
                
                # Data sample (without the label)
                st.dataframe(df.head(5), use_container_width=True)
                
                # Column info in an expander
                with st.expander("Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isna().sum(),
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
            
            with preview_col2:
                # Destination file preview
                st.markdown('<h3 class="sub-header">Destination Dataset</h3>', unsafe_allow_html=True)
                
                # Basic metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Rows", dest_df.shape[0])
                with metric_col2:
                    st.metric("Columns", dest_df.shape[1])
                with metric_col3:
                    st.metric("Missing Values", dest_df.isna().sum().sum())
                
                # Data sample (without the label)
                st.dataframe(dest_df.head(5), use_container_width=True)
                
                # Column info in an expander
                with st.expander("Column Information"):
                    col_info = pd.DataFrame({
                        'Column': dest_df.columns,
                        'Type': dest_df.dtypes,
                        'Non-Null Count': dest_df.count(),
                        'Null Count': dest_df.isna().sum(),
                        'Unique Values': [dest_df[col].nunique() for col in dest_df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)

            # Column mapping section
            st.markdown('<h2 class="sub-header">Column Mapping</h2>', unsafe_allow_html=True)

            # Start mapping button
            if st.button("Start Column Mapping") or st.session_state.mapping_data is not None:
                # Create mapping table if it doesn't exist
                if st.session_state.mapping_data is None:
                    mapping_data = []
                    for col in df.columns:
                        # Get sample data (first non-null value)
                        sample_data = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No data"
                        
                        # Get mapped field and similarity score using word embeddings and data type compatibility
                        mapped_field, similarity_score = get_best_match(col, df[col].dtype, data_standard.keys(), data_standard)
                        
                        # Get data type information
                        source_type = infer_data_type(df[col].dtype)
                        dest_type = data_standard[mapped_field]["type"] if mapped_field != "Unmapped" else "N/A"
                        type_match = source_type == dest_type if mapped_field != "Unmapped" else False
                        
                        mapping_data.append({
                            "Column of Uploaded CSV": col,
                            "Sample Data": str(sample_data),
                            "Source Type": source_type,
                            "Mapped Field of Data Standard": mapped_field if similarity_score > CONFIDENCE_THRESHOLD else "Unmapped",
                            "Destination Type": dest_type if similarity_score > CONFIDENCE_THRESHOLD else "N/A",
                            "Type Match": type_match if similarity_score > CONFIDENCE_THRESHOLD else False,
                            "Confidence Score": round(similarity_score, 2)
                        })
                    st.session_state.mapping_data = mapping_data
                
                # Display mapping table with tabs for different views
                tab1, tab2 = st.tabs(["Edit Mappings", "View Data Standard"])
                
                with tab1:
                    # Create DataFrame from session state
                    mapping_df = pd.DataFrame(st.session_state.mapping_data)
                    
                    # Display each row with edit button
                    for idx, row in mapping_df.iterrows():
                        col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 2, 1, 1, 1])
                        
                        with col1:
                            st.write(row["Column of Uploaded CSV"])
                        with col2:
                            st.write(f"{row['Sample Data']} ({row['Source Type']})")
                        with col3:
                            if st.session_state.editing_row == idx:
                                standard_options = list(data_standard.keys()) + ["Unmapped"]
                                new_value = st.selectbox(
                                    "Select mapping",
                                    options=standard_options,
                                    key=f"select_{idx}",
                                    index=standard_options.index(row["Mapped Field of Data Standard"]) if row["Mapped Field of Data Standard"] in standard_options else len(standard_options)-1
                                )
                                st.session_state.mapping_data[idx]["Mapped Field of Data Standard"] = new_value
                                
                                # Update destination type and type match
                                if new_value != "Unmapped":
                                    st.session_state.mapping_data[idx]["Destination Type"] = data_standard[new_value]["type"]
                                    st.session_state.mapping_data[idx]["Type Match"] = row["Source Type"] == data_standard[new_value]["type"]
                                else:
                                    st.session_state.mapping_data[idx]["Destination Type"] = "N/A"
                                    st.session_state.mapping_data[idx]["Type Match"] = False
                            else:
                                st.write(f"{row['Mapped Field of Data Standard']} ({row['Destination Type']})")
                        with col4:
                            # Display type match indicator
                            if row["Mapped Field of Data Standard"] != "Unmapped":
                                if row["Type Match"]:
                                    st.markdown("✅", unsafe_allow_html=True)
                                else:
                                    st.markdown("❌", unsafe_allow_html=True)
                            else:
                                st.write("")
                        with col5:
                            # Color-code confidence score
                            score = row["Confidence Score"]
                            if score >= 80:
                                st.markdown(f"<span style='color:green'>{score}%</span>", unsafe_allow_html=True)
                            elif score >= 60:
                                st.markdown(f"<span style='color:orange'>{score}%</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color:red'>{score}%</span>", unsafe_allow_html=True)
                        with col6:
                            if st.session_state.editing_row == idx:
                                if st.button("Save", key=f"save_{idx}"):
                                    st.session_state.editing_row = None
                                    st.rerun()
                            else:
                                if st.button("Edit", key=f"edit_{idx}"):
                                    st.session_state.editing_row = idx
                                    st.rerun()
                
                with tab2:
                    # Display data standard information
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### Data Standard Fields")
                    st.markdown("These are the standardized fields that your CSV columns can be mapped to.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Create a DataFrame to display the data standard
                    standard_df = pd.DataFrame([
                        {
                            "Field Name": field,
                            "Data Type": details["type"],
                            "Description": details["description"]
                        }
                        for field, details in data_standard.items()
                    ])
                    
                    st.dataframe(standard_df, use_container_width=True)
                
                # Add Accept Mapping button
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("Accept Mapping", use_container_width=True):
                        # Mark the source file as processed
                        st.session_state.processed_files.add(source_id)
                        
                        # Generate and display mapped DataFrame
                        st.markdown('<h2 class="sub-header">Generated Table Preview</h2>', unsafe_allow_html=True)
                        mapped_df = generate_mapped_dataframe(df, st.session_state.mapping_data)
                        
                        if not mapped_df.empty:
                            # Show preview of mapped data
                            st.dataframe(mapped_df.head(10), use_container_width=True)
                            
                            # Add download button
                            csv = mapped_df.to_csv(index=False)
                            source_filename = st.session_state.source_files_dict[source_id]["name"]
                            default_output_name = source_filename.replace(".csv", "_Mapped.csv")
                            output_filename = st.text_input("Output Filename", value=default_output_name)
                            
                            st.download_button(
                                label="Download Mapped Data as CSV",
                                data=csv,
                                file_name=output_filename,
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Success message
                            st.success(f"Mapping accepted and file '{st.session_state.source_files_dict[source_id]['name']}' processed!")
                            
                            # Store the processed file info for display
                            if 'last_processed_file' not in st.session_state:
                                st.session_state.last_processed_file = {}
                            
                            st.session_state.last_processed_file = {
                                'name': st.session_state.source_files_dict[source_id]['name'],
                                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'mapped_columns': len(mapped_df.columns)
                            }
                            
                            # Set a flag to indicate we should reset for next mapping
                            st.session_state.reset_for_next = True
                            
                            # Don't call st.rerun() here - let the user see the results first
                        else:
                            st.error("No data was mapped. Please check your mapping configuration.")
                
                with col2:
                    if st.button("Export Mapping Configuration", use_container_width=True):
                        # Export mapping configuration as JSON
                        mapping_config = {row["Column of Uploaded CSV"]: row["Mapped Field of Data Standard"] 
                                         for row in st.session_state.mapping_data 
                                         if row["Mapped Field of Data Standard"] != "Unmapped"}
                        
                        json_config = json.dumps(mapping_config, indent=2)
                        st.download_button(
                            label="Download Mapping Configuration",
                            data=json_config,
                            file_name=f"mapping_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )

    else:
        # Display welcome message when no files are selected
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Welcome to the Advanced CSV Data Mapper!
        
        This tool helps you:
        - Map your CSV columns to standardized field names
        - Validate data types and quality
        - Generate properly formatted data
        - Save and reuse mapping templates
        
        To get started:
        1. Upload source and destination files using the uploaders in the sidebar
        2. Select files for mapping from the dropdowns above
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)