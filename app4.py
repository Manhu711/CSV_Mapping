import streamlit as st
import pandas as pd
from thefuzz import fuzz
import numpy as np
import io

# Define the data standard fields with expected data types
DATA_STANDARD = {
    "SubjID": "string",
    "City": "string",
    "State": "string",
    "Date": "datetime",
    "Test": "string",
    "Result": "numeric",
    "Unit": "string"
}

def get_best_match(column_name, standard_fields):
    """Find the best matching standard field using fuzzy matching"""
    scores = [(field, fuzz.ratio(column_name.lower(), field.lower())) 
             for field in standard_fields]
    best_match = max(scores, key=lambda x: x[1])
    return best_match[0], best_match[1]

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
        mapped_field = row["Mapped Field of Data Standard"]
        if mapped_field != "Unmapped":
            if mapped_field not in seen_fields:
                column_mapping[row["Column of Uploaded CSV"]] = mapped_field
                seen_fields.add(mapped_field)
            else:
                duplicate_mappings.append((row["Column of Uploaded CSV"], mapped_field))
    
    # Show warning for duplicate mappings
    if duplicate_mappings:
        warning_msg = "Warning: Multiple columns mapped to the same standard field. Only the first mapping will be used:\n"
        for orig_col, mapped_field in duplicate_mappings:
            warning_msg += f"\n- '{orig_col}' also mapped to '{mapped_field}' (ignored)"
        st.warning(warning_msg)
    
    # Only keep columns that are mapped
    new_df = new_df[list(column_mapping.keys())]
    # Rename columns according to mapping
    new_df = new_df.rename(columns=column_mapping)
    return new_df

def style_mapping_df(df):
    """Style the DataFrame based on confidence scores"""
    def color_confidence(val):
        try:
            score = float(val)
            if score >= 80:
                return 'background-color: #90EE90'  # Light green
            elif score >= 60:
                return 'background-color: #FFFFE0'  # Light yellow
            else:
                return 'background-color: #FFB6C1'  # Light red
        except:
            return ''
    
    return df.style.applymap(color_confidence, subset=['Confidence Score'])

def main():
    st.title("CSV Data Field Mapping")
    
    # Initialize session state for editing
    if 'editing_row' not in st.session_state:
        st.session_state.editing_row = None
    if 'mapping_data' not in st.session_state:
        st.session_state.mapping_data = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    
    # File upload widget
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df
            
            # Display basic information about the dataset
            st.subheader("Dataset Info")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Number of rows:", df.shape[0])
            with col2:
                st.write("Number of columns:", df.shape[1])
            
            # Display the first few rows of the dataset
            st.subheader("Data Preview")
            # Hide index column and display the DataFrame
            st.dataframe(df.head(), hide_index=True)
            
            # Add button for column mapping
            if st.button("Show Column Mapping") or st.session_state.mapping_data is not None:
                st.subheader("Column Mapping Analysis")
                
                # Display column headers
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
                with col1:
                    st.write("**Column Name of Uploaded File**")
                with col2:
                    st.write("**Example Data**")
                with col3:
                    st.write("**Mapped Field of Data Standard**")
                with col4:
                    st.write("**Confidence Score by Fuzzy Match**")
                with col5:
                    st.write("**Actions**")
                
                # Create mapping table if it doesn't exist
                if st.session_state.mapping_data is None:
                    mapping_data = []
                    for col in df.columns:
                        # Get sample data (first non-null value)
                        sample_data = df[col].iloc[0] if not df[col].empty else "No data"
                        
                        # Get mapped field and confidence score using fuzzy matching
                        mapped_field, confidence_score = get_best_match(col, DATA_STANDARD.keys())
                        
                        mapping_data.append({
                            "Column of Uploaded CSV": col,
                            "Sample Data": str(sample_data),
                            "Mapped Field of Data Standard": mapped_field if confidence_score > 50 else "Unmapped",
                            "Confidence Score": confidence_score
                        })
                    st.session_state.mapping_data = mapping_data
                
                # Create DataFrame from session state
                mapping_df = pd.DataFrame(st.session_state.mapping_data)
                
                # Display each row with edit button
                for idx, row in mapping_df.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
                    
                    with col1:
                        st.write(row["Column of Uploaded CSV"])
                    with col2:
                        st.write(row["Sample Data"])
                    with col3:
                        if st.session_state.editing_row == idx:
                            standard_options = list(DATA_STANDARD.keys()) + ["Unmapped"]
                            new_value = st.selectbox(
                                "Select mapping",
                                options=standard_options,
                                key=f"select_{idx}",
                                index=standard_options.index(row["Mapped Field of Data Standard"])
                            )
                            st.session_state.mapping_data[idx]["Mapped Field of Data Standard"] = new_value
                        else:
                            st.write(row["Mapped Field of Data Standard"])
                    with col4:
                        st.write(row["Confidence Score"])
                    with col5:
                        if st.session_state.editing_row == idx:
                            if st.button("Save", key=f"save_{idx}"):
                                st.session_state.editing_row = None
                                st.rerun()
                        else:
                            if st.button("Edit", key=f"edit_{idx}"):
                                st.session_state.editing_row = idx
                                st.rerun()
                
                # Add Accept Mapping button
                st.markdown("---")  # Add a separator line
                if st.button("Accept Mapping"):
                    # Generate and display mapped DataFrame
                    st.subheader("Generated Table Preview")
                    mapped_df = generate_mapped_dataframe(st.session_state.original_df, st.session_state.mapping_data)
                    st.dataframe(mapped_df.head(), hide_index=True)
                    
                    # Add download button
                    if not mapped_df.empty:
                        csv = mapped_df.to_csv(index=False)
                        st.download_button(
                            label="Download Mapped Data as CSV",
                            data=csv,
                            file_name="mapped_data.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()