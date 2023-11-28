import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import io

def read_csv(uploaded_file):
    # Read a CSV file and return a DataFrame
    return pd.read_csv(uploaded_file, header=None, names=['P', 'N', 'E', 'Z', 'D'])

def transform_points(params, points):
    # Apply the transformation: rotation, scaling, translation
    angle, scale, tx, ty = params
    rotation = R.from_euler('z', angle, degrees=True)
    rotated = rotation.apply(points[:, :2]) * scale
    translated = rotated + np.array([tx, ty])
    return np.hstack((translated, points[:, 2:]))

def objective_function(params, field_points, control_points):
    # Objective function to minimize
    transformed_points = transform_points(params, field_points)
    residuals = transformed_points[:, :2] - control_points[:, :2]
    return np.sqrt(np.sum(residuals**2, axis=1))

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def main():
    st.title('GCP Coordinate Transformation')

    # File upload
    field_points_file = st.file_uploader("Upload field points CSV", type='csv')
    control_points_file = st.file_uploader("Upload control points CSV", type='csv')
    pairing_file = st.file_uploader("Upload pairing CSV", type='csv')

    if field_points_file and control_points_file and pairing_file:
        field_points = read_csv(field_points_file)
        control_points = read_csv(control_points_file)
        pairings = pd.read_csv(pairing_file, header=None)

        # Extract paired points
        paired_field_points = field_points[field_points['P'].isin(pairings[0])].sort_values(by='P').to_numpy()
        paired_control_points = control_points[control_points['P'].isin(pairings[1])].sort_values(by='P').to_numpy()

        # Initial transformation
        initial_params = [0, 1, 0, 0]  # Initial guess: no rotation, no scaling, no translation
        res = least_squares(objective_function, initial_params, args=(paired_field_points, paired_control_points))

        # Error detection and outlier removal
        if len(pairings) > 3:
            residuals = objective_function(res.x, paired_field_points, paired_control_points)
            threshold = np.percentile(residuals, 75)
            valid_indices = residuals < threshold
            paired_field_points = paired_field_points[valid_indices]
            paired_control_points = paired_control_points[valid_indices]

            # Final transformation
            res = least_squares(objective_function, res.x, args=(paired_field_points, paired_control_points))

        transformed_points = transform_points(res.x, field_points.to_numpy())
        transformed_df = pd.DataFrame(transformed_points, columns=['N', 'E', 'Z'])
        transformed_df['P'] = field_points['P']
        transformed_df['D'] = field_points['D']
        transformed_df = transformed_df[['P', 'N', 'E', 'Z', 'D']]  # Reordering columns

        st.write("Transformation Parameters:", res.x)
        st.write("Transformed Points:", transformed_df)

        # Download link for transformed points
        csv = transformed_df.to_csv(index=False)
        st.download_button(
            label="Download Transformed Points as CSV",
            data=csv,
            file_name="transformed_points.csv",
            mime='text/csv',
        )

if __name__ == "__main__":
    main()