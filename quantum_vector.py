import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import streamlit as st


def creds_entered(): 
    if st.session_state["user"].strip() == "Cadel_user" and st.session_state["passwd"].strip() == "Cadel2023!QV": 
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        st.error("Invalid Username/Password")

def authenticate_user(): 
    if "authenticated" not in st.session_state:
        st.text_input(label="Username:", value="", key="user", on_change=creds_entered)
        st.text_input(label="Password:", value="", key="passwd", type="password", on_change=creds_entered)
        return False
    elif st.session_state["authenticated"]:
        return True
    else:
        st.text_input(label="Username:", value="", key="user", on_change=creds_entered)
        st.text_input(label="Password:", value="", key="passwd", type="password", on_change=creds_entered)
        return False

if authenticate_user():
    # WEBPAGE FORMAT
    st.set_page_config(page_title='Quantum Vector')
    st.title('Quantum Vector')

    # Add image of logo
    logo_image = 'Quantum Report.png'
    st.image(logo_image, width=200, use_column_width=False)
    # Add the rest of your code here

    cadel_image = "Cadel.png"
    st.image(cadel_image, width=200, use_column_width=False)

  
    # Add CSS to float the image to the right
    st.markdown(
    """
    <style>
    .sidebar .stImage {
        margin-left: auto;
        margin-right: auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

  
    # AREA DROP AREA
    # Add a file drop area in the app
    uploaded_file = st.file_uploader("Feed me with your Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load data into pandas dataframe
        df = pd.read_excel(uploaded_file)
        message = 'Uploaded file:'
        st.write(message)
        st.write(df)
      
        # Display a multiselect widget for column selection
        selected_cols = st.sidebar.multiselect('Select analysis', df.columns, default=df.columns.tolist())
    
        # Selected data to run QV
        message = 'Selected data to run Quantum Vector:'
        st.write(message)
        st.write(df[selected_cols])
    
        # Select the reference
        sample_reference = st.sidebar.selectbox('Select a sample reference:', options=df.columns)
    
        
    #--------------------------------------------------------------------------------------------
    #  Apply the QV 
        # get the data from the selected columns
        data = df[selected_cols].dropna()
        data = data.transpose()
    
        message = 'Sample reference selected is :'
        st.write(message)
        st.markdown(f"**{sample_reference}**")
    
    
        #st.write(sample_reference)
        # normalize the data
        scaler = StandardScaler()
        norm_data = scaler.fit_transform(data)
        
        # perform PCA
        pca = PCA()
        pc = pca.fit_transform(norm_data)
        
        pc_axis = pc[:, :2]
        
        pc1 = pc_axis[:,0]
        pc2 = pc_axis[:,1]
        
        # Plot the data
        fig, ax = plt.subplots()
        for i in range(len(pc1)):
            # plot the PCA scores
            ax.plot(pc1[i], pc2[i], 'o', label=selected_cols[i])
    
        # Add labels and title
        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        ax.set_title('Quantum Vector')
        ax.legend()
        
        # Calculate distances to the reference point
        ref_point = np.array([pc1[selected_cols.index(sample_reference)], pc2[selected_cols.index(sample_reference)]])
    
        
        distances = [euclidean(ref_point, np.array([pc1[i], pc2[i]])) for i in range(len(pc1))]
    
        # Create DataFrame with distances and labels
        distances_df = pd.DataFrame({'Label': selected_cols, 'Distance': distances})
      
        # Print distances to console
        message = ('Distances to ' + sample_reference)
      
        st.write(message)
        # Display distances as a table
        st.table(distances_df)
      
        # Find the point with the smallest distance
        min_idx = np.argmin(distances)
        
        # Print distance and label of the closest point to console
        closest_dist = distances[min_idx]
        closest_label = df.index[min_idx]
        message = (f"Closest point to {sample_reference}: {closest_label} (distance: {closest_dist})")
        #st.write(message)
    
    
        # define the points
        xref, yref = pc1[selected_cols.index(sample_reference)], pc2[selected_cols.index(sample_reference)]
        
        # plot the arrows for all points
        for i in range(len(pc1)):
            # exclude reference point
            if selected_cols[i] != sample_reference:
                x2, y2 = pc1[i], pc2[i]
                # plot the arrow
                plt.annotate("", xy=(x2, y2), xytext=(xref, yref), arrowprops=dict(arrowstyle="->"))
            # include reference point
            else:
                plt.plot(xref, yref, 'o', markersize=8, markerfacecolor='black', markeredgecolor='black')
    
        # Set x and y axis limits
        max_val1 = max(abs(pc1))
        max_val2 = max(abs(pc2))
    
        max_val = max(max_val1,max_val2)
        
        plt.xlim(-max_val-0.5, max_val+0.5)
        plt.ylim(-max_val-0.5, max_val+0.5)
    
        # show the plot
        st.pyplot(fig)
    