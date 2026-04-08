import streamlit as st
import pandas as pd
import plotly.express as px
# To run, go to the terminal and run: streamlit run dashboard.py

st.title("Water Data Dashboard")
st.header("Spring 2026")
st.subheader("Prof. Greg Reis")

st.divider()

st.sidebar.header("Load Datasets")
file_upload = st.sidebar.file_uploader("Upload a file", type=["csv"])
if file_upload is not None:
    df = pd.read_csv(file_upload)
else:
    df = pd.read_csv("logs/biscayne_bay_water_quality2.csv")

tab1, tab2, tab3 = st.tabs(["Raw Data","Charts","Maps"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(df)
    st.caption("Data collected from underwater robot by Dr. Reis.")

    st.divider()

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

with tab2:
    st.subheader("Charts")
    with st.expander("Scatter Plot"):
        fig1 = px.scatter(df,
                          x='Total Water Column (m)',
                          y='Temperature (c)',
                          color='Salinity (ppt)',
                          title='Temperature vs. Total Water Column')
        st.plotly_chart(fig1)

    with st.expander('Line Chart'):
        col1, col2 = st.columns([1,4])
        with col1:
            parameter_selected = st.selectbox('Select Parameter',
                                              df.columns)
            color_selected = st.color_picker('Select a Color', '#6495ED')
            
        with col2:
            fig2 = px.line(
                df,
                x='Time',
                y=parameter_selected,
                title=f'{parameter_selected} over Time',
            )
            fig2.update_traces(line_color=color_selected)
            st.plotly_chart(fig2)
    
    with st.expander('3D Scatter Plot'):
        # Because we have lat, lon, and depth columns
        fig3 = px.scatter_3d(
            df,
            x='Longitude',
            y='Latitude',
            z='Total Water Column (m)',
            color='ODO mg/L',
            title='3D Scatter Plot of Water Depth'
        )
        fig3.update_scenes(zaxis_autorange='reversed')
        st.plotly_chart(fig3)
    with st.expander('Boxplot'):
        selected_parameter = st.selectbox('Select Parameter',
                                            df.columns)
        fig5 = px.box(
            df,
            x=selected_parameter,
            title=f'Boxplot of {selected_parameter}'
        )
        st.plotly_chart(fig5)

    with st.expander('Histogram'):
        col1, col2 = st.columns([1,4])
        with col1:
            x_parameter_selected = st.selectbox('Select X Parameter',
                                              df.columns)
            y_parameter_selected = st.selectbox('Select Y Parameter',
                                 df.columns)
        with col2:
            fig6 = px.histogram(
                df,
                x=x_parameter_selected,
                y=y_parameter_selected,
                title=f'Histogram of {x_parameter_selected} and {y_parameter_selected}'
            )
            st.plotly_chart(fig6)
    
    with st.expander('Heat Map'):
        fig7 = px.imshow(df.corr(numeric_only=True),
                         title='Correlation Matrix Heat Map',
                         text_auto=True,
                         color_continuous_scale='RdBl')
        st.plotly_chart(fig7)

with tab3:
    st.subheader('Maps')
    fig4 = px.scatter_mapbox(
        df,
        lon='Longitude',
        lat='Latitude',
        mapbox_style='open-street-map',
        zoom=17,
        color='Temperature (c)',
        hover_data=df
    )
    st.caption('Hover over a point to see more details')
    st.plotly_chart(fig4)

# TODO: add more charts, such as:
#   boxplots
#   histograms
#   heatmap for correlation (as seen in last weeks class)