import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

st.set_page_config(
    page_title="Excel Data Visualization",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":bar_chart:",
    
)

st.title("Excel Data Visualization")

uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.sidebar.write("Columns in the dataset:")
    chosen_columns = st.sidebar.multiselect("Select the columns you want to visualize", df.columns)

    if chosen_columns:
        df_selected = df[chosen_columns]

        chart_options = ["Bar chart", "Pie chart", "Line chart", "Histogram"]
        chart_type = st.sidebar.selectbox("Choose the type of chart", chart_options)

        st.sidebar.subheader("Chart properties")
        chart_title = st.sidebar.text_input("Title")
        x_axis_label = st.sidebar.text_input("X-axis label")
        y_axis_label = st.sidebar.text_input("Y-axis label")

        show_legend = st.sidebar.checkbox("Show legend", value=True)
        adjust_chart = st.sidebar.checkbox("Adjust chart using Matplotlib's tight_layout()")
        plot_width = st.sidebar.slider("Plot width (inches)", min_value=5, value=10)
        plot_height = st.sidebar.slider("Plot height (inches)", min_value=5, value=5)

        color_map_options = ["viridis", "plasma", "inferno", "magma", "cividis"]
        selected_color_map = st.sidebar.selectbox("Choose a color map", color_map_options)
        cmap = matplotlib.cm.get_cmap(selected_color_map)

        if chart_type != "Histogram":
            groupby_column = st.sidebar.selectbox("Select the column to group by", df_selected.columns)

            aggregation_options = ["sum", "mean", "count", "min", "max"]
            aggregation_type = st.sidebar.selectbox("Choose the aggregation function", aggregation_options)

            grouped_data = df_selected.groupby(groupby_column).agg(aggregation_type)

            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.set_title(chart_title)

            if chart_type == "Bar chart":
                grouped_data.plot(kind='bar', ax=ax, legend=show_legend, colormap=cmap)
            elif chart_type == "Pie chart":
                grouped_data.plot(kind='pie', subplots=True, legend=show_legend, colormap=cmap, ax=ax)
            elif chart_type == "Line chart":
                grouped_data.plot(kind='line', ax=ax, legend=show_legend, colormap=cmap)

            if adjust_chart:
                plt.tight_layout()

            st.pyplot(fig)

        else:
            hist_column = st.sidebar.selectbox("Select the column for histogram", df_selected.columns)
            bin_count = st.sidebar.slider("Select the number of bins", 5, 50, 10)

            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.set_title(chart_title)
            df_selected[hist_column].plot(kind='hist', bins=bin_count, ax=ax, legend=show_legend, color=cmap(range(len(df_selected[hist_column]))))
        if adjust_chart:
            plt.tight_layout()

        st.pyplot(fig)

    else:
        st.write("No columns selected for visualization.")
else:
    st.write("No file uploaded.")
            
