import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm



st.title("Excel Data Visualization and Machine Learning")

uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.sidebar.write("Columns in the dataset:")
    chosen_columns = st.sidebar.multiselect("Select the columns you want to visualize", df.columns)

    if chosen_columns:
        df_selected = df[chosen_columns]

        # Create a four-column layout
        col1, col2, col3, col4 = st.columns(4)

        # Chart options in the first and second columns
        with col1:
            st.subheader("Chart Options")

            chart_options = ["Bar chart", "Pie chart", "Line chart", "Histogram", "Pair plot", "Box plot"]
            chart_type = st.selectbox("Choose the type of chart", chart_options)

            groupby_column = st.selectbox("Select the column to group by (if applicable)", ["None"] + list(df_selected.columns), index=0)

        with col2:
            chart_title = st.text_input("Title")
            x_axis_label = st.text_input("X-axis label")
            y_axis_label = st.text_input("Y-axis label")

            show_legend = st.checkbox("Show legend", value=True)
            adjust_chart = st.checkbox("Adjust chart using Matplotlib's tight_layout()")
            plot_width = st.slider("Plot width (inches)", min_value=5, value=10)
            plot_height = st.slider("Plot height (inches)", min_value=5, value=5)

            color_map_options = ["viridis", "plasma", "inferno", "magma", "cividis"]
            selected_color_map = st.selectbox("Choose a color map", color_map_options)
            cmap = matplotlib.cm.get_cmap(selected_color_map)

        # Machine learning options in the third and fourth columns
        with col3:
            st.subheader("Machine Learning Options")

            ml_options = ["None", "Clustering (K-Means)", "Regression (Linear)", "Time Series Analysis (SARIMA)"]
            ml_option = st.selectbox("Choose a machine learning option", ml_options)

        if ml_option == "Clustering (K-Means)":
            with col4:
                st.write("""
                Clustering is an unsupervised learning technique that groups similar data points based on their features.
                K-Means is a popular clustering algorithm that partitions the data into K clusters.
                """)
                n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)

        elif ml_option == "Regression (Linear)":
            with col4:
                st.write("""
                Linear regression is a supervised learning technique that models the relationship between a target variable and one or more input features.
                The model tries to fit a straight line that best describes the relationship between the target and input features.""")
                target_column = st.selectbox("Select the target column", df_selected.columns)
                input_features = st.multiselect("Select the input features", df_selected.columns, default=[col for col in df_selected.columns if col != target_column])
                test_size = st.slider("Test dataset size (percentage)", min_value=10, max_value=40, value=20)

        elif ml_option == "Time Series Analysis (SARIMA)":
            with col4:
                st.write("""
                Time series analysis is used to analyze and forecast data points collected over time.
                SARIMA is a popular forecasting method that combines Seasonal, Autoregressive, Integrated, and Moving Average models.
                """)
                time_series_column = st.selectbox("Select the time series column", df_selected.columns)

                p = st.slider("ARIMA order p", min_value=0, max_value=5, value=1)
                d = st.slider("ARIMA order d", min_value=0, max_value=5, value=1)
                q = st.slider("ARIMA order q", min_value=0, max_value=5, value=1)
                order = (p, d, q)

                P = st.slider("Seasonal order P", min_value=0, max_value=5, value=1)
                D = st.slider("Seasonal order D", min_value=0, max_value=5, value=1)
                Q = st.slider("Seasonal order Q", min_value=0, max_value=5, value=1)
                s = st.slider("Seasonal order s", min_value=1, max_value=24, value=12)
                seasonal_order = (P, D, Q, s)



                
               # order = st.slider("ARIMA order (p, d, q)", min_value=0, max_value=5, value=(1, 1, 1))
                #seasonal_order = st.slider("Seasonal order (P, D, Q, s)", min_value=0, max_value=12, value=(1, 1, 1, 12))





        # Display DataFrame
        st.write("Selected Data:")
        st.write(df_selected)

        # Chart visualization in the middle of the page
        st.subheader("Chart Visualization")

        if chart_type not in ["Histogram", "Pair plot", "Box plot"] and groupby_column != "None":
            aggregation_options = ["sum", "mean", "count", "min", "max"]
            aggregation_type = st.selectbox("Choose the aggregation function", aggregation_options)

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

        elif chart_type == "Histogram":
            hist_column = st.selectbox("Select the column for histogram", df_selected.columns)
            bin_count = st.slider("Select the number of bins", 5, 50, 10)

            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.set_title(chart_title)
            df_selected[hist_column].plot(kind='hist', bins=bin_count, ax=ax, legend=show_legend, color=[cmap(i) for i in range(cmap.N)])

            if adjust_chart:
                plt.tight_layout()

            st.pyplot(fig)

        elif chart_type == "Pair plot":
            st.write("Pair plot requires at least two numeric columns.")
            numeric_columns = df_selected.select_dtypes(include=['number']).columns
            if len(numeric_columns) >= 2:
                hue_column = st.selectbox("Select the column for hue (optional)", ["None"] + list(numeric_columns), index=0)
                if hue_column != "None":
                    hue = hue_column
                else:
                    hue = None
                fig = sns.pairplot(df_selected, hue=hue, palette=[cmap(i) for i in range(cmap.N)], corner=True)
                fig.fig.suptitle(chart_title, y=1.03)
                if adjust_chart:
                    plt.tight_layout()

                st.pyplot(fig.fig)

            else:
                st.write("Not enough numeric columns for a pair plot.")

        elif chart_type == "Box plot":
            box_column = st.selectbox("Select the column for box plot", df_selected.columns)
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.set_title(chart_title)
            sns.boxplot(x=df_selected[box_column], palette=[cmap(i) for i in range(cmap.N)], ax=ax)

            if adjust_chart:
                plt.tight_layout()

            st.pyplot(fig)

        # Machine learning visualization
        if ml_option == "Clustering (K-Means)":
            st.subheader("Clustering Results")
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(df_selected)

            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            sns.scatterplot(data=df_selected, ax=ax, palette=[cmap(i) for i in range(cmap.N)], hue=kmeans.labels_, legend='full')
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)

        elif ml_option == "Regression (Linear)":
            st.subheader("Regression Results")
            X_train, X_test, y_train, y_test = train_test_split(df_selected[input_features], df_selected[target_column], test_size=test_size/100, random_state=42)
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predictions")
            ax.set_title("Linear Regression Results")

            st.pyplot(fig)
            st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
            st.write("R-squared (R²):", r2_score(y_test, y_pred))

        elif ml_option == "Time Series Analysis (SARIMA)":
            st.subheader("Time Series Analysis Results")
            time_series_data = df_selected.set_index(pd.to_datetime(df_selected[time_series_column]))
            time_series_data = time_series_data.drop(columns=[time_series_column])

            model = sm.tsa.statespace.SARIMAX(time_series_data, order=order, seasonal_order=seasonal_order)
            results = model.fit(disp=0)
            forecast = results.get_forecast(steps=12)

            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            time_series_data.plot(ax=ax)
            forecast.predicted_mean.plot(ax=ax)
            ax.set_title("SARIMA Time Series Forecast")
            st.pyplot(fig)

        else:
            st.write("No columns selected for visualization.")

if __name__ == "__main__":
    st.set_page_config(
    page_title="Excel Data Visualization and Machine Learning",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":bar_chart:",
)
    app.run()
                   
