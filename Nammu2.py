import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data with error handling
try:
    df1 = pd.read_csv("paneldata.csv")
    df2 = pd.read_csv("World bank_countries.csv")
except FileNotFoundError:
    st.error("One or more of the required files (paneldata.csv, World bank_countries.csv) are missing. Please upload them.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# Data preprocessing
df1 = df1.iloc[:, 1:]
merged_df = pd.merge(df1, df2, left_on=['Country_Name', 'Time'], right_on=['Country Name', 'Year'], how='left')
merged_df = merged_df.fillna(0)

# Sidebar: User inputs
st.sidebar.title("Dashboard Options")
countries = st.sidebar.multiselect("Select countries to compare:", merged_df["Country_Name"].unique(), default=merged_df["Country_Name"].unique()[:2])
aggregation_method = st.sidebar.selectbox("Select aggregation method:", ["mean", "median", "max"])
selected_metric = st.sidebar.selectbox("Select a metric for trends:", merged_df.select_dtypes(include='number').columns)
st.sidebar.markdown("#### Export Options")
export_data = st.sidebar.button("Export Data (CSV)")

# Time Range Selection
start_year, end_year = st.sidebar.slider("Select Year Range", int(merged_df["Time"].min()), int(merged_df["Time"].max()), (int(merged_df["Time"].min()), int(merged_df["Time"].max())))

# Filter data for selected countries and time range
filtered_df = merged_df[(merged_df["Country_Name"].isin(countries)) & (merged_df["Time"].between(start_year, end_year))]

# Layout: Title
st.title("Country Comparison Dashboard 🌍")
st.markdown("""
This interactive dashboard allows you to:
- Compare key metrics across selected countries.
- Visualize trends over time.
- Explore correlations between metrics.
- Visualize distributions and relationships through scatter and box plots.
- Rank the first selected country by comparing its top indicators with others.
""")

# Section 1: Summary Statistics
st.header("Summary Statistics")
# Ensure only numeric columns are aggregated
numeric_columns = filtered_df.select_dtypes(include='number')
# Group by country and aggregate
summary = numeric_columns.groupby(filtered_df["Country_Name"]).agg(aggregation_method).reset_index()
st.dataframe(summary)

# Section 2: Top Metrics Comparison
st.header("Top Metrics Comparison")
if len(countries) >= 2:
    # Filter numeric columns
    numeric_data = filtered_df.select_dtypes(include='number')
    numeric_data["Country_Name"] = filtered_df["Country_Name"]

    # Group by country and calculate mean
    avg_metrics = numeric_data.groupby("Country_Name").mean()

    # Transpose for easier comparison
    comparison_df = avg_metrics.loc[countries].transpose()

    # Select top 10 metrics for the first country
    top_metrics = comparison_df.nlargest(10, countries[0])
    
    # Create a grouped bar chart
    fig_bar = go.Figure()
    for country in countries:
        fig_bar.add_trace(go.Bar(
            x=top_metrics.index,
            y=top_metrics[country],
            name=country
        ))
    fig_bar.update_layout(
        title="Top Metrics Across Selected Countries",
        xaxis_title="Metrics",
        yaxis_title="Values",
        barmode="group",
        template="plotly_white"
    )
    st.plotly_chart(fig_bar)
else:
    st.warning("Please select at least two countries to compare metrics.")

# Section 3: Rank First Country Based on Top 10 Indicators
st.header("Rank First Country Based on Top 10 Indicators")

# Ensure a country is selected
if countries:
    first_country = countries[0]
    
    # Get the top 10 metrics for the first country
    top_metrics_first_country = avg_metrics.loc[first_country].nlargest(10).index.tolist()

    # Create a ranking for all countries based on these top 10 indicators
    ranking_df = pd.DataFrame()

    for metric in top_metrics_first_country:
        # Rank countries based on each metric
        ranking_df[metric] = avg_metrics[metric].rank(ascending=False)  # Higher values get better ranks

    # Calculate a composite rank (sum of ranks) for each country
    ranking_df["Composite Rank"] = ranking_df.sum(axis=1)

    # Rank countries based on their composite rank (lower rank is better)
    ranking_df["Final Rank"] = ranking_df["Composite Rank"].rank(ascending=True)

    # Show the ranking of countries based on the top 10 metrics
    ranking_df = ranking_df.sort_values(by="Final Rank")

    # Display the rankings
    st.dataframe(ranking_df[['Composite Rank', 'Final Rank']])

# Section 4: Trends Over Time
st.header("Trends Over Time")
fig_trend = px.line(
    filtered_df,
    x="Time",
    y=selected_metric,
    color="Country_Name",
    title=f"{selected_metric} Over Time",
    labels={"Time": "Year", selected_metric: selected_metric}
)
st.plotly_chart(fig_trend)

# Section 5: Correlation Heatmap
st.header("Correlation Heatmap (Top 10 Metrics)")

# Filter numeric columns
numeric_data = filtered_df.select_dtypes(include='number')
numeric_data["Country_Name"] = filtered_df["Country_Name"]

# Group by country and calculate mean
avg_metrics = numeric_data.groupby("Country_Name").mean()

# Select top 10 metrics for the first country in the list
if countries:
    top_metrics = avg_metrics.loc[countries[0]].nlargest(10).index.tolist()

    # Filter the dataset for top metrics
    filtered_metrics_data = numeric_data[top_metrics]

    # Calculate the correlation matrix
    correlation_matrix = filtered_metrics_data.corr()

    # Create the heatmap
    fig_heatmap = px.imshow(
        correlation_matrix,
        text_auto=True,
        title=f"Correlation Heatmap (Top 10 Metrics for {countries[0]})",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_heatmap)
else:
    st.warning("Please select at least one country to display the correlation map.")

# Section 6: Scatter Plot - Relationship Between Two Metrics
st.header("Scatter Plot - Relationship Between Two Metrics")
# Select two metrics for scatter plot
metric_x = st.sidebar.selectbox("Select the first metric for scatter plot:", merged_df.select_dtypes(include='number').columns)
metric_y = st.sidebar.selectbox("Select the second metric for scatter plot:", merged_df.select_dtypes(include='number').columns)

# Create scatter plot
scatter_fig = px.scatter(
    filtered_df,
    x=metric_x,
    y=metric_y,
    color="Country_Name",
    title=f"Scatter Plot: {metric_x} vs {metric_y}",
    labels={metric_x: metric_x, metric_y: metric_y}
)
st.plotly_chart(scatter_fig)

# Section 7: Box Plot - Distribution of a Metric Across Countries
st.header("Box Plot - Distribution of a Metric Across Countries")
# Select metric for box plot
box_plot_metric = st.sidebar.selectbox("Select a metric for box plot:", merged_df.select_dtypes(include='number').columns)

# Create box plot
box_fig = px.box(
    filtered_df,
    x="Country_Name",
    y=box_plot_metric,
    title=f"Box Plot: {box_plot_metric} Distribution Across Countries",
    labels={"Country_Name": "Country", box_plot_metric: box_plot_metric}
)
st.plotly_chart(box_fig)

# Summary Report Section
st.header("Comprehensive Final Analysis Report")

# Generate Final Analysis Insights
if countries:
    first_country = countries[0]
    top_metrics_first_country = avg_metrics.loc[first_country].nlargest(10).index.tolist()
    
    ranking_insights = f"""
    ### Ranking Insights:
    The first selected country, {first_country}, was ranked based on the top 10 indicators:
    - The composite rank for {first_country} is based on its performance in these key indicators.
    - Countries with a lower composite rank are considered closer in performance to {first_country}.
    """
    
    trend_insights = f"""
    ### Trends Over Time:
    The selected metric, {selected_metric}, shows the following trend:
    - The visualization illustrates the changes in {selected_metric} over time for each of the selected countries.
    - This allows for an understanding of how different countries have evolved in this metric.
    """
    
    correlation_insights = f"""
    ### Correlation Insights:
    The correlation heatmap shows relationships between the top 10 metrics:
    - Strong correlations suggest that certain metrics move together, providing deeper insights into related economic or social factors.
    """
    
    scatter_box_insights = f"""
    ### Scatter Plot & Box Plot Insights:
    - The scatter plot between {metric_x} and {metric_y} reveals the relationship between these two metrics across the selected countries.
    - The box plot shows the distribution of {box_plot_metric}, highlighting the variability and outliers across countries.
    """
    
    final_report = f"""
    ## Final Analysis Report:
    - **Top Metrics:** The top metrics for {first_country} were identified and compared to other selected countries.
    - **Ranking of {first_country}:** {ranking_insights}
    - **Trend Insights:** {trend_insights}
    - **Correlation Insights:** {correlation_insights}
    - **Visual Insights:** {scatter_box_insights}
    """
    
    st.markdown(final_report)
else:
    st.warning("Please select at least one country to generate the final analysis report.")

# Export Data
if export_data:
    # Dynamic filename based on selected countries and aggregation method
    export_filename = f"filtered_data_{'_'.join(countries)}_{aggregation_method}.csv"
    filtered_df.to_csv(export_filename, index=False)
    st.success(f"Data exported successfully as {export_filename}")
