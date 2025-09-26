import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

class VisualizationEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def create_overview_charts(self) -> List[go.Figure]:
        """Create overview visualizations"""
        charts: List[go.Figure] = []
        
        # Data types distribution
        dtype_counts = self.df.dtypes.apply(lambda x: x.name).value_counts()
        fig_dtypes = px.pie(
            names=dtype_counts.index,
            values=dtype_counts.values,
            title="Data Types Distribution"
        )
        charts.append(fig_dtypes)
        
        # Missing values bar
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
            fig_missing = px.bar(
                x=missing_counts.index,
                y=missing_counts.values,
                title="Missing Values by Column",
                labels={"x": "Column", "y": "Missing Count"}
            )
            charts.append(fig_missing)
        
        return charts
    
    def create_numerical_analysis(self) -> List[go.Figure]:
        """Create visualizations for numerical columns"""
        charts: List[go.Figure] = []
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            # Distribution plot with box
            fig_dist = px.histogram(
                self.df,
                x=col,
                title=f"Distribution of {col}",
                marginal="box",
                nbins=30
            )
            charts.append(fig_dist)
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            charts.append(fig_corr)
        
        return charts
    
    def create_categorical_analysis(self) -> List[go.Figure]:
        """Create visualizations for categorical columns"""
        charts: List[go.Figure] = []
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols[:3]:  # Limit to first 3 columns
            value_counts = self.df[col].value_counts(dropna=True).head(10)
            fig_cat = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f"Top Values in {col}",
                labels={"x": col, "y": "Count"}
            )
            charts.append(fig_cat)
        
        return charts
    
    def create_relationship_charts(self) -> List[go.Figure]:
        """Create charts showing relationships between variables"""
        charts: List[go.Figure] = []
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Scatter plot of first two numeric columns
            fig_scatter = px.scatter(
                self.df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                trendline="ols"
            )
            charts.append(fig_scatter)
        
        return charts