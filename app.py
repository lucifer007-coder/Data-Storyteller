import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_analyzer import DataAnalyzer
from src.ai_insights import AIInsightsGenerator
from src.visualizations import VisualizationEngine
from config.settings import Config
from src.utils import allowed_file, check_file_size_bytes
import io

# Page configuration
st.set_page_config(
    page_title="CSV Data Storyteller",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📊 CSV Data Storyteller")
    st.markdown("Upload your CSV file and let AI tell the story hidden in your data!")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help=f"Maximum file size: {Config.MAX_FILE_SIZE}MB"
        )
        
        st.subheader("Analysis Options")
        include_visualizations = st.checkbox("Generate Visualizations", value=True)
        include_ai_insights = st.checkbox("AI Insights", value=True)
        include_suggestions = st.checkbox("Next Steps Suggestions", value=True)
        st.write("---")
        st.markdown("Ollama configuration:")
        st.text(f"URL: {Config.OLLAMA_BASE_URL}")
        st.text(f"Model: {Config.GEMMA_MODEL}")
    
    if uploaded_file is not None:
        # Validate filename and size
        filename = uploaded_file.name
        if not allowed_file(filename):
            st.error("Unsupported file format. Please upload a CSV file.")
            return
        
        try:
            # uploaded_file is a BytesIO-like object. Many have .size attribute.
            file_size = getattr(uploaded_file, "size", None)
            if file_size is None:
                # Fallback: read to measure
                uploaded_file.seek(0, io.SEEK_END)
                file_size = uploaded_file.tell()
                uploaded_file.seek(0)
            ok, msg = check_file_size_bytes(file_size)
            if not ok:
                st.error(msg)
                return
        except Exception:
            # If size check fails, continue cautiously
            pass
        
        try:
            with st.spinner("Loading data..."):
                # Use pandas read_csv; handle common separators and encoding
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            
            st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns!")
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", int(df.isnull().sum().sum()))
            with col4:
                mem_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory Usage", f"{mem_mb:.2f} MB")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Analysis
            with st.spinner("Analyzing data..."):
                analyzer = DataAnalyzer(df)
                analysis_results = {
                    "basic_info": analyzer.basic_info(),
                    "statistical_summary": analyzer.statistical_summary(),
                    "patterns": analyzer.detect_patterns()
                }
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🤖 AI Insights", "📊 Visualizations", "🎯 Next Steps"])
            
            with tab1:
                display_overview(analysis_results)
            
            with tab2:
                if include_ai_insights:
                    display_ai_insights(analysis_results)
                else:
                    st.info("AI Insights disabled. Enable in sidebar to see AI-generated insights.")
            
            with tab3:
                if include_visualizations:
                    display_visualizations(df)
                else:
                    st.info("Visualizations disabled. Enable in sidebar to see charts.")
            
            with tab4:
                if include_suggestions:
                    display_suggestions(analysis_results)
                else:
                    st.info("Suggestions disabled. Enable in sidebar to see recommendations.")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Landing page
        st.markdown("""
        ## How it works:
        1. **Upload** your CSV file using the sidebar
        2. **Analyze** - Our AI examines patterns and trends
        3. **Discover** - Get natural language insights and visualizations
        4. **Act** - Receive suggestions for next steps
        
        ### Features:
        - 🔍 Automatic data profiling and quality assessment
        - 🤖 AI-powered insights using local Gemma models (Ollama)
        - 📊 Interactive visualizations
        - 📝 Natural language summaries
        - 🎯 Actionable recommendations
        """)
        st.info("Tip: Install Ollama locally and pull a Gemma model (e.g. gemma:7b) so AI insights work offline.")
    

def display_overview(analysis_results):
    st.subheader("📈 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Column Information**")
        dtypes = analysis_results['basic_info']['dtypes']
        dtypes_df = pd.DataFrame({
            'Column': list(dtypes.keys()),
            'Data Type': list(dtypes.values())
        })
        st.dataframe(dtypes_df, use_container_width=True)
    
    with col2:
        st.write("**Missing Values**")
        missing = analysis_results['basic_info']['missing_values']
        missing_df = pd.DataFrame({
            'Column': list(missing.keys()),
            'Missing Count': list(missing.values())
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found!")
    
    # Statistical summary
    num_summary = analysis_results['statistical_summary'].get('numerical', {})
    if num_summary:
        st.write("**Numerical Columns Summary**")
        numerical_summary = []
        for col, stats in num_summary.items():
            mean = stats['mean']
            median = stats['median']
            std = stats['std']
            numerical_summary.append({
                'Column': col,
                'Mean': f"{mean:.2f}" if mean is not None else "N/A",
                'Median': f"{median:.2f}" if median is not None else "N/A",
                'Std Dev': f"{std:.2f}" if std is not None else "N/A",
                'Outliers': stats.get('outliers_count', 0)
            })
        st.dataframe(pd.DataFrame(numerical_summary), use_container_width=True)

def display_ai_insights(analysis_results):
    st.subheader("🤖 AI-Generated Insights")
    
    ai_generator = AIInsightsGenerator()
    
    with st.spinner("Generating AI insights..."):
        story = ai_generator.generate_data_story(analysis_results)
    
    st.markdown("### 📖 Data Story")
    st.write(story)
    
    with st.spinner("Analyzing visualization opportunities..."):
        viz_suggestions = ai_generator.suggest_visualizations(analysis_results)
    
    st.markdown("### 📊 Visualization Recommendations")
    st.write(viz_suggestions)

def display_visualizations(df):
    st.subheader("📊 Data Visualizations")
    
    viz_engine = VisualizationEngine(df)
    
    # Overview charts
    st.markdown("#### Overview")
    overview_charts = viz_engine.create_overview_charts()
    if overview_charts:
        for chart in overview_charts:
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No overview charts to display.")
    
    # Numerical analysis
    if len(df.select_dtypes(include=['number']).columns) > 0:
        st.markdown("#### Numerical Analysis")
        numerical_charts = viz_engine.create_numerical_analysis()
        for chart in numerical_charts:
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No numerical columns found.")
    
    # Categorical analysis
    if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
        st.markdown("#### Categorical Analysis")
        categorical_charts = viz_engine.create_categorical_analysis()
        for chart in categorical_charts:
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No categorical columns found.")
    
    # Relationships
    relationship_charts = viz_engine.create_relationship_charts()
    if relationship_charts:
        st.markdown("#### Relationships")
        for chart in relationship_charts:
            st.plotly_chart(chart, use_container_width=True)

def display_suggestions(analysis_results):
    st.subheader("🎯 Next Steps & Recommendations")
    
    ai_generator = AIInsightsGenerator()
    
    with st.spinner("Generating recommendations..."):
        suggestions = ai_generator.suggest_next_steps(analysis_results)
    
    st.write(suggestions)

if __name__ == "__main__":
    main()