import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analysis_results = {}
    
    def basic_info(self) -> Dict[str, Any]:
        """Extract basic dataset information"""
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.apply(lambda x: x.name).to_dict(),
            "memory_usage": int(self.df.memory_usage(deep=True).sum()),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum())
        }
    
    def statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summaries for numerical and categorical columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        stats_summary = {
            "numerical": {},
            "categorical": {}
        }
        
        # Numerical analysis
        for col in numeric_cols:
            col_series = self.df[col].dropna()
            if col_series.empty:
                summary = {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "skewness": None,
                    "kurtosis": None,
                    "outliers_count": 0
                }
            else:
                summary = {
                    "mean": float(col_series.mean()),
                    "median": float(col_series.median()),
                    "std": float(col_series.std()),
                    "min": float(col_series.min()),
                    "max": float(col_series.max()),
                    "skewness": float(stats.skew(col_series)) if col_series.size > 1 else 0.0,
                    "kurtosis": float(stats.kurtosis(col_series)) if col_series.size > 1 else 0.0,
                    "outliers_count": int(self._detect_outliers(col_series))
                }
            stats_summary["numerical"][col] = summary
        
        # Categorical analysis
        for col in categorical_cols:
            col_series = self.df[col].astype('object')
            mode_val = col_series.mode().iloc[0] if not col_series.mode().empty else None
            stats_summary["categorical"][col] = {
                "unique_count": int(col_series.nunique(dropna=True)),
                "most_frequent": mode_val,
                "value_counts": col_series.value_counts(dropna=True).head(10).to_dict()
            }
        
        return stats_summary
    
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect interesting patterns in the data"""
        patterns = {
            "correlations": [],
            "trends": [],
            "anomalies": []
        }
        
        # Correlation analysis
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            high_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.isna(corr_val):
                        continue
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        high_correlations.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
            
            patterns["correlations"] = high_correlations
        
        # Simple trend detection: check monotonic sequences for numeric columns
        for col in numeric_df.columns:
            ser = numeric_df[col].dropna()
            if ser.size >= 3:
                # compute simple rolling slope (difference)
                diffs = ser.diff().dropna()
                if all(diffs > 0):
                    patterns["trends"].append({"column": col, "trend": "increasing"})
                elif all(diffs < 0):
                    patterns["trends"].append({"column": col, "trend": "decreasing"})
        
        # Anomaly detection: large z-score
        for col in numeric_df.columns:
            ser = numeric_df[col].dropna()
            if ser.size > 5:
                zscores = np.abs(stats.zscore(ser))
                anomalies = int((zscores > 3).sum())
                if anomalies > 0:
                    patterns["anomalies"].append({"column": col, "anomalies_count": int(anomalies)})
        
        return patterns
    
    def _detect_outliers(self, series: pd.Series) -> int:
        """Detect outliers using IQR method"""
        if series.empty:
            return 0
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return 0
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(((series < lower_bound) | (series > upper_bound)).sum())
