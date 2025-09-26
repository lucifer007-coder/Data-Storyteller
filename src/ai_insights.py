import requests
import json
from typing import Dict, List, Any
from config.settings import Config

class AIInsightsGenerator:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.GEMMA_MODEL
    
    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama local server (simple wrapper)."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60000)
            resp.raise_for_status()
            data = resp.json()
            # Ollama has returned structure like {"response": "..."} in the spec above; be defensive:
            if isinstance(data, dict):
                if "response" in data:
                    return data["response"]
                if "output" in data:  # fallbacks
                    return data["output"]
                # sometimes the API returns a list of messages or text field
                if "text" in data:
                    return data["text"]
            return json.dumps(data, indent=2)
        except requests.RequestException as e:
            return f"Error generating insights: {str(e)}"
        except Exception as e:
            return f"Unexpected error calling Ollama: {str(e)}"
    
    def generate_data_story(self, analysis_results: Dict[str, Any]) -> str:
        """Generate natural language story from analysis results"""
        prompt = f"""
You are a helpful data analyst. Create a compelling narrative story about this dataset, focusing on the most interesting findings.

Dataset Overview:
- Shape: {analysis_results['basic_info'].get('shape')}
- Columns: {', '.join(analysis_results['basic_info'].get('columns', []))}
- Missing values: {analysis_results['basic_info'].get('missing_values')}

Statistical Summary:
{json.dumps(analysis_results.get('statistical_summary', {}), indent=2, default=str)}

Patterns Found:
{json.dumps(analysis_results.get('patterns', {}), indent=2, default=str)}

Please provide:
1. A brief overview of what this data might represent
2. Key insights and interesting findings (prioritize the top 5)
3. Notable patterns or anomalies
4. Potential business implications and recommended next steps

Write in a clear, engaging narrative style and keep the answer concise (about 5-8 paragraphs).
"""
        return self._call_ollama(prompt)
    
    def suggest_visualizations(self, analysis_results: Dict[str, Any]) -> str:
        """Suggest appropriate visualizations"""
        prompt = f"""
Based on this dataset analysis, suggest the most effective visualizations:

Columns and Types:
{json.dumps(analysis_results['basic_info'].get('dtypes', {}), indent=2, default=str)}

Statistical Summary:
{json.dumps(analysis_results.get('statistical_summary', {}), indent=2, default=str)}

Correlations:
{json.dumps(analysis_results.get('patterns', {}).get('correlations', []), indent=2, default=str)}

Recommend specific chart types for:
- distributions of key numeric variables
- relationships between correlated variables
- top categorical breakdowns

For each recommended chart, provide:
- chart type
- columns to use
- brief explanation of why it's valuable
Limit to 6 suggested visualizations.
"""
        return self._call_ollama(prompt)
    
    def suggest_next_steps(self, analysis_results: Dict[str, Any]) -> str:
        """Suggest next steps for analysis"""
        prompt = f"""
Based on this data analysis, suggest actionable next steps:

Dataset Info:
{json.dumps(analysis_results.get('basic_info', {}), indent=2, default=str)}

Key Patterns:
{json.dumps(analysis_results.get('patterns', {}), indent=2, default=str)}

Provide specific recommendations for:
1. Further analysis opportunities (statistical tests, segmentation, time-series)
2. Data quality improvements (which columns need cleaning or enrichment)
3. Potential machine learning applications (target, features)
4. Business questions to explore

Keep suggestions concise and actionable.
"""
        return self._call_ollama(prompt)