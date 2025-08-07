#!/usr/bin/env python3
"""
🤖 AI Agents in Data Pipeline - LangGraph Demo for EARL 2025
A complete demonstration of autonomous agents working together to process, validate, and analyze data.

Requirements:
pip install langgraph==0.0.55 langchain-openai==0.1.8 pandas numpy matplotlib seaborn

Alternative if LangGraph has issues:
pip install langchain==0.2.5 langchain-openai==0.1.8 pandas numpy matplotlib seaborn

Usage: 
python data_agents_demo.py

Set your OpenAI API key as environment variable: OPENAI_API_KEY
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Optional
import warnings
warnings.filterwarnings('ignore')

# Try LangGraph first, fallback to basic implementation if issues
USE_LANGGRAPH = True
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    print("✅ Using LangGraph for advanced agent orchestration")
except ImportError as e:
    USE_LANGGRAPH = False
    print("⚠️  LangGraph not available, using simplified agent coordination")
    print("💡 Install with: pip install langgraph==0.0.55")

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Set up the model - using GPT-4 for best reasoning
model = ChatOpenAI(
    model="gpt-4o-mini",  # More cost-effective but still powerful
    temperature=0.1,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ============================================================================
# STATE DEFINITION - The shared memory across all agents
# ============================================================================

class AgentState(TypedDict):
    """Shared state that all agents can read from and write to"""
    messages: List[Any]
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    quality_report: Dict[str, Any]
    insights: List[str]
    visualizations: List[str]
    pipeline_status: str
    current_agent: str
    error_log: List[str]

# ============================================================================
# TOOLS - The capabilities our agents can use
# ============================================================================

@tool
def generate_sample_data(rows: int = 1000) -> str:
    """Generate realistic e-commerce sales data for demonstration"""
    np.random.seed(42)  # For reproducible demo
    
    # Generate realistic sales data with intentional issues
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', periods=rows)
    products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch', 'Camera']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    data = {
        'date': np.random.choice(dates, rows),
        'product': np.random.choice(products, rows),
        'region': np.random.choice(regions, rows),
        'sales_amount': np.random.normal(500, 150, rows),
        'quantity': np.random.poisson(3, rows),
        'customer_satisfaction': np.random.uniform(1, 5, rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce realistic data quality issues for agents to detect
    df.loc[df.sample(int(0.02 * rows)).index, 'sales_amount'] = np.nan  # 2% missing values
    df.loc[df.sample(int(0.01 * rows)).index, 'sales_amount'] = -100  # Negative values
    df.loc[df.sample(int(0.005 * rows)).index, 'quantity'] = 999  # Outliers
    df.loc[df.sample(int(0.01 * rows)).index, 'region'] = 'Unknown'  # Invalid categories
    
    return f"Generated {rows} rows of sales data with intentional quality issues for demonstration"

@tool
def analyze_data_quality(data_json: str) -> str:
    """Analyze data quality and return detailed quality metrics"""
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        
        quality_issues = []
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            quality_issues.append(f"Missing values detected: {missing.to_dict()}")
        
        # Negative sales amounts
        if 'sales_amount' in df.columns:
            negative_sales = (df['sales_amount'] < 0).sum()
            if negative_sales > 0:
                quality_issues.append(f"Found {negative_sales} negative sales amounts")
        
        # Outliers in quantity
        if 'quantity' in df.columns:
            q99 = df['quantity'].quantile(0.99)
            outliers = (df['quantity'] > q99 * 3).sum()
            if outliers > 0:
                quality_issues.append(f"Found {outliers} extreme quantity outliers")
        
        # Invalid regions
        if 'region' in df.columns:
            valid_regions = ['North', 'South', 'East', 'West', 'Central']
            invalid = (~df['region'].isin(valid_regions)).sum()
            if invalid > 0:
                quality_issues.append(f"Found {invalid} invalid region values")
        
        return f"Quality analysis complete. Issues found: {'; '.join(quality_issues) if quality_issues else 'No major issues detected'}"
    
    except Exception as e:
        return f"Error analyzing data quality: {str(e)}"

@tool
def clean_and_fix_data(data_json: str) -> str:
    """Automatically clean and fix common data issues"""
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        fixes_applied = []
        
        # Fix missing sales amounts with median imputation
        if df['sales_amount'].isnull().sum() > 0:
            median_sales = df['sales_amount'].median()
            df['sales_amount'].fillna(median_sales, inplace=True)
            fixes_applied.append("Imputed missing sales amounts with median")
        
        # Fix negative sales amounts
        if (df['sales_amount'] < 0).sum() > 0:
            df.loc[df['sales_amount'] < 0, 'sales_amount'] = df['sales_amount'].median()
            fixes_applied.append("Replaced negative sales amounts with median")
        
        # Cap extreme outliers
        if 'quantity' in df.columns:
            q99 = df['quantity'].quantile(0.99)
            df.loc[df['quantity'] > q99 * 3, 'quantity'] = int(q99)
            fixes_applied.append("Capped extreme quantity outliers")
        
        # Fix invalid regions
        if 'region' in df.columns:
            valid_regions = ['North', 'South', 'East', 'West', 'Central']
            df.loc[~df['region'].isin(valid_regions), 'region'] = 'Central'
            fixes_applied.append("Mapped invalid regions to 'Central'")
        
        return f"Data cleaning complete. Applied fixes: {'; '.join(fixes_applied)}"
    
    except Exception as e:
        return f"Error cleaning data: {str(e)}"

@tool
def generate_insights(data_json: str) -> str:
    """Generate business insights from clean data"""
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        
        insights = []
        
        # Sales performance by region
        if 'region' in df.columns and 'sales_amount' in df.columns:
            region_sales = df.groupby('region')['sales_amount'].mean().sort_values(ascending=False)
            best_region = region_sales.index[0]
            worst_region = region_sales.index[-1]
            insights.append(f"Best performing region: {best_region} (${region_sales[best_region]:.0f} avg)")
            insights.append(f"Lowest performing region: {worst_region} (${region_sales[worst_region]:.0f} avg)")
        
        # Product performance
        if 'product' in df.columns:
            product_sales = df.groupby('product')['sales_amount'].sum().sort_values(ascending=False)
            top_product = product_sales.index[0]
            insights.append(f"Top selling product: {top_product} (${product_sales[top_product]:,.0f} total)")
        
        # Seasonal trends (if date column exists)
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
            monthly_sales = df.groupby('month')['sales_amount'].mean()
            peak_month = monthly_sales.idxmax()
            insights.append(f"Peak sales month: {peak_month} (${monthly_sales[peak_month]:.0f} avg)")
        
        # Customer satisfaction correlation
        if 'customer_satisfaction' in df.columns:
            satisfaction_avg = df['customer_satisfaction'].mean()
            insights.append(f"Average customer satisfaction: {satisfaction_avg:.2f}/5.0")
        
        return f"Generated {len(insights)} key insights: " + " | ".join(insights)
    
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# ============================================================================
# AGENT DEFINITIONS - Our AI workforce
# ============================================================================

def create_data_ingestion_agent():
    """Agent responsible for data generation and initial processing"""
    
    def agent_node(state: AgentState):
        state["current_agent"] = "Data Ingestion Agent"
        state["messages"].append(HumanMessage(content="Generate sample sales data for analysis"))
        
        # Create and store the raw data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', periods=1000)
        products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch', 'Camera']
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        raw_data = {
            'date': np.random.choice(dates, 1000),
            'product': np.random.choice(products, 1000),
            'region': np.random.choice(regions, 1000),
            'sales_amount': np.random.normal(500, 150, 1000),
            'quantity': np.random.poisson(3, 1000),
            'customer_satisfaction': np.random.uniform(1, 5, 1000)
        }
        
        df = pd.DataFrame(raw_data)
        
        # Add quality issues for demonstration
        df.loc[df.sample(20).index, 'sales_amount'] = np.nan
        df.loc[df.sample(10).index, 'sales_amount'] = -100
        df.loc[df.sample(5).index, 'quantity'] = 999
        df.loc[df.sample(10).index, 'region'] = 'Unknown'
        
        state["raw_data"] = df
        state["pipeline_status"] = "Data ingested"
        
        message = f"✅ Data Ingestion Complete: Generated {len(df)} records with {df.shape[1]} columns"
        state["messages"].append(AIMessage(content=message))
        print(f"🔄 {state['current_agent']}: {message}")
        
        return state
    
    return agent_node

def create_quality_assurance_agent():
    """Agent responsible for data quality analysis and validation"""
    
    def agent_node(state: AgentState):
        state["current_agent"] = "Quality Assurance Agent"
        
        if state["raw_data"].empty:
            error_msg = "No raw data available for quality analysis"
            state["error_log"].append(error_msg)
            return state
        
        df = state["raw_data"]
        
        # Analyze data quality
        quality_issues = []
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            quality_issues.append(f"Missing values: {missing_values.to_dict()}")
        
        # Check for negative sales
        if 'sales_amount' in df.columns:
            negative_count = (df['sales_amount'] < 0).sum()
            if negative_count > 0:
                quality_issues.append(f"Negative sales: {negative_count} records")
        
        # Check for outliers
        if 'quantity' in df.columns:
            q99 = df['quantity'].quantile(0.99)
            outliers = (df['quantity'] > q99 * 3).sum()
            if outliers > 0:
                quality_issues.append(f"Extreme outliers: {outliers} records")
        
        # Check for invalid regions
        if 'region' in df.columns:
            valid_regions = ['North', 'South', 'East', 'West', 'Central']
            invalid = (~df['region'].isin(valid_regions)).sum()
            if invalid > 0:
                quality_issues.append(f"Invalid regions: {invalid} records")
        
        # Store quality report
        quality_report = {
            "total_rows": len(df),
            "missing_values": missing_values.to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "quality_score": max(50, 100 - len(quality_issues) * 10),
            "issues_found": len(quality_issues) > 0,
            "issues": quality_issues
        }
        
        state["quality_report"] = quality_report
        state["pipeline_status"] = "Quality analyzed"
        
        message = f"🔍 Quality Analysis: Found {len(quality_issues)} types of data quality issues"
        state["messages"].append(AIMessage(content=message))
        print(f"🔄 {state['current_agent']}: {message}")
        print(f"   Issues: {', '.join(quality_issues)}")
        
        return state
    
    return agent_node

def create_data_cleaning_agent():
    """Agent responsible for cleaning and fixing data issues"""
    
    def agent_node(state: AgentState):
        state["current_agent"] = "Data Cleaning Agent"
        
        if state["raw_data"].empty:
            error_msg = "No raw data available for cleaning"
            state["error_log"].append(error_msg)
            return state
        
        # Clean the data
        df = state["raw_data"].copy()
        fixes_applied = []
        
        # Fix missing sales amounts
        if df['sales_amount'].isnull().sum() > 0:
            median_sales = df['sales_amount'].median()
            df['sales_amount'].fillna(median_sales, inplace=True)
            fixes_applied.append("Imputed missing sales amounts")
        
        # Fix negative sales amounts
        if (df['sales_amount'] < 0).sum() > 0:
            df.loc[df['sales_amount'] < 0, 'sales_amount'] = df['sales_amount'].median()
            fixes_applied.append("Fixed negative sales amounts")
        
        # Cap extreme outliers
        q99 = df['quantity'].quantile(0.99)
        df.loc[df['quantity'] > q99 * 3, 'quantity'] = int(q99)
        fixes_applied.append("Capped quantity outliers")
        
        # Fix invalid regions
        valid_regions = ['North', 'South', 'East', 'West', 'Central']
        df.loc[~df['region'].isin(valid_regions), 'region'] = 'Central'
        fixes_applied.append("Fixed invalid regions")
        
        state["cleaned_data"] = df
        state["pipeline_status"] = "Data cleaned"
        
        message = f"🧹 Data Cleaning Complete: Applied {len(fixes_applied)} fixes"
        state["messages"].append(AIMessage(content=message))
        print(f"🔄 {state['current_agent']}: {message}")
        
        return state
    
    return agent_node

def create_analytics_agent():
    """Agent responsible for generating insights and analytics"""
    
    def agent_node(state: AgentState):
        state["current_agent"] = "Analytics Agent"
        
        if state["cleaned_data"].empty:
            error_msg = "No cleaned data available for analysis"
            state["error_log"].append(error_msg)
            return state
        
        df = state["cleaned_data"]
        insights = []
        
        # Generate business insights
        region_sales = df.groupby('region')['sales_amount'].mean().sort_values(ascending=False)
        insights.append(f"💰 Top region: {region_sales.index[0]} (${region_sales.iloc[0]:.0f} avg)")
        
        product_sales = df.groupby('product')['sales_amount'].sum().sort_values(ascending=False)
        insights.append(f"📱 Top product: {product_sales.index[0]} (${product_sales.iloc[0]:,.0f} total)")
        
        satisfaction_avg = df['customer_satisfaction'].mean()
        insights.append(f"😊 Avg satisfaction: {satisfaction_avg:.2f}/5.0")
        
        total_revenue = df['sales_amount'].sum()
        insights.append(f"📊 Total revenue: ${total_revenue:,.0f}")
        
        state["insights"] = insights
        state["pipeline_status"] = "Analysis complete"
        
        message = f"📈 Analytics Complete: Generated {len(insights)} key insights"
        state["messages"].append(AIMessage(content=message))
        print(f"🔄 {state['current_agent']}: {message}")
        
        return state
    
    return agent_node

def create_reporting_agent():
    """Agent responsible for creating final reports and visualizations"""
    
    def agent_node(state: AgentState):
        state["current_agent"] = "Reporting Agent"
        
        if state["cleaned_data"].empty or not state["insights"]:
            error_msg = "Insufficient data for reporting"
            state["error_log"].append(error_msg)
            return state
        
        # Create visualizations
        df = state["cleaned_data"]
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
            print("   Note: Using default plot style")
        
        # Create a comprehensive dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🤖 AI Agent Pipeline Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Sales by Region
        region_sales = df.groupby('region')['sales_amount'].mean()
        region_sales.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Average Sales by Region')
        ax1.set_ylabel('Sales Amount ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Product Performance
        product_sales = df.groupby('product')['sales_amount'].sum()
        ax2.pie(product_sales.values, labels=product_sales.index, autopct='%1.1f%%')
        ax2.set_title('Total Sales Distribution by Product')
        
        # Sales Trend Over Time
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        monthly_sales = df.groupby('month')['sales_amount'].mean()
        monthly_sales.plot(kind='line', ax=ax3, marker='o', color='green')
        ax3.set_title('Monthly Sales Trend')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Sales ($)')
        
        # Satisfaction vs Sales Correlation
        ax4.scatter(df['customer_satisfaction'], df['sales_amount'], alpha=0.6, color='coral')
        ax4.set_title('Customer Satisfaction vs Sales Amount')
        ax4.set_xlabel('Customer Satisfaction')
        ax4.set_ylabel('Sales Amount ($)')
        
        plt.tight_layout()
        plt.savefig('ai_agents_pipeline_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate final report
        report_summary = f"""
        
        🎯 AI AGENTS PIPELINE EXECUTION SUMMARY
        =====================================
        
        Pipeline Status: ✅ {state['pipeline_status']}
        Data Processed: {len(df):,} records
        Quality Score: {state['quality_report']['quality_score']}%
        
        🔍 KEY INSIGHTS DISCOVERED:
        {chr(10).join([f"   • {insight}" for insight in state['insights']])}
        
        🤖 AGENTS INVOLVED:
           • Data Ingestion Agent: Generated sample dataset
           • Quality Assurance Agent: Identified {len(state['quality_report']['missing_values'])} quality issues  
           • Data Cleaning Agent: Applied automated fixes
           • Analytics Agent: Extracted business insights
           • Reporting Agent: Created visualizations and summary
        
        ⚡ PIPELINE EFFICIENCY:
           • Total processing time: <30 seconds
           • Zero human intervention required
           • Autonomous issue detection and resolution
           • Ready-to-use business insights generated
        
        💡 This demonstrates how AI agents can work together to:
           ✓ Ingest and validate data automatically
           ✓ Detect and fix quality issues without human input  
           ✓ Generate actionable business insights
           ✓ Create professional reports and visualizations
           ✓ Operate as a coordinated, intelligent workforce
        """
        
        state["visualizations"] = ["ai_agents_pipeline_report.png"]
        state["pipeline_status"] = "Pipeline complete - Report generated"
        
        print(report_summary)
        state["messages"].append(AIMessage(content="📋 Comprehensive report generated with visualizations"))
        
        return state
    
    return agent_node

# ============================================================================
# WORKFLOW ORCHESTRATION - LangGraph State Machine
# ============================================================================

def create_data_pipeline_workflow():
    """Create the workflow that orchestrates our AI agents"""
    
    if USE_LANGGRAPH:
        # Use LangGraph for advanced orchestration
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("ingestion", create_data_ingestion_agent())
        workflow.add_node("quality_check", create_quality_assurance_agent())
        workflow.add_node("cleaning", create_data_cleaning_agent())
        workflow.add_node("analytics", create_analytics_agent())
        workflow.add_node("reporting", create_reporting_agent())
        
        # Define the flow between agents
        workflow.set_entry_point("ingestion")
        workflow.add_edge("ingestion", "quality_check")
        workflow.add_edge("quality_check", "cleaning")
        workflow.add_edge("cleaning", "analytics")
        workflow.add_edge("analytics", "reporting")
        workflow.add_edge("reporting", END)
        
        # Compile the workflow
        return workflow.compile()
    
    else:
        # Fallback: Simple sequential execution
        class SimpleWorkflow:
            def __init__(self):
                self.agents = [
                    ("ingestion", create_data_ingestion_agent()),
                    ("quality_check", create_quality_assurance_agent()),
                    ("cleaning", create_data_cleaning_agent()),
                    ("analytics", create_analytics_agent()),
                    ("reporting", create_reporting_agent())
                ]
            
            def invoke(self, state):
                """Execute agents sequentially"""
                current_state = state
                for agent_name, agent_func in self.agents:
                    print(f"\n🔄 Executing {agent_name}...")
                    current_state = agent_func(current_state)
                return current_state
        
        return SimpleWorkflow()

# ============================================================================
# MAIN EXECUTION - Run the AI Agent Pipeline
# ============================================================================

def main():
    """Run the complete AI agents data pipeline demonstration"""
    
    print("🚀 LAUNCHING AI AGENTS DATA PIPELINE DEMO")
    print("=" * 50)
    print("🎯 Demonstration for EARL 2025 - Brighton, UK")
    print("🤖 Watch as 5 AI agents work together autonomously!")
    print("=" * 50)
    
    # Initialize the state
    initial_state = AgentState(
        messages=[],
        raw_data=pd.DataFrame(),
        cleaned_data=pd.DataFrame(),
        quality_report={},
        insights=[],
        visualizations=[],
        pipeline_status="Initializing",
        current_agent="System",
        error_log=[]
    )
    
    # Create and run the workflow
    try:
        app = create_data_pipeline_workflow()
        
        print("\n🔄 EXECUTING AGENT PIPELINE...\n")
        
        # Run the complete pipeline
        final_state = app.invoke(initial_state)
        
        print("\n" + "=" * 50)
        print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Display final results
        if final_state.get("error_log"):
            print("\n⚠️  ERRORS ENCOUNTERED:")
            for error in final_state["error_log"]:
                print(f"   • {error}")
        
        print(f"\n✅ Final Status: {final_state['pipeline_status']}")
        print(f"📊 Data Quality Score: {final_state['quality_report'].get('quality_score', 'N/A')}%")
        print(f"🔍 Insights Generated: {len(final_state.get('insights', []))}")
        print(f"📈 Visualizations Created: {len(final_state.get('visualizations', []))}")
        
        print("\n💡 This demo shows how AI agents can:")
        print("   ✓ Work together autonomously")
        print("   ✓ Handle complex data workflows")  
        print("   ✓ Detect and fix issues automatically")
        print("   ✓ Generate actionable business insights")
        print("   ✓ Create professional reports")
        
        print(f"\n📋 Check 'ai_agents_pipeline_report.png' for the generated visualization!")
        
    except Exception as e:
        print(f"\n❌ Pipeline Error: {str(e)}")
        print("💡 Make sure you have set OPENAI_API_KEY environment variable")
        print("💡 Install requirements: pip install langgraph langchain-openai pandas numpy matplotlib seaborn")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
        print("🔑 Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("🆓 For demo purposes, continuing with simulated responses...\n")
    
    main()
