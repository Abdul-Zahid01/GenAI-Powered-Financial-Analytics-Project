import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import sys
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# 1. DEFINE THE TOOLS FIRST 🛠️
def execute_ai_code(code_string, data_frame):
    # 0. Unwrap the code (The Advanced Vacuum) 🧹
    code_blocks = re.findall(r"```(?:python)?(.*?)```", code_string, re.DOTALL)
    if code_blocks:
        clean_code = "\n".join(code_blocks)
    else:
        clean_code = code_string.strip()

    # 1. Setup the Sandbox
    local_env = {
        "pd": pd,
        "df": data_frame,
        "plt": plt,
        "sns": sns,
        "st": st,
        "np": np,
        "StringIO": StringIO
    }
    
    # 2. Setup the Trap
    old_stdout = sys.stdout 
    capture_buffer = StringIO()
    sys.stdout = capture_buffer 
    
    try:
        # 3. Run the code
        exec(clean_code, {}, local_env)
        return_value = capture_buffer.getvalue()
    except Exception as e:
        return_value = f"Error executing code: {e}"
    finally:
        # 4. Restore the original "mouth"
        sys.stdout = old_stdout
        
    return return_value

# 2. APP SETUP 🏗️
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(layout="wide") # Makes the dashboard use the full screen width
st.title("🍕 Restaurant Financial Analytics")
st.subheader("AI-Powered Insights")

# Step 1: Upload the file (Sidebar)
uploaded_file = st.sidebar.file_uploader("Upload Data", type="csv")

if uploaded_file is not None:
    # --- PHASE 1: LOAD & PREP ---
    df = pd.read_csv(uploaded_file)
    
    # Cleaning
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Manager'] = df['Manager'].fillna('Unknown')
    average_price = df['Price'].mean()
    df['Price'] = df['Price'].fillna(average_price)
    
    # Feature Engineering (Making the Money Column!) 💰
    df['Revenue'] = df['Price'] * df['Quantity']

    # --- PHASE 2: CALCULATE METRICS ---
    # Now that 'Revenue' exists, we can calculate these safely
    total_revenue = df['Revenue'].sum()
    top_product = df.groupby('Product')['Revenue'].sum().idxmax()
    top_city = df.groupby('City')['Revenue'].sum().idxmax()
    
    # --- PHASE 3: DRAW THE DASHBOARD ---
    
    # 1. The Headline Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Top Product", top_product)
    col3.metric("Best City", top_city)
    
    st.divider()

    # 2. The Data Inspector (Hidden by default)
    with st.expander("👀 View Raw Data"):
        st.write("### Raw Data Preview")
        st.dataframe(df.head())

    with st.expander("🔍 Data Overview"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("### Data Types")
            st.write(df.dtypes)
        with col_b:
            st.write("### Missing Values")
            st.write(df.isnull().sum())

    # 3. Interactive Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("💰 Price vs. Quantity")
        st.scatter_chart(df, x="Price", y="Quantity")
    
    with col_chart2:
        st.subheader("📈 Revenue Distribution")
        st.bar_chart(df.groupby("City")["Revenue"].sum())

    st.divider()

    # --- PHASE 4: THE AI BRAIN ---
    st.subheader("👨‍🍳 Executive AI Analyst")
    
    if st.button("Generate Strategic Report"):
        
        # Prepare Context
        buffer = StringIO() 
        df.info(buf=buffer)
        data_info = buffer.getvalue()

        # 2. The Request (Upgraded!)
        user_question = """
        Full Strategic Audit Task:
        1. Sales Performance: Analyze total revenue and identifying the peak sales period.
        2. Product Mix: Compare the Top 5 products vs. Bottom 5 products by revenue.
        3. Channel Analysis: Compare 'Online' vs 'In-store' performance.
        4. Geographic Analysis: Identify the best and worst performing cities.
        5. Deep Dive: Generate 3 specific SMART questions based on the data and ANSWER them with evidence.
        6. Visuals: Create a separate chart for each analysis point above.
        """

        prompt = f"""
        You are a Senior Data Strategist.
        You have access to a dataframe named 'df'.
        
        Here is the dataframe info:
        {data_info}
        
        Your task: Write python code to answer this question: "{user_question}"
        
        PREPROCESSING Rules (The Janitor):
        1. Clean the data (Duplicates, Missing Values).
        2. CRITICAL: Do NOT use df.append(). Use pd.concat().
        3. Use print() for cleaning logs.
        4. Standardization: Convert categorical columns to Title Case.
        
        ANALYSIS Rules (The Reporter):
        1. CRITICAL: Do NOT use print() for analysis. Use Streamlit functions directly:
           - st.subheader("Title")
           - st.write("Text explanation")
           - st.pyplot(plt)
        2. 7-Step Framework:
           - Answer the SMART questions using data.
           - Structure: Question -> Answer -> Chart.
        3. Create separate figures for each insight. Do not use subplots.
        4. Use st.markdown("---") to separate sections.
        
        VISUALIZATION Rules (The Designer):
        1. CRITICAL: Whenever you create a plot, ALWAYS set figsize=(12, 5). 
           - Example: plt.figure(figsize=(12, 5))
           - This ensures charts are wide and easy to read without scrolling too much.
        2. Style: Use sns.set_style("whitegrid") for a clean, professional look.
        3. Titles: Add a clear title to every chart using plt.title().
        
        OUTPUT FORMATTING:
        - Wrap code in markdown block.
        - Do not create sample data.
        - CRITICAL: Write the script to run immediately. Do NOT wrap the main logic in a function.
        """

        # Call Gemini
        generation_config = genai.types.GenerationConfig(max_output_tokens=8192)
        genai_model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)
        
        with st.spinner("Analyzing data..."):
            response = genai_model.generate_content(prompt)

        # Show Code
        with st.expander("See Generated Code"):
             st.code(response.text)

        # Execute
        cleaning_logs_placeholder = st.empty()
        st.write("### 📊 Strategic Report")
        cleaning_output = execute_ai_code(response.text, df)
        
        # Show Logs
        with cleaning_logs_placeholder.expander("Data Cleaning Logs"):
            st.text(cleaning_output)