# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import time
import json
import random
from pathlib import Path

# Import your optimization modules from the combopt package
from combopt import tsp, knapsack, matching
from combopt.utils import seed_everything

# --- PAGE CONFIG & STYLING ---
st.set_page_config(
    page_title="CombOpt Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { height: 3rem; padding: 0 2rem; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION & SESSION STATE ---
DATA_DIR = Path("benchmarks") / "datasets"
SAMPLE_DATA = {
    "TSP": DATA_DIR / "tsp_cities.csv",
    "Knapsack": DATA_DIR / "knapsack_items.csv",
    "Assignment": DATA_DIR / "assignment_cost.csv",
}

if 'results_history' not in st.session_state:
    st.session_state.results_history = []

# --- HEADER ---
st.markdown('<p class="main-header">🎯 CombOpt Studio</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Multi-Strategy Combinatorial Optimization Toolkit</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=CombOpt+Studio", width='stretch')
    
    st.markdown("### ⚙️ Configuration")
    problem_choice = st.selectbox(
        "Select Problem Type",
        ["🗺️ Traveling Salesman (TSP)", "🎒 Knapsack", "👥 Assignment Matching"],
        index=0
    )
    problem_key = problem_choice.split(" ")[1].replace("(", "").replace(")", "")

    seed_val = st.number_input("Random Seed", value=42, min_value=0, step=1)
    seed_everything(seed_val)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    show_comparison = st.checkbox("Enable Algorithm Comparison", value=False)
    
    st.markdown("---")
    st.markdown("### 📁 Quick Actions")
    if st.button("🗑️ Clear History", width='stretch'):
        st.session_state.results_history = []
        st.rerun()
    
    if st.session_state.results_history:
        st.download_button(
            "💾 Export Results (JSON)",
            data=json.dumps(st.session_state.results_history, indent=2, default=str),
            file_name="combopt_studio_results.json",
            mime="application/json",
            width='stretch'
        )

# --- MAIN CONTENT AREA ---

# -----------------
# TSP Problem Page
# -----------------
if problem_key == "TSP":
    st.markdown("## 🗺️ Traveling Salesman Problem")
    tab1, tab2, tab3 = st.tabs(["📍 Input Data", "🚀 Run Algorithms", "📈 Analysis"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Data Input Method")
            input_method = st.radio("", ["Use Sample Data", "Generate Random Cities", "Upload CSV", "Manual Entry"], horizontal=True, key="tsp_input")
            
            if input_method == "Generate Random Cities":
                n_cities = st.slider("Number of Cities", 5, 30, 15)
                if st.button("🎲 Generate Cities", width='stretch'):
                    st.session_state.tsp_points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_cities)]
            elif input_method == "Upload CSV":
                uploaded = st.file_uploader("Upload CSV (columns: x, y)", type=['csv'])
                if uploaded:
                    df = pd.read_csv(uploaded)
                    st.session_state.tsp_points = list(zip(df['x'], df['y']))
            elif input_method == "Use Sample Data":
                 if SAMPLE_DATA['TSP'].exists():
                     df = pd.read_csv(SAMPLE_DATA['TSP'])
                     st.session_state.tsp_points = list(zip(df['x'], df['y']))
                 else:
                     st.error(f"Sample data file not found at `{SAMPLE_DATA['TSP']}`! Please create it.")
            else:
                st.info("Enter coordinates in format: x1,y1; x2,y2; ...")
                coords_input = st.text_area("Coordinates", "60,200; 180,200; 80,180; 140,180; 20,160")
                if st.button("Parse Coordinates"):
                    try:
                        st.session_state.tsp_points = [tuple(map(float, p.strip().split(','))) for p in coords_input.split(';')]
                    except: st.error("Invalid format!")
        
        with col2:
            if 'tsp_points' in st.session_state:
                st.markdown("### Preview")
                st.success(f"✅ {len(st.session_state.tsp_points)} cities loaded")
                df = pd.DataFrame(st.session_state.tsp_points, columns=['x', 'y'])
                fig = px.scatter(df, x='x', y='y', title="City Locations")
                fig.update_traces(marker=dict(size=12, color='#667eea'))
                st.plotly_chart(fig, width='stretch')
    
    with tab2:
        if 'tsp_points' not in st.session_state:
            st.warning("⚠️ Please load data in the Input Data tab first!")
        else:
            points = st.session_state.tsp_points
            algos_info = {"Greedy (NN + 2-Opt)": tsp.tsp_greedy_nn_2opt, "Branch & Bound": tsp.tsp_branch_and_bound}
            
            if show_comparison:
                selected = st.multiselect("Choose algorithms to compare", list(algos_info.keys()), default=list(algos_info.keys()))
            else:
                selected = [st.selectbox("Choose algorithm", list(algos_info.keys()))]
            
            if st.button("🚀 Run Optimization", type="primary", width='stretch'):
                results = {}
                progress_bar = st.progress(0, text="Initializing...")
                for i, algo_name in enumerate(selected):
                    progress_bar.progress((i) / len(selected), text=f"Running {algo_name}...")
                    try:
                        result_obj = algos_info[algo_name](points)
                        results[algo_name] = result_obj.__dict__
                    except Exception as e:
                        st.error(f"Error in {algo_name}: {e}")
                progress_bar.progress(1.0, text="✅ Complete!")
                st.session_state.tsp_results = results
                st.session_state.results_history.append({'problem': 'TSP', 'timestamp': time.time(), 'results': results})
            
            if 'tsp_results' in st.session_state:
                st.markdown("--- \n### 🎯 Results")
                results = st.session_state.tsp_results
                cols = st.columns(len(results))
                for col, (name, data) in zip(cols, results.items()):
                    with col:
                        st.metric(name, f"{data['objective']:.2f}", f"{data['elapsed_sec']*1000:.2f}ms")
                
                st.markdown("### 🗺️ Tour Visualization")
                viz_algo = st.selectbox("Select tour to visualize", list(results.keys()))
                if viz_algo:
                    tour = results[viz_algo]['solution']
                    tour_points = [points[i] for i in tour] + [points[tour[0]]]
                    fig_tour = go.Figure(go.Scatter(x=[p[0] for p in tour_points], y=[p[1] for p in tour_points], mode='lines+markers', line=dict(color='#667eea', width=2), marker=dict(size=10, color='#764ba2')))
                    for i, (x, y) in enumerate(points):
                        fig_tour.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(color='white', size=10))
                    fig_tour.update_layout(title=f"{viz_algo} - Tour Length: {results[viz_algo]['objective']:.2f}", xaxis_title="X", yaxis_title="Y")
                    st.plotly_chart(fig_tour, width='stretch')

    with tab3:
        if 'tsp_results' in st.session_state:
            st.markdown("### 📊 Performance Analysis")
            results = st.session_state.tsp_results
            df_analysis = pd.DataFrame([{'Algorithm': name, 'Tour Length': data['objective'], 'Time (ms)': data['elapsed_sec'] * 1000} for name, data in results.items()])
            st.dataframe(df_analysis, width='stretch')
            
            fig = px.scatter(df_analysis, x='Time (ms)', y='Tour Length', text='Algorithm', title='Solution Quality vs Speed Trade-off')
            fig.update_traces(marker=dict(size=15, color='#667eea'), textposition='top center')
            st.plotly_chart(fig, width='stretch')

# --------------------
# Knapsack Problem Page
# --------------------
elif problem_key == "Knapsack":
    st.markdown("## 🎒 Knapsack Problem")
    tab1, tab2, tab3 = st.tabs(["📦 Input Data", "🚀 Run Algorithms", "📈 Analysis"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.session_state.knapsack_capacity = st.number_input("Knapsack Capacity", min_value=1, value=100, step=1)
            input_method = st.radio("Data Input Method", ["Use Sample Data", "Generate Random Items", "Upload CSV", "Manual Entry"], horizontal=True, key="knap_input")

            if input_method == "Generate Random Items":
                n_items = st.slider("Number of Items", 5, 50, 20)
                if st.button("🎲 Generate Items", width='stretch'):
                    st.session_state.knapsack_items = [(random.randint(5, 100), random.randint(1, 30)) for _ in range(n_items)]
            elif input_method == "Upload CSV":
                uploaded = st.file_uploader("Upload CSV (columns: value, weight)", type=['csv'])
                if uploaded:
                    df = pd.read_csv(uploaded)
                    st.session_state.knapsack_items = list(zip(df['value'], df['weight']))
            elif input_method == "Use Sample Data":
                if SAMPLE_DATA['Knapsack'].exists():
                    df = pd.read_csv(SAMPLE_DATA['Knapsack'])
                    st.session_state.knapsack_items = list(zip(df['value'], df['weight']))
                else:
                    st.error(f"Sample data file not found at `{SAMPLE_DATA['Knapsack']}`! Please create it.")
            else:
                st.info("Enter items as value,weight pairs (one per line)")
                items_input = st.text_area("Items", "22,8\n20,12\n15,7\n30,15\n25,10")
                if st.button("Parse Items"):
                    try:
                        st.session_state.knapsack_items = [tuple(map(int, line.split(','))) for line in items_input.strip().split('\n')]
                    except: st.error("Invalid format!")
        
        with col2:
            if 'knapsack_items' in st.session_state:
                items = st.session_state.knapsack_items
                st.markdown("### Preview")
                st.success(f"✅ {len(items)} items loaded")
                st.metric("Total Value of All Items", sum(v for v, w in items))
                st.metric("Total Weight of All Items", sum(w for v, w in items))
                df_items = pd.DataFrame(items, columns=['Value', 'Weight'])
                df_items['Ratio'] = df_items.apply(lambda row: row['Value'] / row['Weight'] if row['Weight'] > 0 else 0, axis=1)
                fig = px.scatter(df_items, x='Weight', y='Value', title='Items Distribution', color='Ratio', color_continuous_scale='Viridis')
                st.plotly_chart(fig, width='stretch')

    with tab2:
        if 'knapsack_items' not in st.session_state:
            st.warning("⚠️ Please load data in the Input Data tab first!")
        else:
            items, capacity = st.session_state.knapsack_items, st.session_state.knapsack_capacity
            algos_info = {
                "Greedy (Ratio)": knapsack.knapsack_greedy, "Divide & Conquer": knapsack.knapsack_divide_conquer,
                "Dynamic Programming": knapsack.knapsack_dp, "Branch & Bound": knapsack.knapsack_branch_and_bound
            }
            
            if show_comparison:
                selected = st.multiselect("Choose algorithms", list(algos_info.keys()), default=["Greedy (Ratio)", "Dynamic Programming"])
            else:
                selected = [st.selectbox("Choose algorithm", list(algos_info.keys()))]
            
            if st.button("🚀 Run Optimization", type="primary", width='stretch'):
                results = {}
                progress_bar = st.progress(0, text="Initializing...")
                for i, algo_name in enumerate(selected):
                    progress_bar.progress(i / len(selected), text=f"Running {algo_name}...")
                    start_time = time.perf_counter()
                    take, value, meta = algos_info[algo_name](items, capacity)
                    elapsed = time.perf_counter() - start_time
                    results[algo_name] = {'solution': take, 'objective': value, 'elapsed_sec': elapsed, 'meta': meta, 'weight': sum(items[i][1] for i, t in enumerate(take) if t)}
                progress_bar.progress(1.0, text="✅ Complete!")
                st.session_state.knapsack_results = results
                st.session_state.results_history.append({'problem': 'Knapsack', 'timestamp': time.time(), 'results': {k: {k2: v2 for k2, v2 in v.items() if k2 != 'solution'} for k, v in results.items()}})
            
            if 'knapsack_results' in st.session_state:
                st.markdown("--- \n### 🎯 Results")
                results = st.session_state.knapsack_results
                cols = st.columns(len(results))
                for col, (name, data) in zip(cols, results.items()):
                    with col:
                        st.metric(f"{name}", f"Value: {data['objective']:.0f}")
                        st.caption(f"Weight: {data['weight']}/{capacity} | Time: {data['elapsed_sec']*1000:.2f}ms")
                
                st.markdown("### 📦 Selected Items Visualization")
                viz_algo = st.selectbox("Algorithm to visualize", list(results.keys()))
                if viz_algo:
                    take = results[viz_algo]['solution']
                    df_viz = pd.DataFrame([{'Item': i, 'Value': items[i][0], 'Selected': 'Yes' if take[i] else 'No'} for i in range(len(items))])
                    fig = px.bar(df_viz, x='Item', y='Value', color='Selected', title=f"{viz_algo} - Total Value: {results[viz_algo]['objective']}", color_discrete_map={'Yes': '#667eea', 'No': '#cccccc'})
                    st.plotly_chart(fig, width='stretch')

    with tab3:
        if 'knapsack_results' in st.session_state:
            st.markdown("### 📊 Performance Analysis")
            results, capacity = st.session_state.knapsack_results, st.session_state.knapsack_capacity
            df_analysis = pd.DataFrame([{'Algorithm': name, 'Value': data['objective'], 'Weight Used': data['weight'], 'Capacity Util %': (data['weight']/capacity*100), 'Time (ms)': data['elapsed_sec']*1000} for name, data in results.items()])
            st.dataframe(df_analysis, width='stretch')
            
            fig = go.Figure(data=go.Parcoords(dimensions=[dict(label='Value', values=df_analysis['Value']), dict(label='Weight Used', values=df_analysis['Weight Used']), dict(label='Time (ms)', values=df_analysis['Time (ms)'])], line=dict(color=df_analysis['Value'], colorscale='Viridis')))
            fig.update_layout(title='Multi-dimensional Performance View')
            st.plotly_chart(fig, width='stretch')

# -----------------------
# Assignment Problem Page
# -----------------------
elif problem_key == "Assignment":
    st.markdown("## 👥 Assignment Matching")
    tab1, tab2, tab3 = st.tabs(["📋 Input Data", "🚀 Run Algorithms", "📈 Analysis"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            input_method = st.radio("Data Input", ["Use Sample Data", "Generate Random Matrix", "Upload CSV"], horizontal=True, key="assign_input")
            if input_method == "Generate Random Matrix":
                n = st.slider("Matrix Size (n×n)", 3, 12, 5)
                if st.button("🎲 Generate Cost Matrix", width='stretch'):
                    st.session_state.assignment_matrix = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
            elif input_method == "Upload CSV":
                uploaded = st.file_uploader("Upload CSV (square cost matrix, no header)", type=['csv'])
                if uploaded:
                    st.session_state.assignment_matrix = pd.read_csv(uploaded, header=None).values.tolist()
            elif input_method == "Use Sample Data":
                if SAMPLE_DATA['Assignment'].exists():
                     st.session_state.assignment_matrix = pd.read_csv(SAMPLE_DATA['Assignment'], header=None).values.tolist()
                else:
                    st.error(f"Sample data file not found at `{SAMPLE_DATA['Assignment']}`! Please create it.")

        with col2:
            if 'assignment_matrix' in st.session_state:
                matrix = st.session_state.assignment_matrix
                st.markdown("### Preview")
                st.success(f"✅ {len(matrix)}×{len(matrix)} matrix loaded")
                fig = go.Figure(data=go.Heatmap(z=matrix, colorscale='RdYlGn_r', text=matrix, texttemplate='%{text}'))
                fig.update_layout(title="Cost Matrix", xaxis_title="Tasks", yaxis_title="Workers")
                st.plotly_chart(fig, width='stretch')

    with tab2:
        if 'assignment_matrix' not in st.session_state:
            st.warning("⚠️ Please load data first!")
        else:
            matrix = st.session_state.assignment_matrix
            algos_info = {"Greedy": matching.assignment_greedy, "Hungarian": matching.assignment_hungarian}
            
            if show_comparison:
                selected = st.multiselect("Algorithms", list(algos_info.keys()), default=list(algos_info.keys()))
            else:
                selected = [st.selectbox("Algorithm", list(algos_info.keys()))]
            
            if st.button("🚀 Run Optimization", type="primary", width='stretch'):
                results = {}
                progress_bar = st.progress(0, text="Initializing...")
                for i, algo_name in enumerate(selected):
                    progress_bar.progress(i / len(selected), text=f"Running {algo_name}...")
                    start_time = time.perf_counter()
                    assignment, cost, meta = algos_info[algo_name](matrix)
                    elapsed = time.perf_counter() - start_time
                    results[algo_name] = {'solution': assignment, 'objective': cost, 'elapsed_sec': elapsed, 'meta': meta}
                progress_bar.progress(1.0, text="✅ Complete!")
                st.session_state.assignment_results = results
                st.session_state.results_history.append({'problem': 'Assignment', 'timestamp': time.time(), 'results': {k: {k2: v2 for k2, v2 in v.items() if k2 != 'solution'} for k, v in results.items()}})
            
            if 'assignment_results' in st.session_state:
                st.markdown("--- \n### 🎯 Results")
                results = st.session_state.assignment_results
                cols = st.columns(len(results))
                for col, (name, data) in zip(cols, results.items()):
                    with col:
                        st.metric(name, f"Cost: {data['objective']:.2f}", f"{data['elapsed_sec']*1000:.2f}ms")
                
                st.markdown("### 🔗 Assignment Visualization")
                viz_algo = st.selectbox("Algorithm to visualize", list(results.keys()))
                if viz_algo:
                    assignment = results[viz_algo]['solution']
                    fig = go.Figure(data=go.Heatmap(z=matrix, colorscale='Viridis', showscale=False))
                    for worker, task in enumerate(assignment):
                        fig.add_shape(type="rect", x0=task-0.5, y0=worker-0.5, x1=task+0.5, y1=worker+0.5, line=dict(color="Red", width=3))
                    fig.update_layout(title=f"{viz_algo} - Total Cost: {results[viz_algo]['objective']:.2f}", xaxis_title="Tasks", yaxis_title="Workers")
                    st.plotly_chart(fig, width='stretch')

    with tab3:
        if 'assignment_results' in st.session_state:
            st.markdown("### 📊 Algorithm Comparison")
            results = st.session_state.assignment_results
            df_comp = pd.DataFrame([{'Algorithm': name, 'Total Cost': data['objective'], 'Time (ms)': data['elapsed_sec']*1000} for name, data in results.items()])
            st.dataframe(df_comp, width='stretch')
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Cost Comparison", "Time Comparison"))
            fig.add_trace(go.Bar(x=df_comp['Algorithm'], y=df_comp['Total Cost'], marker_color='#667eea'), 1, 1)
            fig.add_trace(go.Bar(x=df_comp['Algorithm'], y=df_comp['Time (ms)'], marker_color='#764ba2'), 1, 2)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 2rem;'>🎯 <strong>CombOpt Studio</strong></div>", unsafe_allow_html=True)