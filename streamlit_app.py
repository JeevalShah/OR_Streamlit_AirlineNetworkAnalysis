"""
Airline Network Optimization Dashboard (fixed)
Using Graph Theory and Operations Research Concepts
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration (must be before other streamlit UI calls)
st.set_page_config(
    page_title="Airline Network Optimization",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ Airline Network Optimization Dashboard")
st.markdown("""
This tool applies **Operations Research** concepts and **Graph Theory** to analyze airline networks.
Upload your airline data to find optimal routes, identify hub airports, and analyze network connectivity.
""")

# Sidebar
st.sidebar.header("📊 Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Airline Dataset (CSV)",
    type=['csv'],
    help="Upload a CSV file with columns: origin, dest, sched_dep_time, sched_arr_time, dep_time, arr_time, air_time, year, month, day"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'graph' not in st.session_state:
    st.session_state.graph = None

def format_time_field(val):
    """Given a numeric or string time like 500, 515, 1230, '  5', return HH:MM:SS or NaT for missing."""
    if pd.isna(val) or val == 0 or str(val).strip() == '':
        return pd.NaT
    try:
        s = str(int(val)).zfill(4)
        hh = s[:2]
        mm = s[2:]
        return f"{hh}:{mm}:00"
    except Exception:
        return pd.NaT

def preprocess_data(df):
    """Preprocess the airline dataset robustly."""
    try:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        required = {'origin', 'dest'}
        if not required.issubset(set(df.columns)):
            st.error(f"Dataset must contain columns: {required}")
            return None

        for col_in, col_out in [('sched_dep_time', 'std'), ('sched_arr_time', 'sta'),
                                ('dep_time', 'atd'), ('arr_time', 'ata')]:
            if col_in in df.columns:
                df[col_out] = df[col_in].apply(format_time_field)
            else:
                df[col_out] = pd.NaT

        if all(col in df.columns for col in ['year', 'month', 'day']):
            try:
                df['date'] = pd.to_datetime(df[['year','month','day']])
            except Exception:
                df['date'] = pd.NaT

        if 'air_time' in df.columns:
            df['air_time'] = pd.to_numeric(df['air_time'], errors='coerce')

        df['origin'] = df['origin'].astype(str).str.strip()
        df['dest'] = df['dest'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

def create_graph(df):
    """Create a directed graph with aggregated 'air_time' as edge weight."""
    try:
        if 'air_time' in df.columns:
            agg = df.groupby(['origin','dest'], as_index=False).agg({'air_time': lambda x: np.nanmean(x.values)})
        else:
            agg = df[['origin','dest']].drop_duplicates().copy()
            agg['air_time'] = np.nan

        G = nx.DiGraph()
        for _, row in agg.iterrows():
            u, v = row['origin'], row['dest']
            if not np.isnan(row['air_time']):
                G.add_edge(u, v, air_time=float(row['air_time']))
            else:
                G.add_edge(u, v)
        return G
    except Exception as e:
        st.error(f"Error creating graph: {e}")
        return None

def calculate_network_metrics(G):
            try:
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)
                return {
                    'degree_centrality': degree_centrality,
                    'betweenness_centrality': betweenness_centrality,
                    'closeness_centrality': closeness_centrality
                }
            except Exception as e:
                st.error(f"Error calculating network metrics: {e}")
                return {'degree_centrality': {}, 'betweenness_centrality': {}, 'closeness_centrality': {}}

def plot_network_interactive(G, highlight_nodes=None, highlight_edges=None):
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color='#888'),
                            hoverinfo='none', mode='lines', name='Edges')
    node_x, node_y, node_text, node_size = [], [], [], []
    degs = dict(G.degree())
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{n}<br>Degree: {degs.get(n,0)}")
        node_size.append(10 + degs.get(n,0) * 4)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=[n for n in G.nodes()],
        textposition="top center", hovertext=node_text, hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlOrRd', size=node_size,
                    color=[degs.get(n,0) for n in G.nodes()],
                    colorbar=dict(title='Node Degree'),
                    line=dict(width=1, color='white')), name='Airports'
    )
    data = [edge_trace, node_trace]
    if highlight_edges and len(highlight_edges) >= 2:
        path_edge_x, path_edge_y = [], []
        for i in range(len(highlight_edges) - 1):
            a, b = highlight_edges[i], highlight_edges[i+1]
            if a in pos and b in pos:
                x0, y0 = pos[a]
                x1, y1 = pos[b]
                path_edge_x.extend([x0, x1, None])
                path_edge_y.extend([y0, y1, None])
        if path_edge_x:
            path_trace = go.Scatter(x=path_edge_x, y=path_edge_y,
                                    line=dict(width=4, color='red'),
                                    hoverinfo='none', mode='lines',
                                    name='Highlighted Path')
            data.append(path_trace)
    fig = go.Figure(data=data)
    fig.update_layout(title='Airline Network Graph', showlegend=True, hovermode='closest',
                      margin=dict(b=0, l=0, r=0, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      height=600, plot_bgcolor='white')
    return fig


# Main app logic
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None
    if df is not None:
        df = preprocess_data(df)
        if df is not None:
            st.session_state.data = df
            G = create_graph(df)
            st.session_state.graph = G
            st.success(f"✅ Data loaded successfully! {len(df)} flights, {G.number_of_nodes()} airports")
        else:
            st.stop()
    else:
        st.stop()

    df = st.session_state.data
    G = st.session_state.graph
    if G is None or G.number_of_nodes() == 0:
        st.error("Graph could not be constructed (no nodes). Check your dataset.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Network Overview",
        "🗺️ Route Analysis",
        "🏢 Hub Identification",
        "🔍 Path Optimization",
        "💥 Resilience Analysis"
    ])

    # TAB 1
    with tab1:
        st.header("Network Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Airports", G.number_of_nodes())
        with col2:
            st.metric("Total Routes", G.number_of_edges())
        with col3:
            st.metric("Network Density", f"{nx.density(G.to_undirected()):.4f}")
        with col4:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.metric("Avg Degree", f"{avg_degree:.2f}")
        st.subheader("Network Visualization")
        fig = plot_network_interactive(G)
        st.plotly_chart(fig, use_container_width=True, key="overview_network")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

    # TAB 2
    with tab2:
        st.header("Route Analysis")
        metrics = calculate_network_metrics(G)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Network Connectivity")
            asp = metrics.get('avg_shortest_path', 'N/A')
            st.write(f"**Average Shortest Path Length:** {asp}")
            if 'avg_shortest_path_note' in metrics:
                st.info(metrics['avg_shortest_path_note'])
            degrees = [G.degree(n) for n in G.nodes()]
            fig_deg = go.Figure(data=[go.Histogram(x=degrees, nbinsx=20)])
            fig_deg.update_layout(title="Degree Distribution", xaxis_title="Degree", yaxis_title="Frequency", height=300)
            st.plotly_chart(fig_deg, use_container_width=True, key="degree_hist")
        with col2:
            st.subheader("Top 10 Connected Airports")
            degree_df = pd.DataFrame(sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:10], columns=['Airport', 'Number of Connections'])
            st.dataframe(degree_df, use_container_width=True, hide_index=True)
            fig_bar = go.Figure(data=[go.Bar(x=degree_df['Airport'], y=degree_df['Number of Connections'])])
            fig_bar.update_layout(xaxis_title="Airport", yaxis_title="Connections", height=300)
            st.plotly_chart(fig_bar, use_container_width=True, key="top10_bar")

    # TAB 3
    with tab3:
        st.header("Hub Airport Identification")
        st.markdown("Using **centrality measures** to identify critical hub airports")

        # 🧠 Ensure the graph exists and has edges
        if 'G' not in locals() or len(G.nodes) == 0 or len(G.edges) == 0:
            st.warning("⚠️ No network graph found or it is empty. Please load your data first.")
        else:
            # ✅ Safe network metrics calculation

            metrics = calculate_network_metrics(G)

            col1, col2 = st.columns(2)

            # ---- DEGREE CENTRALITY ----
            with col1:
                st.subheader("Degree Centrality (Top 10)")
                degree_cent = metrics.get('degree_centrality', {})
                if degree_cent:
                    degree_cent_df = pd.DataFrame(
                        sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10],
                        columns=['Airport', 'Degree Centrality']
                    )
                    degree_cent_df['Degree Centrality'] = degree_cent_df['Degree Centrality'].round(4)
                    st.dataframe(degree_cent_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No Degree Centrality data available.")

            # ---- BETWEENNESS CENTRALITY ----
            with col2:
                st.subheader("Betweenness Centrality (Top 10)")
                between_cent = metrics.get('betweenness_centrality', {})
                if between_cent:
                    between_cent_df = pd.DataFrame(
                        sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:10],
                        columns=['Airport', 'Betweenness Centrality']
                    )
                    between_cent_df['Betweenness Centrality'] = between_cent_df['Betweenness Centrality'].round(4)
                    st.dataframe(between_cent_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No Betweenness Centrality data available.")

            # ---- HUB COMPARISON ----
            st.subheader("Hub Comparison")

            if degree_cent and between_cent:
                top_hubs = list(degree_cent_df['Airport'][:5]) if not degree_cent_df.empty else []
                top_b = list(between_cent_df['Airport'][:5]) if not between_cent_df.empty else []
                top_combined = list(set(top_hubs + top_b))

                hub_comp = pd.DataFrame({
                    'Airport': top_combined,
                    'Degree Centrality': [degree_cent.get(h, 0) for h in top_combined],
                    'Betweenness Centrality': [between_cent.get(h, 0) for h in top_combined]
                })

                if not hub_comp.empty:
                    fig_hub = go.Figure()
                    fig_hub.add_trace(go.Bar(
                        x=hub_comp['Airport'],
                        y=hub_comp['Degree Centrality'],
                        name='Degree Centrality'
                    ))
                    fig_hub.add_trace(go.Bar(
                        x=hub_comp['Airport'],
                        y=hub_comp['Betweenness Centrality'],
                        name='Betweenness Centrality'
                    ))
                    fig_hub.update_layout(
                        barmode='group',
                        title='Hub Airport Centrality Comparison',
                        xaxis_title='Airport',
                        yaxis_title='Centrality Score',
                        height=400
                    )
                    st.plotly_chart(fig_hub, use_container_width=True, key="hub_comparison_unique")
                else:
                    st.info("Not enough data to display hub comparison.")
            else:
                st.info("Centrality metrics missing — cannot compute hub comparison.")


    # TAB 4
    with tab4:
        st.header("Shortest Path Analysis (Dijkstra's Algorithm)")
        airports = sorted(list(G.nodes()))
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source Airport", airports, key='source')
        with col2:
            dest = st.selectbox("Destination Airport", airports, index=min(5, len(airports)-1), key='dest')
        if source and dest and source != dest:
            st.subheader(f"Route Analysis: {source} → {dest}")
            try:
                und = G.to_undirected()
                if nx.has_path(und, source, dest):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Unweighted Shortest Path")
                        unweighted_path = nx.shortest_path(und, source, dest)
                        st.success(f"**Path:** {' → '.join(unweighted_path)}")
                        st.info(f"**Number of stops:** {len(unweighted_path)-1}")
                    with c2:
                        st.markdown("### Weighted Shortest Path")
                        has_air_time = any('air_time' in data for _, _, data in G.edges(data=True))
                        if has_air_time:
                            try:
                                weighted_path = nx.dijkstra_path(G, source, dest, weight='air_time')
                                total_time = nx.dijkstra_path_length(G, source, dest, weight='air_time')
                                st.success(f"**Path:** {' → '.join(weighted_path)}")
                                st.info(f"**Total air time:** {total_time:.0f} minutes")
                            except nx.NetworkXNoPath:
                                st.warning("No weighted (air_time) path exists between the chosen airports.")
                                weighted_path = unweighted_path
                        else:
                            st.warning("Air time data not available.")
                            weighted_path = unweighted_path
                    st.subheader("Path Visualization")
                    fig_path = plot_network_interactive(G, highlight_edges=unweighted_path)
                    st.plotly_chart(fig_path, use_container_width=True, key="path_viz")
            except Exception as e:
                st.error(f"Error finding path: {e}")
        else:
            st.info("Please select different source and destination airports")

    # TAB 5
    with tab5:
        st.header("Network Resilience Analysis")
        airports = sorted(list(G.nodes()))
        disrupted_airport = st.selectbox("Select airport to simulate disruption", airports)
        if st.button("Simulate Disruption", type="primary"):
            with st.spinner("Analyzing network resilience..."):
                G_disrupted = G.copy()
                if disrupted_airport in G_disrupted:
                    G_disrupted.remove_node(disrupted_airport)
                else:
                    st.error("Selected airport not in graph")
                    st.stop()
                col1, col2, col3 = st.columns(3)
                with col1:
                    removed_nodes = st.session_state.graph.number_of_nodes() - G_disrupted.number_of_nodes()
                    st.metric("Airports Affected", removed_nodes)
                with col2:
                    removed_edges = st.session_state.graph.number_of_edges() - G_disrupted.number_of_edges()
                    st.metric("Routes Affected", removed_edges)
                with col3:
                    num_components_before = nx.number_connected_components(st.session_state.graph.to_undirected())
                    num_components_after = nx.number_connected_components(G_disrupted.to_undirected())
                    st.metric("Connected Components", num_components_after, delta=num_components_after - num_components_before)
                st.subheader("Impact Analysis")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Original Network")
                    fig_original = plot_network_interactive(st.session_state.graph)
                    st.plotly_chart(fig_original, use_container_width=True, key="original_net")
                with c2:
                    st.markdown("### After Disruption")
                    fig_disrupted = plot_network_interactive(G_disrupted)
                    st.plotly_chart(fig_disrupted, use_container_width=True, key="disrupted_net")
else:
    st.info("👆 Please upload an airline dataset CSV file to begin analysis")