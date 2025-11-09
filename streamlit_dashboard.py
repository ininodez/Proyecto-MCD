import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(layout="wide", page_title="Dashboard de Ciberseguridad - IPS Output")

CSV_PATH = r"C:\Users\DmP\Desktop\Gemma-MCD\input\Empresa Sanitaria IPS_final_priorizado.csv"

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    return df

df = load_data(CSV_PATH)

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
if "event_time" in df.columns:
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

st.title("Dashboard de Seguridad - IPS / Threat Intelligence")

st.sidebar.header("Filtros")
countries = st.sidebar.multiselect("País de IP", sorted(df["ip_country"].dropna().unique()) if "ip_country" in df.columns else [])
severity = st.sidebar.multiselect("Severidad", sorted(df["severity"].dropna().unique()) if "severity" in df.columns else [])
policy = st.sidebar.multiselect("Política", sorted(df["policy_name"].dropna().unique()) if "policy_name" in df.columns else [])

filtered = df.copy()
if countries:
    filtered = filtered[filtered["ip_country"].isin(countries)]
if severity:
    filtered = filtered[filtered["severity"].isin(severity)]
if policy:
    filtered = filtered[filtered["policy_name"].isin(policy)]

st.markdown("## Indicadores Clave de Seguridad")
col1, col2, col3, col4 = st.columns(4)
total_events = len(filtered)
avg_severity = round(filtered["severity"].astype(float).mean(), 2) if "severity" in filtered.columns else "N/A"
malicious = filtered["vt_malicious"].sum() if "vt_malicious" in filtered.columns else 0
suspicious = filtered["vt_suspicious"].sum() if "vt_suspicious" in filtered.columns else 0
malicious_pct = malicious / (malicious + suspicious + 1e-5) * 100
unique_ips = filtered["source_ip"].nunique() if "source_ip" in filtered.columns else 0

col1.metric("Total de eventos", total_events)
col2.metric("Severidad promedio", avg_severity)
col3.metric("% Maliciosos", f"{malicious_pct:.1f}%")
col4.metric("IPs únicas origen", unique_ips)

if "severity" in filtered.columns:
    st.markdown("## Distribución por Severidad")
    fig_sev = px.histogram(filtered, x="severity", nbins=10, color_discrete_sequence=["#E45756"], title="Distribución de Severidad")
    st.plotly_chart(fig_sev, use_container_width=True)

if "signature_name" in filtered.columns:
    st.markdown("## Top 10 Firmas Detectadas")
    top_sigs = filtered["signature_name"].value_counts().head(10).reset_index()
    top_sigs.columns = ["signature_name", "count"]
    fig_sigs = px.bar(top_sigs, x="count", y="signature_name", orientation="h", title="Firmas más detectadas")
    st.plotly_chart(fig_sigs, use_container_width=True)

if {"ip_country", "ip_lat", "ip_lon"}.issubset(filtered.columns):
    st.markdown("## Mapa de Actividad por Ubicación IP")
    map_df = filtered.dropna(subset=["ip_lat", "ip_lon"])
    fig_map = px.scatter_geo(
        map_df,
        lat="ip_lat",
        lon="ip_lon",
        color="severity" if "severity" in map_df.columns else None,
        hover_name="source_ip",
        hover_data=["ip_country", "ip_city", "ip_org", "vt_malicious", "vt_reputation"],
        title="Ubicación Geográfica de IPs Origen",
    )
    st.plotly_chart(fig_map, use_container_width=True)

if "event_time" in filtered.columns:
    st.markdown("## Tendencia Temporal de Eventos")
    ts = filtered.groupby(pd.Grouper(key="event_time", freq="D")).size().reset_index(name="Eventos")
    fig_ts = px.line(ts, x="event_time", y="Eventos", title="Eventos por Día")
    st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("## Casos Relevantes y Diagnóstico")
cols_show = ["event_time", "source_ip", "destination", "ip_country", "severity", "signature_name", "diagnostico"]
if all(col in filtered.columns for col in cols_show):
    st.dataframe(filtered[cols_show].sort_values(by="event_time", ascending=False).head(20))
else:
    st.dataframe(filtered.head(20))

if "server_name" in filtered.columns:
    st.markdown("## Servidores Más Atacados")
    top_servers = filtered["server_name"].value_counts().reset_index()
    top_servers.columns = ["server_name", "cantidad"]
    fig_servers = px.bar(top_servers.head(15), x="server_name", y="cantidad", title="Servidores más atacados", text="cantidad")
    st.plotly_chart(fig_servers, use_container_width=True)

if "server_name" in filtered.columns and "conectado_a" in filtered.columns:
    st.subheader("Relación entre Servidores y Conexiones Detectadas")
    G = nx.from_pandas_edgelist(filtered, source="server_name", target="conectado_a")
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        count = filtered[filtered["server_name"] == node].shape[0]
        node_text.append(f"{node} — {count} eventos")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', textposition="top center", hovertext=node_text,
        marker=dict(showscale=True, colorscale='YlOrRd',
                    color=[filtered[filtered["server_name"] == n].shape[0] for n in G.nodes()],
                    size=[max(8, filtered[filtered["server_name"] == n].shape[0] / 3) for n in G.nodes()],
                    colorbar=dict(title="Eventos")),
        text=[n for n in G.nodes()]
    )

    fig_network = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(title="Mapa de conexiones entre servidores",
                                             showlegend=False, hovermode='closest',
                                             margin=dict(b=0, l=0, r=0, t=30),
                                             xaxis=dict(showgrid=False, zeroline=False),
                                             yaxis=dict(showgrid=False, zeroline=False)))
    st.plotly_chart(fig_network, use_container_width=True)

csv_export = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV filtrado", data=csv_export, file_name="seguridad_filtrado.csv", mime="text/csv")

st.markdown("---")
st.markdown("_Dashboard SOC generado automáticamente con IA_")
