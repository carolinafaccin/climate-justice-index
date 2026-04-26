import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import leafmap.foliumap as leafmap
import folium
import branca.colormap as branca_cm
from branca.element import MacroElement
from jinja2 import Template
from pathlib import Path
from shapely.geometry import Polygon
import h3
from src import config as cfg

# --- LÓGICA DE SENHA ---
def check_password():
    def password_entered():
        if st.session_state.get("password") == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    _, col, _ = st.columns([2, 1, 2])
    if "password_correct" not in st.session_state:
        with col:
            st.text_input(
                "Digite a senha para acessar o Índice:", type="password", on_change=password_entered, key="password"
            )
        return False
    elif not st.session_state["password_correct"]:
        with col:
            st.text_input(
                "Digite a senha para acessar o Índice:", type="password", on_change=password_entered, key="password"
            )
            st.error("Senha incorreta.")
        return False
    else:
        return True

st.set_page_config(layout="wide", page_title="Índice de Injustiça Climática", initial_sidebar_state="expanded")

st.markdown("""<style>
[data-testid="stAlert"] {
    background-color: #FEF3C7 !important;
    border-left-color: #F0AB00 !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] span {
    color: #92680B !important;
}
[data-testid="stAlert"] svg {
    fill: #F0AB00 !important; color: #F0AB00 !important;
}
</style>""", unsafe_allow_html=True)

if not check_password():
    st.stop()

# ==============================================================================
# FONTES E TEMA
# ==============================================================================
def apply_fonts():
    st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeading"] {
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: 700 !important;
    }

    p, label, li,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    .stSelectbox label, .stTextInput label {
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: 400 !important;
    }

    /* Caixas de alerta — esquema âmbar unificado */
    [data-testid="stAlert"] {
        background-color: #FEF3C7 !important;
        border-left-color: #F0AB00 !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #92680B !important;
    }
    [data-testid="stAlert"] svg {
        fill: #F0AB00 !important;
        color: #F0AB00 !important;
    }

    /* Reduz espaçamento superior da área principal para alinhar com a sidebar */
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
    }

    /* Oculta toolbar e status widget */
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }

    /* Mantém a sidebar sempre visível, ignorando o estado salvo no localStorage */
    section[data-testid="stSidebar"] {
        transform: none !important;
        min-width: 244px !important;
        visibility: visible !important;
    }

    /* Oculta os botões de abrir/fechar a sidebar */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 1. CARREGAMENTO DE DADOS
# ==============================================================================
def find_latest_dashboard() -> str:
    """Retorna o parquet de dashboard mais recente na pasta de resultados do repo."""
    results_dir = cfg.FILES['output']['repo_results_dir']
    files = sorted(results_dir.glob(f"{cfg.DASHBOARD_FILE_PREFIX}_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum parquet de dashboard encontrado em {results_dir}. "
            "Execute `python main.py` para gerar."
        )
    return str(files[-1])  # nome com timestamp → ordem alfabética = cronológica

@st.cache_data(show_spinner=False)
def load_data():
    path = find_latest_dashboard()
    return pd.read_parquet(path)

_, _col_load, _ = st.columns([2, 1, 2])
with _col_load:
    with st.spinner("Carregando base de dados..."):
        df_brasil = load_data()

DIM_COLS = {
    'ip': 'Grupos Prioritários',
    'iv': 'Vulnerabilidade',
    'ie': 'Exposição',
    'ig': 'Capacidade de Gestão Municipal',
}

# ==============================================================================
# 2. BARRA LATERAL
# ==============================================================================
st.sidebar.header("Selecione o município")

ufs  = sorted(df_brasil['nm_uf'].dropna().unique())
uf_sel = st.sidebar.selectbox("Estado", ufs)

muns = sorted(df_brasil[df_brasil['nm_uf'] == uf_sel]['nm_mun'].dropna().unique())
mun_sel = st.sidebar.selectbox("Município", muns)

apply_fonts()
plot_tpl = "plotly_white"

# ==============================================================================
# CABEÇALHO
# ==============================================================================
col_intro, _ = st.columns([3, 2])
with col_intro:
    st.markdown(
        "<h1 style=\"font-weight:700;\">Índice de Injustiça Climática para municípios brasileiros</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "O Índice de Injustiça Climática (IIC) mensura desigualdades territoriais frente aos riscos climáticos, "
        "combinando exposição a eventos extremos, vulnerabilidade socioeconômica, presença de grupos populacionais "
        "prioritários e capacidade de gestão municipal. "
        "Desenvolvido pelo WRI Brasil, o índice é calculado em grade hexagonal H3 de alta resolução (~0,1 km²) "
        "para todos os municípios brasileiros."
    )

if st.sidebar.button("Gerar mapa", type="primary"):
    st.session_state['gerar_mapa'] = True
    st.session_state['sel_uf'] = uf_sel
    st.session_state['sel_mun'] = mun_sel

if st.session_state.get('gerar_mapa', False):
    uf_sel = st.session_state['sel_uf']
    mun_sel = st.session_state['sel_mun']

    # --------------------------------------------------------------------------
    # Filtragem
    # --------------------------------------------------------------------------
    df_city = df_brasil[
        (df_brasil['nm_uf'] == uf_sel) & (df_brasil['nm_mun'] == mun_sel)
    ].copy()

    if df_city.empty:
        st.error("Nenhum dado encontrado para essa seleção.")
        st.stop()

    # Contexto nacional
    city_mean    = df_city['iic_final'].mean()
    national_pct = (df_brasil['iic_final'] < city_mean).mean() * 100

    # --------------------------------------------------------------------------
    # TABS
    # --------------------------------------------------------------------------
    tab_mapa, tab_analise = st.tabs(["Mapa", "Análise"])

    # ==========================================================================
    # TAB 1 — MAPA
    # ==========================================================================
    with tab_mapa:
        st.info(f"Carregando {len(df_city):,} hexágonos para {mun_sel} — {uf_sel}...")

        def get_geometry(h3_id):
            try:
                boundary = h3.cell_to_boundary(h3_id)
                return Polygon([(v[1], v[0]) for v in boundary])
            except:
                return None

        df_city['geometry'] = df_city['h3_id'].apply(get_geometry)
        df_city = df_city.dropna(subset=['geometry'])
        gdf_city = gpd.GeoDataFrame(df_city, geometry='geometry', crs="EPSG:4326")

        m = leafmap.Map(draw_control=False, measure_control=False, google_map="HYBRID")

        colormap = branca_cm.StepColormap(
            colors=["#236915", "#54ad42", "#e9dd99", "#e6b274", "#e96767"],
            vmin=0.0, vmax=1.0,
            index=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        _legend = MacroElement()
        _legend._template = Template("""
            {% macro html(this, kwargs) %}
            <div style="
                position: fixed; bottom: 40px; right: 10px; z-index: 1000;
                background: rgba(60,60,60,0.70); border-radius: 6px;
                padding: 10px 14px; font-family: Arial, sans-serif;
                font-size: 13px; color: #fff; line-height: 1.7;
            ">
              <b style="display:block; margin-bottom:6px;">Índice de Injustiça Climática</b>
              <div style="display:flex;align-items:center;">
                <span style="background:#e96767;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,8 – 1,0
              </div>
              <div style="display:flex;align-items:center;">
                <span style="background:#e6b274;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,6 – 0,8
              </div>
              <div style="display:flex;align-items:center;">
                <span style="background:#e9dd99;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,4 – 0,6
              </div>
              <div style="display:flex;align-items:center;">
                <span style="background:#54ad42;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,2 – 0,4
              </div>
              <div style="display:flex;align-items:center;">
                <span style="background:#236915;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,0 – 0,2
              </div>
            </div>
            {% endmacro %}
        """)
        _legend.add_to(m)

        folium.GeoJson(
            gdf_city,
            style_function=lambda f: {
                "fillColor": colormap(f["properties"]["iic_final"])
                    if f["properties"].get("iic_final") is not None else "#cccccc",
                "fillOpacity": 0.8,
                "stroke": False,
                "weight": 0,
            },
            name="Injustiça Climática",
        ).add_to(m)

        m.zoom_to_gdf(gdf_city)
        st_folium(m, height=650, use_container_width=True, returned_objects=[])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IIC médio", f"{city_mean:.3f}")
        c2.metric("Mais injusto", f"{df_city['iic_final'].max():.3f}")
        c3.metric("Menos injusto", f"{df_city['iic_final'].min():.3f}")
        c4.metric(
            "Posição nacional", f"Percentil {national_pct:.0f}",
            help=f"{mun_sel} tem IIC médio maior que {national_pct:.1f}% dos hexágonos do Brasil."
        )

    # ==========================================================================
    # TAB 2 — ANÁLISE
    # ==========================================================================
    with tab_analise:

        avail_dims = {k: v for k, v in DIM_COLS.items() if k in df_city.columns and k in df_brasil.columns}
        brazil_mean = df_brasil['iic_final'].mean()

        row1_esq, row1_dir = st.columns(2)

        # ----------------------------------------------------------------------
        # 1. Histograma do IIC
        # ----------------------------------------------------------------------
        with row1_esq:
            st.subheader("Distribuição do IIC")
            st.caption("Frequência dos hexágonos por faixa de injustiça climática.")

            fig_hist = px.histogram(
                df_city,
                x='iic_final',
                nbins=40,
                range_x=[0, 1],
                labels={'iic_final': 'IIC', 'count': 'Hexágonos'},
                template=plot_tpl,
                color_discrete_sequence=['#fde68a'],
            )
            fig_hist.update_traces(marker_line_width=0)
            fig_hist.add_vline(
                x=city_mean,
                line_dash='solid', line_color='#c47f00', line_width=2,
                annotation_text=f'Média {mun_sel} ({city_mean:.3f})',
                annotation_position='top left',
            )
            fig_hist.add_vline(
                x=brazil_mean,
                line_dash='dash', line_color='gray', line_width=1.5,
                annotation_text=f'Média Brasil ({brazil_mean:.3f})',
                annotation_position='top right',
            )
            fig_hist.update_layout(
                xaxis_title='IIC',
                yaxis_title='Número de hexágonos',
                showlegend=False,
                margin=dict(t=30, b=10),
            )
            st.plotly_chart(fig_hist, width='stretch')

        # ----------------------------------------------------------------------
        # 2. Radar de dimensões
        # ----------------------------------------------------------------------
        with row1_dir:
            st.subheader("Perfil por dimensão")
            st.caption("Média do município vs. média nacional em cada dimensão do índice.")

            if avail_dims:
                labels      = list(avail_dims.values())
                city_vals   = [df_city[k].mean()   for k in avail_dims]
                brazil_vals = [df_brasil[k].mean()  for k in avail_dims]

                labels_closed      = labels      + [labels[0]]
                city_vals_closed   = city_vals   + [city_vals[0]]
                brazil_vals_closed = brazil_vals + [brazil_vals[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=city_vals_closed,
                    theta=labels_closed,
                    fill='toself',
                    name=mun_sel,
                    line_color='#fcba03',
                    fillcolor='rgba(252, 186, 3, 0.25)',
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=brazil_vals_closed,
                    theta=labels_closed,
                    fill='toself',
                    name='Brasil',
                    line_color='#aaaaaa',
                    fillcolor='rgba(170, 170, 170, 0.15)',
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template=plot_tpl,
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=-0.15),
                    margin=dict(t=30, b=60),
                )
                st.plotly_chart(fig_radar, width='stretch')
            else:
                st.warning("Colunas de dimensão não encontradas no parquet. Regenere o parquet com `python main.py`.")


else:
    st.info("Selecione uma cidade e clique em 'Gerar mapa' para visualizar.")
