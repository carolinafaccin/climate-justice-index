import streamlit as st
import pandas as pd
import geopandas as gpd
import leafmap.foliumap as leafmap
from shapely.geometry import Polygon
import h3
from src import config as cfg

# --- LÓGICA DE SENHA ---
def check_password():
    """Retorna True se o usuário tiver a senha correta."""

    def password_entered():
        """Verifica se a senha inserida está correta."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # não armazena a senha
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Primeira execução, mostra entrada de senha
        st.text_input(
            "Digite a senha para acessar o Atlas:", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Senha incorreta
        st.text_input(
            "Digite a senha para acessar o Atlas:", type="password", on_change=password_entered, key="password"
        )
        st.error("Senha incorreta.")
        return False
    else:
        # Senha correta
        return True

if not check_password():
    st.stop()  # Para a execução se a senha não for inserida/correta

# Configuração da Página
st.set_page_config(layout="wide", page_title="Atlas de Injustiça Climática")

st.title("Índice de Injustiça Climática para municípios brasileiros")
st.markdown("Análise intramunicipal através de hexágonos H3 (resolução 9).")

# ==============================================================================
# 1. CARREGAMENTO DE DADOS
# ==============================================================================
@st.cache_data
def load_data():
    path = cfg.FILES['output']['h3_final']
    cols = ['h3_id', 'nm_mun', 'nm_uf', 'iic_final']
    df = pd.read_parquet(path, columns=cols)
    return df

with st.spinner("Carregando base de dados..."):
    df_brasil = load_data()

# ==============================================================================
# 2. BARRA LATERAL (FILTROS)
# ==============================================================================
st.sidebar.header("📍 Selecione o Local")

# Filtra UFs removendo nulos e ordenando
ufs = sorted(df_brasil['nm_uf'].dropna().unique())
uf_sel = st.sidebar.selectbox("Estado", ufs)

# CORREÇÃO DO ERRO DE TYPEERROR:
# Adicionado .dropna() antes do unique() para garantir que não haja NoneType na lista
muns = sorted(df_brasil[df_brasil['nm_uf'] == uf_sel]['nm_mun'].dropna().unique())
mun_sel = st.sidebar.selectbox("Município", muns)

if st.sidebar.button("Gerar Mapa", type="primary"):
    
    # ==========================================================================
    # 3. PROCESSAMENTO GEESPACIAL
    # ==========================================================================
    df_city = df_brasil[(df_brasil['nm_uf'] == uf_sel) & (df_brasil['nm_mun'] == mun_sel)].copy()
    
    if df_city.empty:
        st.error("Nenhum dado encontrado para essa seleção.")
        st.stop()

    st.info(f"Carregando {len(df_city)} hexágonos para {mun_sel} - {uf_sel}...")

    def get_geometry(h3_id):
        try:
            boundary = h3.cell_to_boundary(h3_id)
            # Inverte lat/lon para lon/lat que o GeoPandas/Folium esperam (x, y)
            boundary_xy = [(v[1], v[0]) for v in boundary]
            return Polygon(boundary_xy)
        except:
            return None

    df_city['geometry'] = df_city['h3_id'].apply(get_geometry)
    
    # Remove geometrias falhas se houver
    df_city = df_city.dropna(subset=['geometry'])

    gdf_city = gpd.GeoDataFrame(df_city, geometry='geometry', crs="EPSG:4326")

    # ==========================================================================
    # 4. RENDERIZAÇÃO DO MAPA
    # ==========================================================================
    
    m = leafmap.Map(draw_control=False, measure_control=False, google_map="HYBRID")
    
    paleta = ["#e96767", "#e6b274", "#e9dd99", "#54ad42", "#236915"]
    
    m.add_data(
        data=gdf_city,
        column="iic_final",
        scheme="UserDefined",
        classification_kwds={'bins': [0.2754, 0.4725, 0.6453, 0.8040]},
        colors=paleta,
        legend_title="Índice de Injustiça Climática",
        layer_name="Injustiça Climática",
        # --- AQUI ESTÃO AS MUDANÇAS DE ESTILO ---
        style_kwds={
            "stroke": False,      # Remove o contorno azul (o jeito mais limpo)
            "fillOpacity": 0.8    # 80% de opacidade (20% transparente)
        }
    )
    
    m.zoom_to_gdf(gdf_city)
    m.to_streamlit(height=700)

    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("IIC médio", f"{df_city['iic_final'].mean():.3f}")
    c2.metric("Mais injusto", f"{df_city['iic_final'].max():.3f}")
    c3.metric("Menos injusto", f"{df_city['iic_final'].min():.3f}")

else:
    st.info("👈 Selecione uma cidade e clique em 'Gerar Mapa' para visualizar.")