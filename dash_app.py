import os
import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import folium
import leafmap.foliumap as leafmap
import branca.colormap as branca_cm
from branca.element import MacroElement
from jinja2 import Template
from shapely.geometry import Polygon
import h3
from src import config as cfg

# --- Carregamento de dados (uma vez na inicialização) ---
def find_latest_dashboard():
    # 1. Release: pasta commitada no git → usada no Render e em produção
    release_dir = cfg.BASE_DIR / "data" / "outputs" / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    release_files = sorted(release_dir.glob(f"{cfg.DASHBOARD_FILE_PREFIX}_*.parquet"))
    if release_files:
        return str(release_files[-1])

    # 2. Fallback local: pasta gitignored → usada durante desenvolvimento
    results_dir = cfg.FILES['output']['repo_results_dir']
    results_files = sorted(results_dir.glob(f"{cfg.DASHBOARD_FILE_PREFIX}_*.parquet"))
    if results_files:
        return str(results_files[-1])

    raise FileNotFoundError(
        "Nenhum parquet de dashboard encontrado.\n"
        "  → Deploy: copie o parquet aprovado para data/outputs/release/\n"
        "  → Local:  execute `python main.py`"
    )

df_brasil = pd.read_parquet(find_latest_dashboard())

DIM_COLS = {
    'ip': 'Grupos Prioritários',
    'iv': 'Vulnerabilidade',
    'ie': 'Exposição',
    'ig': 'Capacidade de Gestão Municipal',
}
PLOT_TPL = "plotly_white"
COLORMAP_COLORS = ["#236915", "#54ad42", "#e9dd99", "#e6b274", "#e96767"]
COLORMAP_INDEX = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

ufs = sorted(df_brasil['nm_uf'].dropna().unique())

# --- App ---
app = dash.Dash(__name__, title="Índice de Injustiça Climática")
server = app.server  # ponto de entrada para gunicorn

app.layout = html.Div([
    dcc.Store(id='auth-store', storage_type='session', data=False),

    # Tela de login
    html.Div(id='login-screen', style={
        'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center',
        'height': '100vh', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa',
    }, children=[
        html.Div([
            html.H2("Índice de Injustiça Climática",
                    style={'textAlign': 'center', 'marginBottom': '24px', 'fontWeight': '700'}),
            dcc.Input(
                id='senha-input', type='password', placeholder='Digite a senha...',
                debounce=True,
                style={
                    'width': '100%', 'padding': '10px', 'fontSize': '16px',
                    'borderRadius': '4px', 'border': '1px solid #ccc', 'boxSizing': 'border-box',
                },
            ),
            html.Div(id='senha-erro', style={'color': '#b91c1c', 'marginTop': '8px', 'fontSize': '14px'}),
        ], style={
            'width': '320px', 'padding': '40px',
            'boxShadow': '0 2px 16px rgba(0,0,0,0.12)', 'borderRadius': '8px',
            'backgroundColor': '#fff',
        }),
    ]),

    # App principal
    html.Div(id='main-content', style={'display': 'none'}, children=[

        # Barra lateral
        html.Div([
            html.H3("Selecione o município",
                    style={'marginTop': '0', 'marginBottom': '16px', 'fontWeight': '700'}),
            html.Label("Estado", style={'fontWeight': '700', 'marginBottom': '4px', 'display': 'block'}),
            dcc.Dropdown(
                id='uf-dropdown',
                options=[{'label': u, 'value': u} for u in ufs],
                value=ufs[0], clearable=False,
                style={'marginBottom': '16px'},
            ),
            html.Label("Município", style={'fontWeight': '700', 'marginBottom': '4px', 'display': 'block'}),
            dcc.Dropdown(id='mun-dropdown', options=[], value=None, clearable=False,
                         style={'marginBottom': '24px'}),
            html.Button("Gerar mapa", id='gerar-btn', n_clicks=0, style={
                'width': '100%', 'padding': '10px', 'backgroundColor': '#1a6b3c',
                'color': 'white', 'border': 'none', 'borderRadius': '4px',
                'fontSize': '15px', 'cursor': 'pointer', 'fontWeight': '700',
            }),
        ], style={
            'position': 'fixed', 'top': '0', 'left': '0', 'width': '260px', 'height': '100vh',
            'backgroundColor': '#f8f9fa', 'padding': '24px 20px', 'boxSizing': 'border-box',
            'borderRight': '1px solid #dee2e6', 'overflowY': 'auto', 'fontFamily': 'Arial, sans-serif',
        }),

        # Área principal
        html.Div([
            html.H1(
                "Índice de Injustiça Climática para municípios brasileiros",
                style={'fontWeight': '700', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'},
            ),
            html.P(
                "O Índice de Injustiça Climática (IIC) mensura desigualdades territoriais frente aos riscos "
                "climáticos, combinando exposição a eventos extremos, vulnerabilidade socioeconômica, presença "
                "de grupos populacionais prioritários e capacidade de gestão municipal. Desenvolvido pelo WRI "
                "Brasil, o índice é calculado em grade hexagonal H3 de alta resolução (~0,1 km²) para todos os "
                "municípios brasileiros.",
                style={'fontFamily': 'Arial, sans-serif', 'color': '#444', 'marginBottom': '24px', 'maxWidth': '760px'},
            ),
            dcc.Loading(
                id='loading', type='circle', color='#1a6b3c',
                children=html.Div(id='dashboard-content', children=[
                    html.Div(
                        "Selecione um município e clique em 'Gerar mapa' para visualizar.",
                        style={
                            'backgroundColor': '#FEF3C7', 'borderLeft': '4px solid #F0AB00',
                            'padding': '12px 16px', 'borderRadius': '4px',
                            'color': '#92680B', 'fontFamily': 'Arial, sans-serif',
                        },
                    ),
                ]),
            ),
        ], style={
            'marginLeft': '280px', 'padding': '32px 40px',
            'fontFamily': 'Arial, sans-serif', 'minHeight': '100vh', 'boxSizing': 'border-box',
        }),
    ]),
], style={'margin': '0', 'padding': '0'})


# --- Callbacks ---

@app.callback(
    Output('auth-store', 'data'),
    Output('senha-erro', 'children'),
    Input('senha-input', 'value'),
    prevent_initial_call=True,
)
def check_password(senha):
    expected = os.environ.get('DASH_PASSWORD', '')
    if not expected:
        return True, ''
    if senha == expected:
        return True, ''
    return False, 'Senha incorreta.'


@app.callback(
    Output('login-screen', 'style'),
    Output('main-content', 'style'),
    Input('auth-store', 'data'),
)
def toggle_auth(authenticated):
    hidden = {'display': 'none'}
    login_visible = {
        'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center',
        'height': '100vh', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa',
    }
    if authenticated:
        return hidden, {'display': 'block'}
    return login_visible, hidden


@app.callback(
    Output('mun-dropdown', 'options'),
    Output('mun-dropdown', 'value'),
    Input('uf-dropdown', 'value'),
)
def update_municipios(uf):
    if not uf:
        return [], None
    muns = sorted(df_brasil[df_brasil['nm_uf'] == uf]['nm_mun'].dropna().unique())
    opts = [{'label': m, 'value': m} for m in muns]
    return opts, muns[0] if muns else None


@app.callback(
    Output('dashboard-content', 'children'),
    Input('gerar-btn', 'n_clicks'),
    State('uf-dropdown', 'value'),
    State('mun-dropdown', 'value'),
    prevent_initial_call=True,
)
def update_dashboard(n_clicks, uf, mun):
    if not uf or not mun:
        return no_update

    df_city = df_brasil[
        (df_brasil['nm_uf'] == uf) & (df_brasil['nm_mun'] == mun)
    ].copy()

    if df_city.empty:
        return html.Div("Nenhum dado encontrado para essa seleção.",
                        style={'color': '#b91c1c', 'fontFamily': 'Arial, sans-serif'})

    city_mean = df_city['iic_final'].mean()
    national_pct = (df_brasil['iic_final'] < city_mean).mean() * 100
    brazil_mean = df_brasil['iic_final'].mean()

    # --- Mapa ---
    def get_geometry(h3_id):
        try:
            boundary = h3.cell_to_boundary(h3_id)
            return Polygon([(v[1], v[0]) for v in boundary])
        except Exception:
            return None

    df_city['geometry'] = df_city['h3_id'].apply(get_geometry)
    df_city = df_city.dropna(subset=['geometry'])
    gdf_city = gpd.GeoDataFrame(df_city, geometry='geometry', crs="EPSG:4326")

    m = leafmap.Map(draw_control=False, measure_control=False, google_map="HYBRID")

    colormap = branca_cm.StepColormap(
        colors=COLORMAP_COLORS,
        vmin=0.0, vmax=1.0,
        index=COLORMAP_INDEX,
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
          <div style="display:flex;align-items:center;"><span style="background:#e96767;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,8 – 1,0</div>
          <div style="display:flex;align-items:center;"><span style="background:#e6b274;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,6 – 0,8</div>
          <div style="display:flex;align-items:center;"><span style="background:#e9dd99;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,4 – 0,6</div>
          <div style="display:flex;align-items:center;"><span style="background:#54ad42;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,2 – 0,4</div>
          <div style="display:flex;align-items:center;"><span style="background:#236915;width:16px;height:16px;display:inline-block;margin-right:8px;border-radius:2px;flex-shrink:0;"></span>0,0 – 0,2</div>
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
    map_html = m._repr_html_()

    # --- Métricas ---
    def metric_box(label, value):
        return html.Div([
            html.Div(label, style={'fontSize': '13px', 'color': '#666', 'marginBottom': '4px'}),
            html.Div(value, style={'fontSize': '22px', 'fontWeight': '700', 'color': '#111'}),
        ], style={
            'flex': '1', 'padding': '16px', 'backgroundColor': '#f8f9fa',
            'borderRadius': '6px', 'border': '1px solid #dee2e6', 'textAlign': 'center',
            'minWidth': '140px',
        })

    metrics = html.Div([
        metric_box("IIC médio", f"{city_mean:.3f}"),
        metric_box("Mais injusto", f"{df_city['iic_final'].max():.3f}"),
        metric_box("Menos injusto", f"{df_city['iic_final'].min():.3f}"),
        metric_box("Posição nacional", f"Percentil {national_pct:.0f}"),
    ], style={'display': 'flex', 'gap': '16px', 'marginTop': '16px', 'flexWrap': 'wrap'})

    # --- Histograma ---
    fig_hist = px.histogram(
        df_city, x='iic_final', nbins=40, range_x=[0, 1],
        labels={'iic_final': 'IIC', 'count': 'Hexágonos'},
        template=PLOT_TPL, color_discrete_sequence=['#fde68a'],
    )
    fig_hist.update_traces(marker_line_width=0)
    fig_hist.add_vline(
        x=city_mean, line_dash='solid', line_color='#c47f00', line_width=2,
        annotation_text=f'Média {mun} ({city_mean:.3f})', annotation_position='top left',
    )
    fig_hist.add_vline(
        x=brazil_mean, line_dash='dash', line_color='gray', line_width=1.5,
        annotation_text=f'Média Brasil ({brazil_mean:.3f})', annotation_position='top right',
    )
    fig_hist.update_layout(
        xaxis_title='IIC', yaxis_title='Número de hexágonos',
        showlegend=False, margin=dict(t=30, b=10),
    )

    # --- Radar ---
    avail_dims = {k: v for k, v in DIM_COLS.items()
                  if k in df_city.columns and k in df_brasil.columns}
    labels = list(avail_dims.values())
    city_vals = [df_city[k].mean() for k in avail_dims]
    brazil_vals = [df_brasil[k].mean() for k in avail_dims]

    labels_c = labels + [labels[0]]
    city_c = city_vals + [city_vals[0]]
    brazil_c = brazil_vals + [brazil_vals[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=city_c, theta=labels_c, fill='toself', name=mun,
        line_color='#fcba03', fillcolor='rgba(252, 186, 3, 0.25)',
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=brazil_c, theta=labels_c, fill='toself', name='Brasil',
        line_color='#aaaaaa', fillcolor='rgba(170, 170, 170, 0.15)',
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template=PLOT_TPL, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15),
        margin=dict(t=30, b=60),
    )

    return dcc.Tabs([
        dcc.Tab(label='Mapa', children=[
            html.Iframe(
                srcDoc=map_html,
                style={'width': '100%', 'height': '650px', 'border': 'none', 'display': 'block'},
            ),
            metrics,
        ]),
        dcc.Tab(label='Análise', children=[
            html.Div([
                html.Div([
                    html.H3("Distribuição do IIC", style={'marginBottom': '4px'}),
                    html.P("Frequência dos hexágonos por faixa de injustiça climática.",
                           style={'color': '#666', 'fontSize': '13px', 'marginTop': '0'}),
                    dcc.Graph(figure=fig_hist),
                ], style={'flex': '1', 'minWidth': '300px'}),
                html.Div([
                    html.H3("Perfil por dimensão", style={'marginBottom': '4px'}),
                    html.P("Média do município vs. média nacional em cada dimensão do índice.",
                           style={'color': '#666', 'fontSize': '13px', 'marginTop': '0'}),
                    dcc.Graph(figure=fig_radar),
                ], style={'flex': '1', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'gap': '32px', 'flexWrap': 'wrap', 'marginTop': '16px'}),
        ]),
    ])


if __name__ == '__main__':
    app.run(debug=True)
