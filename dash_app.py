import os
import numpy as np
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
        "  → Local:  execute `python run.py`"
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
COLORMAP_INDEX  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Cores WRI Brasil
WRI_YELLOW  = "#F0AB00"   # primária — destaque de dados
WRI_GREEN   = "#32864B"   # secundária — botões de ação
FONT_FAMILY = "Arial, Helvetica, sans-serif"

ufs = sorted(df_brasil['nm_uf'].dropna().unique())

# Estatísticas nacionais pré-calculadas na inicialização
_iic_vals            = df_brasil['iic_final'].dropna().values
_hist_counts, _hist_edges = np.histogram(_iic_vals, bins=50, range=(0.0, 1.0))
_hist_centers        = (_hist_edges[:-1] + _hist_edges[1:]) / 2
_national_mean       = float(_iic_vals.mean())
_n_hex               = len(df_brasil)
_pct_alta            = float((_iic_vals >= 0.8).mean() * 100)


def build_home_content():
    """Constrói a visão geral nacional exibida na página inicial."""
    fig = go.Figure(go.Bar(
        x=_hist_centers,
        y=_hist_counts,
        width=float(_hist_edges[1] - _hist_edges[0]),
        marker_color=WRI_GREEN,
        marker_line_width=0,
    ))
    fig.add_vline(
        x=_national_mean, line_dash='solid', line_color=WRI_YELLOW, line_width=2,
        annotation=dict(
            text=f'Média Brasil ({_national_mean:.3f})',
            yref='paper', y=0.98,
            font=dict(color=WRI_YELLOW, family=FONT_FAMILY, size=11),
            showarrow=False,
            xanchor='left',
        ),
    )
    fig.update_layout(
        xaxis_title='Índice de Injustiça Climática',
        yaxis_title='Número de hexágonos',
        template=PLOT_TPL, showlegend=False,
        margin=dict(t=20, b=10),
        font=dict(family=FONT_FAMILY),
    )

    def stat_box(label, value):
        return html.Div([
            html.Div(label, style={'fontSize': '12px', 'color': '#666', 'marginBottom': '4px'}),
            html.Div(value, style={'fontSize': '20px', 'fontWeight': '700', 'color': '#111'}),
        ], style={
            'flex': '1', 'padding': '14px', 'backgroundColor': '#f8f9fa',
            'borderRadius': '6px', 'border': '1px solid #dee2e6',
            'textAlign': 'center', 'minWidth': '130px',
        })

    return html.Div([
        html.H3("Panorama Nacional", style={'marginBottom': '16px', 'fontWeight': '700'}),
        html.Div([
            stat_box("Hexágonos", f"{_n_hex:,}".replace(',', '.')),
            stat_box("IIC médio", f"{_national_mean:.3f}"),
            stat_box("IIC alto (> 0,8)", f"{_pct_alta:.1f}%"),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '24px', 'flexWrap': 'wrap'}),
        html.H4("Distribuição nacional do Índice de Injustiça Climática",
                style={'marginBottom': '4px', 'fontWeight': '700', 'fontSize': '16px'}),
        html.P("Frequência de hexágonos por faixa de injustiça climática em todo o Brasil.",
               style={'color': '#666', 'fontSize': '13px', 'marginTop': '0', 'marginBottom': '8px'}),
        dcc.Graph(figure=fig, style={'height': '320px'}),
    ])


# --- App ---
app = dash.Dash(__name__, title="Índice de Injustiça Climática")
server = app.server  # ponto de entrada para gunicorn

app.layout = html.Div([
    dcc.Store(id='auth-store', storage_type='session', data=False),

    # Tela de login
    html.Div(id='login-screen', style={
        'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center',
        'height': '100vh', 'fontFamily': FONT_FAMILY, 'backgroundColor': '#f8f9fa',
    }, children=[
        html.Div([
            html.H2("Índice de Injustiça Climática",
                    style={'textAlign': 'center', 'marginBottom': '24px', 'fontWeight': '700'}),
            dcc.Input(
                id='senha-input', type='password', placeholder='Digite a senha...',
                debounce=True,
                style={
                    'width': '100%', 'padding': '10px', 'fontSize': '14px',
                    'borderRadius': '4px', 'border': '1px solid #ccc', 'boxSizing': 'border-box',
                },
            ),
            html.Div(id='senha-erro', style={'color': '#b91c1c', 'marginTop': '8px', 'fontSize': '13px'}),
        ], style={
            'width': '420px', 'padding': '40px',
            'boxShadow': '0 2px 16px rgba(0,0,0,0.12)', 'borderRadius': '8px',
            'backgroundColor': '#fff',
        }),
    ]),

    # App principal
    html.Div(id='main-content', style={'display': 'none'}, children=[

        # Barra lateral
        html.Div([
            html.H3("Selecione o município",
                    style={'marginTop': '0', 'marginBottom': '14px', 'fontWeight': '700', 'fontSize': '15px'}),
            html.Label("Estado", style={'fontWeight': '700', 'marginBottom': '4px', 'display': 'block', 'fontSize': '13px'}),
            dcc.Dropdown(
                id='uf-dropdown',
                options=[{'label': u, 'value': u} for u in ufs],
                value=ufs[0], clearable=False,
                style={'marginBottom': '14px', 'fontSize': '13px'},
            ),
            html.Label("Município", style={'fontWeight': '700', 'marginBottom': '4px', 'display': 'block', 'fontSize': '13px'}),
            dcc.Dropdown(
                id='mun-dropdown', options=[], value=None, clearable=False,
                style={'marginBottom': '20px', 'fontSize': '13px'},
            ),
            html.Button("Gerar mapa", id='gerar-btn', n_clicks=0, style={
                'width': '100%', 'padding': '10px', 'backgroundColor': WRI_GREEN,
                'color': 'white', 'border': 'none', 'borderRadius': '4px',
                'fontSize': '14px', 'cursor': 'pointer', 'fontWeight': '700',
            }),
            html.Button("← Visão geral", id='home-btn', n_clicks=0, style={
                'width': '100%', 'padding': '8px', 'backgroundColor': 'transparent',
                'color': WRI_GREEN, 'border': f'1px solid {WRI_GREEN}', 'borderRadius': '4px',
                'fontSize': '13px', 'cursor': 'pointer', 'marginTop': '8px',
            }),
        ], style={
            'position': 'fixed', 'top': '0', 'left': '0', 'width': '260px', 'height': '100vh',
            'backgroundColor': '#f8f9fa', 'padding': '24px 20px', 'boxSizing': 'border-box',
            'borderRight': '1px solid #dee2e6', 'overflowY': 'auto', 'fontFamily': FONT_FAMILY,
        }),

        # Área principal
        html.Div([
            html.H1(
                "Índice de Injustiça Climática para municípios brasileiros",
                style={'fontWeight': '700', 'fontFamily': FONT_FAMILY, 'marginBottom': '8px'},
            ),
            html.P(
                "O Índice de Injustiça Climática (IIC) mensura desigualdades territoriais frente aos riscos "
                "climáticos, combinando exposição a eventos extremos, vulnerabilidade socioeconômica, presença "
                "de grupos populacionais prioritários e capacidade de gestão municipal. Desenvolvido pelo WRI "
                "Brasil, o índice é calculado em grade hexagonal H3 de alta resolução (~0,1 km²) para todos os "
                "municípios brasileiros.",
                style={'fontFamily': FONT_FAMILY, 'color': '#444', 'marginBottom': '24px', 'maxWidth': '760px'},
            ),
            dcc.Loading(
                id='loading', type='circle', color=WRI_YELLOW,
                children=html.Div(id='dashboard-content', children=build_home_content()),
            ),
        ], style={
            'marginLeft': '280px', 'padding': '32px 40px',
            'fontFamily': FONT_FAMILY, 'minHeight': '100vh', 'boxSizing': 'border-box',
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
        'height': '100vh', 'fontFamily': FONT_FAMILY, 'backgroundColor': '#f8f9fa',
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
    Input('home-btn', 'n_clicks'),
    State('uf-dropdown', 'value'),
    State('mun-dropdown', 'value'),
    prevent_initial_call=True,
)
def update_dashboard(gerar_clicks, home_clicks, uf, mun):
    triggered = dash.callback_context.triggered[0]['prop_id']

    if 'home-btn' in triggered:
        return build_home_content()

    if not uf or not mun:
        return no_update

    df_city = df_brasil[
        (df_brasil['nm_uf'] == uf) & (df_brasil['nm_mun'] == mun)
    ].copy()

    if df_city.empty:
        return html.Div("Nenhum dado encontrado para essa seleção.",
                        style={'color': '#b91c1c', 'fontFamily': FONT_FAMILY})

    city_mean    = df_city['iic_final'].mean()
    national_pct = (df_brasil['iic_final'] < city_mean).mean() * 100
    brazil_mean  = _national_mean

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
    # Posiciona as anotações em alturas distintas para evitar sobreposição
    city_anchor   = 'right' if city_mean > 0.6 else 'left'
    brazil_anchor = 'right' if brazil_mean > 0.6 else 'left'

    fig_hist = px.histogram(
        df_city, x='iic_final', nbins=40, range_x=[0, 1],
        labels={'iic_final': 'Índice de Injustiça Climática', 'count': 'Hexágonos'},
        template=PLOT_TPL, color_discrete_sequence=['#fde68a'],
    )
    fig_hist.update_traces(marker_line_width=0)
    fig_hist.add_vline(
        x=city_mean, line_dash='solid', line_color=WRI_YELLOW, line_width=2,
        annotation=dict(
            text=f'Média {mun} ({city_mean:.3f})',
            yref='paper', y=0.98,
            font=dict(color=WRI_YELLOW, family=FONT_FAMILY, size=11),
            showarrow=False, xanchor=city_anchor,
        ),
    )
    fig_hist.add_vline(
        x=brazil_mean, line_dash='dash', line_color='#888', line_width=1.5,
        annotation=dict(
            text=f'Média Brasil ({brazil_mean:.3f})',
            yref='paper', y=0.82,
            font=dict(color='#888', family=FONT_FAMILY, size=11),
            showarrow=False, xanchor=brazil_anchor,
        ),
    )
    fig_hist.update_layout(
        xaxis_title='Índice de Injustiça Climática',
        yaxis_title='Número de hexágonos',
        showlegend=False, margin=dict(t=30, b=10),
        font=dict(family=FONT_FAMILY),
    )

    # --- Radar ---
    avail_dims  = {k: v for k, v in DIM_COLS.items()
                   if k in df_city.columns and k in df_brasil.columns}
    labels      = list(avail_dims.values())
    city_vals   = [df_city[k].mean()  for k in avail_dims]
    brazil_vals = [df_brasil[k].mean() for k in avail_dims]

    labels_c  = labels      + [labels[0]]
    city_c    = city_vals   + [city_vals[0]]
    brazil_c  = brazil_vals + [brazil_vals[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=city_c, theta=labels_c, fill='toself', name=mun,
        line_color=WRI_YELLOW, fillcolor='rgba(240, 171, 0, 0.25)',
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
        font=dict(family=FONT_FAMILY),
    )

    def caption(text):
        return html.P(text, style={
            'color': '#666', 'fontSize': '12px', 'marginTop': '4px',
            'fontStyle': 'italic', 'fontFamily': FONT_FAMILY,
        })

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
                    html.H3("Distribuição do Índice de Injustiça Climática",
                            style={'marginBottom': '4px'}),
                    html.P("Frequência dos hexágonos por faixa de injustiça climática.",
                           style={'color': '#666', 'fontSize': '13px', 'marginTop': '0'}),
                    dcc.Graph(figure=fig_hist),
                    caption(
                        "Hexágonos à direita do gráfico concentram as piores condições: "
                        "maior exposição a riscos climáticos, maior vulnerabilidade socioeconômica "
                        "e menor capacidade de resposta municipal."
                    ),
                ], style={'flex': '1', 'minWidth': '300px'}),
                html.Div([
                    html.H3("Perfil por dimensão", style={'marginBottom': '4px'}),
                    html.P("Média do município vs. média nacional em cada dimensão do índice.",
                           style={'color': '#666', 'fontSize': '13px', 'marginTop': '0'}),
                    dcc.Graph(figure=fig_radar),
                    caption(
                        "Quanto mais próximo de 1,0 em cada eixo, maior a injustiça climática "
                        "naquela dimensão. A dimensão Capacidade de Gestão já está invertida: "
                        "menor capacidade municipal equivale a maior injustiça."
                    ),
                ], style={'flex': '1', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'gap': '32px', 'flexWrap': 'wrap', 'marginTop': '16px'}),
        ]),
    ])


if __name__ == '__main__':
    app.run(debug=True)
