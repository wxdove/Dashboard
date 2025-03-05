import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
import calendar
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(42)
months = list(calendar.month_abbr)[1:]
coffee_types = ['Espresso', 'Latte', 'Cappuccino']

base_pattern = np.sin(np.linspace(0, 2 * np.pi, 12)) * 0.3 + 0.7
espresso_sales = (np.random.normal(1000, 100, 12) * base_pattern).astype(int)
latte_sales = (np.random.normal(1500, 150, 12) * base_pattern).astype(int)
cappuccino_sales = (np.random.normal(1200, 120, 12) * (base_pattern[::-1])).astype(int)

espresso_prices = np.round(np.random.normal(3.5, 0.2, 12) * (base_pattern * 0.1 + 0.95), 2)
latte_prices = np.round(np.random.normal(4.5, 0.3, 12) * (base_pattern * 0.1 + 0.95), 2)
cappuccino_prices = np.round(np.random.normal(4.2, 0.25, 12) * (base_pattern * 0.1 + 0.95), 2)

sales_data = pd.DataFrame({
    'Month': months,
    'Espresso': espresso_sales,
    'Latte': latte_sales,
    'Cappuccino': cappuccino_sales,
    'EspressoPrice': espresso_prices,
    'LattePrice': latte_prices,
    'CappuccinoPrice': cappuccino_prices,
    'MonthNum': range(1, 13)  # For sorting
})

sales_data['Total'] = sales_data['Espresso'] + sales_data['Latte'] + sales_data['Cappuccino']
sales_data['Revenue'] = (sales_data['Espresso'] * sales_data['EspressoPrice'] +
                         sales_data['Latte'] * sales_data['LattePrice'] +
                         sales_data['Cappuccino'] * sales_data['CappuccinoPrice'])

total_by_coffee = {
    'Type': coffee_types,
    'Sales': [sales_data['Espresso'].sum(), sales_data['Latte'].sum(), sales_data['Cappuccino'].sum()]
}

top_month_idx = sales_data['Total'].argmax()
top_month = sales_data.iloc[top_month_idx]['Month']
top_product = coffee_types[np.argmax([sales_data['Espresso'].sum(),
                                      sales_data['Latte'].sum(),
                                      sales_data['Cappuccino'].sum()])]

sales_long = pd.melt(
    sales_data,
    id_vars=['Month', 'MonthNum'],
    value_vars=coffee_types,
    var_name='Coffee Type',
    value_name='Sales'
)

price_long = pd.melt(
    sales_data,
    id_vars=['Month', 'MonthNum'],
    value_vars=['EspressoPrice', 'LattePrice', 'CappuccinoPrice'],
    var_name='Coffee Type',
    value_name='Price'
)
price_long['Coffee Type'] = price_long['Coffee Type'].str.replace('Price', '')

future_months = ['Jan', 'Feb', 'Mar', 'Apr']
prediction_months = months + future_months


def predict_future_values(data, periods=4):
    model = ARIMA(data, order=(1, 0, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return list(forecast)


espresso_predictions = list(espresso_sales) + predict_future_values(espresso_sales)
latte_predictions = list(latte_sales) + predict_future_values(latte_sales)
cappuccino_predictions = list(cappuccino_sales) + predict_future_values(cappuccino_sales)

colors = {
    'background': '#FAF7F0',  # Light cream
    'card_bg': '#FFFFFF',     # White
    'text': '#4A3933',        # Dark brown
    'espresso': '#654321',    # Dark coffee brown
    'latte': '#C4A484',       # Light brown
    'cappuccino': '#D2B48C',  # Tan
    'accent1': '#8B4513',     # SaddleBrown
    'accent2': '#A52A2A',     # Brown
    'header': '#3A2618',      # Very dark brown
    'sidebar': '#EBE3D5',     # Light beige
    'success': '#2E8B57',     # SeaGreen
    'warning': '#DAA520',     # GoldenRod
    'info': '#4682B4'         # SteelBlue
}

coffee_colors = {
    'Espresso': colors['espresso'],
    'Latte': colors['latte'],
    'Cappuccino': colors['cappuccino']
}

card_style = {
    'backgroundColor': colors['card_bg'],
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
    'marginBottom': '15px'
}

# ---------------------- APP CREATION ----------------------
# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Raleway:wght@300;400;500&display=swap"
    ]
)
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True

# ---------------------- LAYOUT COMPONENTS ----------------------
# Current time and user - UPDATED
current_time = "from 2024-01-01 to 2025-01-01"
current_user1 = "Jessica Julian"
current_user2 = "Twinkie"

avatar_section = html.Div([
    # 两个头像并排排列
    html.Div([
        # 第一个用户头像和信息
        html.Div([
            html.Img(
                src="https://i.pinimg.com/236x/45/ba/c2/45bac2a0b90f98d1db4a64ac7b4e5b2d.jpg",
                style={
                    'height': '50px',
                    'width': '50px',
                    'borderRadius': '50%',
                    'objectFit': 'cover',
                    'border': f'2px solid {colors["accent1"]}'
                }
            ),
            html.H5(current_user1, style={
                'textAlign': 'center',
                'color': colors['text'],
                'margin': '5px 0',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }),
            html.P("Coffee Analyst", style={
                'textAlign': 'center',
                'color': colors['text'],
                'opacity': '0.7',
                'fontSize': '10px',
                'margin': '0'
            })
        ], style={'width': '50%', 'display': 'inline-block', 'textAlign': 'center'}),

        # 第二个用户头像和信息
        html.Div([
            html.Img(
                src="https://i.pinimg.com/236x/1b/89/96/1b899655b58140af2e41c83a4a1394f4.jpg",  # 不同头像
                style={
                    'height': '50px',
                    'width': '50px',
                    'borderRadius': '50%',
                    'objectFit': 'cover',
                    'border': f'2px solid {colors["cappuccino"]}'  # 不同边框颜色
                }
            ),
            html.H5(current_user2, style={
                'textAlign': 'center',
                'color': colors['text'],
                'margin': '5px 0',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }),
            html.P("Data Scientist", style={  # 不同职位
                'textAlign': 'center',
                'color': colors['text'],
                'opacity': '0.7',
                'fontSize': '10px',
                'margin': '0'
            })
        ], style={'width': '50%', 'display': 'inline-block', 'textAlign': 'center'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '15px'}),

    # 共享信息栏
    html.Div([
        html.P("Coffee shop owner", style={
            'textAlign': 'center',
            'color': colors['accent2'],
            'fontSize': '11px',
            'margin': '5px 0',
            'fontWeight': 'bold'
        })
    ])
], style={'marginBottom': '20px', 'padding': '5px'})

sidebar = html.Div([
    avatar_section,
    html.Hr(style={'margin': '0 0 15px 0'}),

    html.H6("NAVIGATION", style={'fontSize': '12px', 'color': '#777', 'fontWeight': 'bold', 'marginLeft': '5px'}),
    dbc.Nav([
        dbc.NavLink([
            html.I(className="fas fa-home me-2"),
            "Dashboard"
        ], href="#", active=True, id="nav-dashboard", className="sidebar-link"),
        dbc.NavLink([
            html.I(className="fas fa-chart-line me-2"),
            "Trends"
        ], href="#", active=False, id="nav-trends", className="sidebar-link"),
        dbc.NavLink([
            html.I(className="fas fa-chart-area me-2"),
            "Predictions"
        ], href="#", active=False, id="nav-predictions", className="sidebar-link"),
    ], vertical=True, pills=True, className="mb-3"),

    html.Hr(style={'margin': '15px 0'}),

    # Coffee Filter Section
    # 修改咖啡过滤部分
    html.H6("FILTER BY COFFEE", style={'fontSize': '12px', 'color': '#777', 'fontWeight': 'bold', 'marginLeft': '5px'}),
    dbc.Nav([
        dbc.NavLink([
            html.Div(style={
                'width': '10px',
                'height': '10px',
                'borderRadius': '50%',
                'backgroundColor': colors['espresso'],
                'display': 'inline-block',
                'marginRight': '10px'
            }),
            "Espresso"
        ], href="#", active=False, id="filter-espresso", className="sidebar-link"),
        dbc.NavLink([
            html.Div(style={
                'width': '10px',
                'height': '10px',
                'borderRadius': '50%',
                'backgroundColor': colors['latte'],
                'display': 'inline-block',
                'marginRight': '10px'
            }),
            "Latte"
        ], href="#", active=False, id="filter-latte", className="sidebar-link"),
        dbc.NavLink([
            html.Div(style={
                'width': '10px',
                'height': '10px',
                'borderRadius': '50%',
                'backgroundColor': colors['cappuccino'],
                'display': 'inline-block',
                'marginRight': '10px'
            }),
            "Cappuccino"
        ], href="#", active=False, id="filter-cappuccino", className="sidebar-link"),
        dbc.NavLink([
            html.I(className="fas fa-undo me-2"),
            "Show All"
        ], href="#", active=True, id="filter-all", className="sidebar-link"),
    ], vertical=True, pills=True),

    html.Hr(style={'margin': '15px 0'}),

    # Time indicator
    html.Div([
        html.Small([
            html.I(className="fas fa-clock me-1", style={'color': colors['accent1']}),
            current_time
        ], style={'color': '#777', 'fontSize': '10px'})
    ], style={'marginTop': 'auto', 'textAlign': 'center'})

], style={
    'width': '300px',
    'height': '100vh',
    'position': 'fixed',
    'top': '0',
    'left': '0',
    'backgroundColor': colors['sidebar'],
    'paddingTop': '15px',
    'paddingBottom': '15px',
    'paddingLeft': '15px',
    'paddingRight': '15px',
    'display': 'flex',
    'flexDirection': 'column',
    'overflowY': 'auto'
})

header = dbc.Row([
    dbc.Col([
        html.Div([
            html.H4("Coffee Bean Analytics", style={
                'color': colors['header'],
                'fontFamily': '"Playfair Display", serif',
                'fontWeight': 'bold',
                'margin': '0',
                'textAlign': 'right'  # 向右对齐
            }),
            html.Div([
                html.Img(src="", height="18px",
                         style={'display': 'inline-block', 'marginRight': '5px', 'verticalAlign': 'middle'}),
                html.Span("Twinkie & Jessica Julian.", style={'fontSize': '12px', 'color': colors['accent1']})
            ], style={'textAlign': 'right', 'paddingRight': '50px'})  # 向右对齐并添加一些右侧填充
        ])
    ], width=8),
    dbc.Col([
        html.Div([
            dbc.Badge("OVERVIEW", color="red", className="me-1", id="active-view"),
            dbc.Badge(
                html.I(className="fas fa-bell"),
                color="warning",
                className="me-1",
                style={'cursor': 'pointer'}
            ),
            dbc.Badge(
                html.I(className="fas fa-cog"),
                color="light",
                className="me-1",
                style={'cursor': 'pointer'}
            )
        ], style={'textAlign': 'right'})
    ], width=4)
], className="mb-2 mt-2")

kpi_cards = dbc.Row([
    # Total Sales Card
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-mug-hot", style={
                        'fontSize': '18px',
                        'color': colors['accent1'],
                        'marginRight': '8px'
                    }),
                    html.Div([
                        html.H6("Total Sales", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                        html.H4(
                            f"{sales_data['Total'].sum():,}",
                            id="total-sales-value",
                            style={'fontWeight': 'bold', 'color': colors['text'], 'margin': '0', 'fontSize': '16px'}
                        )
                    ])
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'padding': '8px'})  # Reduced padding
        ], style=card_style)
    ], width=3),

    # Average Monthly Sales
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-chart-line", style={
                        'fontSize': '18px',
                        'color': colors['accent1'],
                        'marginRight': '8px'
                    }),
                    html.Div([
                        html.H6("Monthly Average", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                        html.H4(
                            f"{int(sales_data['Total'].mean()):,}",
                            id="monthly-avg-value",
                            style={'fontWeight': 'bold', 'color': colors['text'], 'margin': '0', 'fontSize': '16px'}
                        )
                    ])
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'padding': '8px'})
        ], style=card_style)
    ], width=3),

    # Revenue
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-dollar-sign", style={
                        'fontSize': '18px',
                        'color': colors['accent1'],
                        'marginRight': '8px'
                    }),
                    html.Div([
                        html.H6("Total Revenue", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                        html.H4(
                            f"${sales_data['Revenue'].sum():,.2f}",
                            id="revenue-value",
                            style={'fontWeight': 'bold', 'color': colors['text'], 'margin': '0', 'fontSize': '16px'}
                        )
                    ])
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'padding': '8px'})
        ], style=card_style)
    ], width=3),

    # Top Product
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-trophy", style={
                        'fontSize': '18px',
                        'color': colors['accent1'],
                        'marginRight': '8px'
                    }),
                    html.Div([
                        html.H6("Top Product", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                        html.H4(
                            top_product,
                            id="top-product-value",
                            style={'fontWeight': 'bold', 'color': coffee_colors[top_product], 'margin': '0',
                                   'fontSize': '16px'}
                        )
                    ])
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'padding': '8px'})
        ], style=card_style)
    ], width=3),
], className="mb-2")

dashboard_view = dbc.Row([
    # Left column with charts
    dbc.Col([
        # Sales trend chart
        dbc.Card([
            dbc.CardBody([
                html.H6("Annual Coffee Sales Trend",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="sales-trend-chart",
                    figure=px.line(
                        sales_long,
                        x='Month',
                        y='Sales',
                        color='Coffee Type',
                        color_discrete_map=coffee_colors,
                        markers=True,
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
                        xaxis=dict(tickmode='array', tickvals=months, tickfont=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8)),
                        height=150,
                        hovermode="x unified"
                    ).update_traces(
                        line=dict(width=2),
                        marker=dict(size=4),
                        hovertemplate='%{y:,.0f} cups'
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '150px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style),

        # Heatmap - more compact
        dbc.Card([
            dbc.CardBody([
                html.H6("Coffee Sales by Day & Hour",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="heatmap-chart",
                    figure=px.imshow(
                        np.outer(
                            np.array([0.7, 0.6, 0.7, 0.8, 0.9, 1.3, 1.2]),
                            np.array([1.4, 1.6, 1.2, 1.3, 0.8, 0.9, 1.0])
                        ) * np.random.uniform(0.8, 1.2, size=(7, 7)),
                        x=['7AM', '9AM', '11AM', '1PM', '3PM', '5PM', '7PM'],
                        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        color_continuous_scale=[colors['card_bg'], colors['latte'], colors['cappuccino'],
                                                colors['espresso']]
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        height=130,
                        coloraxis_showscale=True,
                        coloraxis_colorbar=dict(
                            title=dict(text=""),
                            thicknessmode="pixels", thickness=8,
                            lenmode="pixels", len=100,
                            tickfont=dict(size=7)
                        ),
                        xaxis=dict(tickfont=dict(size=7)),
                        yaxis=dict(tickfont=dict(size=7))
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '130px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style),
    ], width=7),

    # Right column with charts
    dbc.Col([
        # Donut chart - more compact
        dbc.Card([
            dbc.CardBody([
                html.H6("Share by Coffee Type",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="share-pie-chart",
                    figure=px.pie(
                        pd.DataFrame(total_by_coffee),
                        values='Sales',
                        names='Type',
                        hole=0.6,
                        color='Type',
                        color_discrete_map=coffee_colors
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(size=8)),
                        annotations=[dict(
                            text=f"{sum(total_by_coffee['Sales']):,}",
                            showarrow=False,
                            font=dict(size=14)
                        )],
                        height=145
                    ).update_traces(
                        textposition='inside',
                        textinfo='percent',
                        textfont=dict(size=9),
                        hovertemplate='%{label}<br>%{value:,.0f} cups<br>%{percent}'
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '145px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style),

        # Bar chart - more compact
        dbc.Card([
            dbc.CardBody([
                html.H6("Monthly Sales by Coffee Type",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="monthly-bar-chart",
                    figure=px.bar(
                        sales_long,
                        x='Month',
                        y='Sales',
                        color='Coffee Type',
                        barmode='group',
                        color_discrete_map=coffee_colors,
                        category_orders={"Month": months},
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
                        xaxis=dict(tickfont=dict(size=7)),
                        yaxis=dict(tickfont=dict(size=7)),
                        height=130,
                        bargap=0.15,
                        bargroupgap=0.05
                    ).update_traces(
                        hovertemplate='%{y:,.0f} cups'
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '130px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style)
    ], width=5)
], className="mb-1")

trends_view = dbc.Row([
    # Left column
    dbc.Col([
        # Price trends chart
        dbc.Card([
            dbc.CardBody([
                html.H6("Coffee Price Trends", style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="price-trend-chart",
                    figure=px.line(
                        price_long,
                        x='Month',
                        y='Price',
                        color='Coffee Type',
                        color_discrete_map=coffee_colors,
                        markers=True,
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
                        xaxis=dict(tickmode='array', tickvals=months, tickfont=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8), tickprefix='$'),
                        height=150,
                        hovermode="x unified"
                    ).update_traces(
                        line=dict(width=2),
                        marker=dict(size=4),
                        hovertemplate='$%{y:.2f}'
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '150px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style),

        # Sales vs Price correlation
        dbc.Card([
            dbc.CardBody([
                html.H6("Sales vs Price Correlation",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="correlation-chart",
                    figure=px.scatter(
                        x=[3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
                        y=[1100, 1050, 1000, 970, 930, 880, 850],
                        color_discrete_sequence=[colors['espresso']],
                        labels={"x": "Price ($)", "y": "Sales (cups)"}
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        xaxis=dict(tickfont=dict(size=8), tickprefix='$'),
                        yaxis=dict(tickfont=dict(size=8)),
                        height=130,
                        showlegend=False
                    ).add_shape(
                        type="line",
                        x0=3.2, y0=1100,
                        x1=3.8, y1=850,
                        line=dict(color=colors['espresso'], width=2)
                    ).add_annotation(
                        x=3.5, y=970,
                        text="Price Elasticity: -1.2",
                        showarrow=False,
                        font=dict(size=9)
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '130px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style),
    ], width=6),

    # Right column
    dbc.Col([
        # Seasonal pattern
        dbc.Card([
            dbc.CardBody([
                html.H6("Seasonal Consumption Patterns",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="seasonal-chart",
                    figure=px.bar(
                        pd.DataFrame({
                            'Season': ['Winter', 'Spring', 'Summer', 'Fall'],
                            'Espresso': [1050, 980, 920, 1020],
                            'Latte': [1550, 1480, 1400, 1500],
                            'Cappuccino': [1250, 1200, 1150, 1180]
                        }).melt(
                            id_vars=['Season'],
                            value_vars=['Espresso', 'Latte', 'Cappuccino'],
                            var_name='Coffee Type',
                            value_name='Consumption'
                        ),
                        x='Season',
                        y='Consumption',
                        color='Coffee Type',
                        barmode='group',
                        color_discrete_map=coffee_colors
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
                        xaxis=dict(tickfont=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8)),
                        height=150,
                        bargap=0.15,
                        bargroupgap=0.05
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '150px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style),

        # Customer demographic chart
        dbc.Card([
            dbc.CardBody([
                html.H6("Customer Demographics",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="demographic-chart",
                    figure=px.pie(
                        pd.DataFrame({
                            'Age Group': ['18-25', '26-35', '36-45', '46-55', '56+'],
                            'Percentage': [15, 35, 25, 15, 10]
                        }),
                        values='Percentage',
                        names='Age Group',
                        hole=0.4,
                        color_discrete_sequence=[colors['espresso'], colors['latte'], colors['cappuccino'],
                                                 '#A67B5B', '#8B7355']
                    ).update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(size=8)),
                        height=130
                    ).update_traces(
                        textposition='inside',
                        textinfo='percent',
                        hovertemplate='%{label}<br>%{percent}'
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '130px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style)
    ], width=6)
], className="mb-1")

predictions_view = dbc.Row([
    # Left column
    dbc.Col([
        # Sales predictions chart
        dbc.Card([
            dbc.CardBody([
                html.H6("Sales Forecast (Next 4 Months)",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                dcc.Graph(
                    id="predictions-chart",
                    figure=go.Figure()
                    .add_trace(go.Scatter(
                        x=prediction_months[:12],
                        y=espresso_predictions[:12],
                        mode='lines+markers',
                        name='Espresso',
                        line=dict(color=colors['espresso'], width=2)
                    ))
                    .add_trace(go.Scatter(
                        x=prediction_months[11:],
                        y=espresso_predictions[11:],
                        mode='lines+markers',
                        name='Espresso Forecast',
                        line=dict(color=colors['espresso'], width=2, dash='dot')
                    ))
                    .add_trace(go.Scatter(
                        x=prediction_months[:12],
                        y=latte_predictions[:12],
                        mode='lines+markers',
                        name='Latte',
                        line=dict(color=colors['latte'], width=2)
                    ))
                    .add_trace(go.Scatter(
                        x=prediction_months[11:],
                        y=latte_predictions[11:],
                        mode='lines+markers',
                        name='Latte Forecast',
                        line=dict(color=colors['latte'], width=2, dash='dot')
                    ))
                    .add_trace(go.Scatter(
                        x=prediction_months[:12],
                        y=cappuccino_predictions[:12],
                        mode='lines+markers',
                        name='Cappuccino',
                        line=dict(color=colors['cappuccino'], width=2)
                    ))
                    .add_trace(go.Scatter(
                        x=prediction_months[11:],
                        y=cappuccino_predictions[11:],
                        mode='lines+markers',
                        name='Cappuccino Forecast',
                        line=dict(color=colors['cappuccino'], width=2, dash='dot')
                    ))
                    .update_layout(
                        plot_bgcolor=colors['card_bg'],
                        paper_bgcolor=colors['card_bg'],
                        font=dict(color=colors['text'], size=9),
                        margin=dict(l=5, r=5, t=5, b=5),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
                        xaxis=dict(tickfont=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8)),
                        height=150,
                        hovermode="x unified",
                        shapes=[{
                            'type': 'line',
                            'x0': months[-1],
                            'y0': 0,
                            'x1': months[-1],
                            'y1': 2000,
                            'line': {
                                'color': 'gray',
                                'width': 1,
                                'dash': 'dot',
                            }
                        }],
                        annotations=[{
                            'x': months[-1],
                            'y': 2000,
                            'text': 'Forecast Starts',
                            'showarrow': False,
                            'font': {'size': 8},
                            'xanchor': 'left',
                            'yanchor': 'top'
                        }]
                    ),
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '150px'}
                )
            ], style={'padding': '8px'})
        ], style=card_style),

        # Predicted revenue
        dbc.Card([
            dbc.CardBody([
                html.H6("Forecasted Revenue Growth",
                        style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("$147,862", style={'fontSize': '14px', 'fontWeight': 'bold', 'margin': '0',
                                                           'color': colors['text']}),
                                html.P("Current Annual", style={'fontSize': '9px', 'margin': '0', 'color': '#777'})
                            ], style={'textAlign': 'center'})
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H5("$156,435", style={'fontSize': '14px', 'fontWeight': 'bold', 'margin': '0',
                                                           'color': colors['success']}),
                                html.P("Forecasted Annual", style={'fontSize': '9px', 'margin': '0', 'color': '#777'})
                            ], style={'textAlign': 'center'})
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H5("+5.8%", style={'fontSize': '14px', 'fontWeight': 'bold', 'margin': '0',
                                                        'color': colors['success']}),
                                html.P("Growth Rate", style={'fontSize': '9px', 'margin': '0', 'color': '#777'})
                            ], style={'textAlign': 'center'})
                        ], width=4)
                    ], className="mb-2"),

                    dcc.Graph(
                        figure=go.Figure().add_trace(
                            go.Indicator(
                                mode="gauge+number",
                                value=5.8,
                                title={'text': "Growth", 'font': {'size': 10}},
                                gauge={
                                    'axis': {'range': [0, 10], 'tickwidth': 1, 'tickfont': {'size': 8}},
                                    'bar': {'color': colors['success']},
                                    'steps': [
                                        {'range': [0, 3], 'color': '#EBE3D5'},
                                        {'range': [3, 7], 'color': '#D2B48C'},
                                        {'range': [7, 10], 'color': '#8B4513'}
                                    ],
                                    'threshold': {
                                        'line': {'color': colors['accent2'], 'width': 2},
                                        'thickness': 0.8,
                                        'value': 5.8
                                    }
                                }
                            )
                        ).update_layout(
                            height=90,
                            margin=dict(l=5, r=5, t=5, b=5),
                            paper_bgcolor=colors['card_bg'],
                            font=dict(color=colors['text'], size=9)
                        ),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '120px'}
                    )
                ])
            ], style={'padding': '8px'})
        ], style=card_style)
    ], width=7),

    # Right column
    dbc.Col([
        # Recommendations
        dbc.Card([
            dbc.CardBody([
                html.H6("AI Recommendations", style={'fontSize': '12px', 'margin': '0 0 8px 0', 'fontWeight': 'bold'}),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-lightbulb mr-2", style={'color': colors['warning']}),
                        html.Span("Increase Latte marketing in summer to boost sales", style={'fontSize': '11px'})
                    ], className="mb-2", style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                    html.Div([
                        html.I(className="fas fa-chart-line mr-2", style={'color': colors['info']}),
                        html.Span("Espresso shows strongest growth potential", style={'fontSize': '11px'})
                    ], className="mb-2", style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                    html.Div([
                        html.I(className="fas fa-tag mr-2", style={'color': colors['success']}),
                        html.Span("Consider 5% price increase for Cappuccino", style={'fontSize': '11px'})
                    ], className="mb-2", style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                    html.Div([
                        html.I(className="fas fa-clock mr-2", style={'color': colors['accent2']}),
                        html.Span("Extend morning hours to capture peak demand", style={'fontSize': '11px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                ], style={'marginBottom': '10px'}),
                dbc.Button("Generate New Insights",
                           color="light",
                           size="sm",
                           style={
                               'backgroundColor': colors['sidebar'],
                               'borderColor': colors['accent1'],
                               'color': colors['text'],
                               'fontSize': '10px',
                               'width': '100%'
                           })
            ], style={'padding': '10px'})
        ], style=card_style),

        # Price optimization
        dbc.Card([
            dbc.CardBody([
                html.H6("Price Optimization", style={'fontSize': '12px', 'margin': '0 0 8px 0', 'fontWeight': 'bold'}),
                html.Div([
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Product",
                                        style={'fontSize': '10px', 'textAlign': 'left', 'paddingRight': '8px',
                                               'paddingBottom': '6px'}),
                                html.Th("Current",
                                        style={'fontSize': '10px', 'textAlign': 'right', 'paddingRight': '8px',
                                               'paddingBottom': '6px'}),
                                html.Th("Optimal",
                                        style={'fontSize': '10px', 'textAlign': 'right', 'paddingRight': '8px',
                                               'paddingBottom': '6px'}),
                                html.Th("∆ Revenue",
                                        style={'fontSize': '10px', 'textAlign': 'right', 'paddingBottom': '6px'})
                            ], style={'borderBottom': '1px solid #ddd'})
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td([
                                    html.Div(style={'width': '8px', 'height': '8px', 'borderRadius': '50%',
                                                    'backgroundColor': colors['espresso'], 'display': 'inline-block',
                                                    'marginRight': '5px'}),
                                    "Espresso"
                                ], style={'fontSize': '10px', 'whiteSpace': 'nowrap', 'padding': '8px 0'}),
                                html.Td("$3.50", style={'fontSize': '10px', 'textAlign': 'right', 'padding': '8px 0'}),
                                html.Td("$3.75",
                                        style={'fontSize': '10px', 'textAlign': 'right', 'color': colors['success'],
                                               'padding': '8px 0'}),
                                html.Td("+4.2%",
                                        style={'fontSize': '10px', 'textAlign': 'right', 'color': colors['success'],
                                               'padding': '8px 0'})
                            ], style={'borderBottom': '1px solid #f5f5f5'}),
                            html.Tr([
                                html.Td([
                                    html.Div(style={'width': '8px', 'height': '8px', 'borderRadius': '50%',
                                                    'backgroundColor': colors['latte'], 'display': 'inline-block',
                                                    'marginRight': '5px'}),
                                    "Latte"
                                ], style={'fontSize': '10px', 'whiteSpace': 'nowrap', 'padding': '8px 0'}),
                                html.Td("$4.50", style={'fontSize': '10px', 'textAlign': 'right', 'padding': '8px 0'}),
                                html.Td("$4.50", style={'fontSize': '10px', 'textAlign': 'right', 'padding': '8px 0'}),
                                html.Td("0%", style={'fontSize': '10px', 'textAlign': 'right', 'padding': '8px 0'})
                            ], style={'borderBottom': '1px solid #f5f5f5'}),
                            html.Tr([
                                html.Td([
                                    html.Div(style={'width': '8px', 'height': '8px', 'borderRadius': '50%',
                                                    'backgroundColor': colors['cappuccino'], 'display': 'inline-block',
                                                    'marginRight': '5px'}),
                                    "Cappuccino"
                                ], style={'fontSize': '10px', 'whiteSpace': 'nowrap', 'padding': '8px 0'}),
                                html.Td("$4.20", style={'fontSize': '10px', 'textAlign': 'right', 'padding': '8px 0'}),
                                html.Td("$4.40",
                                        style={'fontSize': '10px', 'textAlign': 'right', 'color': colors['success'],
                                               'padding': '8px 0'}),
                                html.Td("+3.5%",
                                        style={'fontSize': '10px', 'textAlign': 'right', 'color': colors['success'],
                                               'padding': '8px 0'})
                            ], style={'borderBottom': '1px solid #f5f5f5'})
                        ])
                    ], style={'width': '100%'}),

                    # 添加更多空间和额外内容
                    html.Div([
                        html.Hr(style={'margin': '15px 0 10px 0', 'opacity': '0.3'}),
                        html.Div([
                            html.Small("Suggested price changes for Q2 2025",
                                       style={'fontSize': '9px', 'fontStyle': 'italic', 'color': '#777',
                                              'marginBottom': '5px'}),
                        ], style={'textAlign': 'center'}),
                        html.Div([
                            html.Small([
                                html.I(className="fas fa-info-circle mr-1",
                                       style={'color': colors['accent1'], 'marginRight': '4px'}),
                                f"Last updated: {current_time} by {current_user1}"
                            ], style={'fontSize': '8px', 'color': '#999', 'display': 'block', 'textAlign': 'center',
                                      'marginTop': '10px'})
                        ])
                    ])
                ], style={'minHeight': '100px'})  # 设置最小高度
            ], style={'padding': '10px'})  # 增加内边距
        ], style=dict(card_style, **{'height': 'calc(50%)'}))
    ], width=5)
], className="mb-1")

active_view_store = dcc.Store(id='active-view-store', data='dashboard')
active_filter_store = dcc.Store(id='active-filter-store', data='all')

content_area = html.Div([
    active_view_store,
    active_filter_store,
    kpi_cards,
    html.Div(id='view-content', children=[dashboard_view])
], style={
    'marginLeft': '150px',  # Make room for the sidebar
    'padding': '10px',
    'height': '100vh',
    'overflowY': 'auto'
})

app.layout = html.Div([
    sidebar,
    html.Div([
        header,
        content_area
    ], style={'width': 'calc(100% - 220px)', 'float': 'right'})
], style={
    'backgroundColor': colors['background'],
    'fontFamily': '"Raleway", sans-serif',
    'height': '100vh',
    'overflowX': 'hidden'
})


def filter_data(coffee_filter):
    """根据过滤器状态过滤数据"""
    if coffee_filter == 'all':
        filtered_sales = sales_long
        total_sales = sales_data['Total'].sum()
        monthly_avg = int(sales_data['Total'].mean())
        total_revenue = sales_data['Revenue'].sum()
        top_coffee = top_product
        pie_data = pd.DataFrame(total_by_coffee)
        heatmap_colors = [colors['card_bg'], colors['latte'], colors['cappuccino'], colors['espresso']]
    elif coffee_filter == 'Espresso':
        filtered_sales = sales_long[sales_long['Coffee Type'] == 'Espresso']
        total_sales = sales_data['Espresso'].sum()
        monthly_avg = int(sales_data['Espresso'].mean())
        total_revenue = (sales_data['Espresso'] * sales_data['EspressoPrice']).sum()
        top_coffee = 'Espresso'
        pie_data = pd.DataFrame({'Type': ['Espresso'], 'Sales': [total_sales]})
        heatmap_colors = [colors['card_bg'], colors['espresso']]
    elif coffee_filter == 'Latte':
        filtered_sales = sales_long[sales_long['Coffee Type'] == 'Latte']
        total_sales = sales_data['Latte'].sum()
        monthly_avg = int(sales_data['Latte'].mean())
        total_revenue = (sales_data['Latte'] * sales_data['LattePrice']).sum()
        top_coffee = 'Latte'
        pie_data = pd.DataFrame({'Type': ['Latte'], 'Sales': [total_sales]})
        heatmap_colors = [colors['card_bg'], colors['latte']]
    elif coffee_filter == 'Cappuccino':
        filtered_sales = sales_long[sales_long['Coffee Type'] == 'Cappuccino']
        total_sales = sales_data['Cappuccino'].sum()
        monthly_avg = int(sales_data['Cappuccino'].mean())
        total_revenue = (sales_data['Cappuccino'] * sales_data['CappuccinoPrice']).sum()
        top_coffee = 'Cappuccino'
        pie_data = pd.DataFrame({'Type': ['Cappuccino'], 'Sales': [total_sales]})
        heatmap_colors = [colors['card_bg'], colors['cappuccino']]
    else:
        # 默认情况
        filtered_sales = sales_long
        total_sales = sales_data['Total'].sum()
        monthly_avg = int(sales_data['Total'].mean())
        total_revenue = sales_data['Revenue'].sum()
        top_coffee = top_product
        pie_data = pd.DataFrame(total_by_coffee)
        heatmap_colors = [colors['card_bg'], colors['latte'], colors['cappuccino'], colors['espresso']]

    return {
        'filtered_sales': filtered_sales,
        'total_sales': total_sales,
        'monthly_avg': monthly_avg,
        'total_revenue': total_revenue,
        'top_coffee': top_coffee,
        'pie_data': pie_data,
        'heatmap_colors': heatmap_colors
    }


def create_kpi_cards(filtered_data):
    total_sales = filtered_data['total_sales']
    monthly_avg = filtered_data['monthly_avg']
    total_revenue = filtered_data['total_revenue']
    top_coffee = filtered_data['top_coffee']

    return dbc.Row([
        # Total Sales Card
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-mug-hot", style={
                            'fontSize': '18px',
                            'color': colors['accent1'],
                            'marginRight': '8px'
                        }),
                        html.Div([
                            html.H6("Total Sales", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                            html.H4(
                                f"{total_sales:,}",
                                id="total-sales-value",
                                style={'fontWeight': 'bold', 'color': colors['text'], 'margin': '0', 'fontSize': '16px'}
                            )
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'padding': '8px'})
            ], style=card_style)
        ], width=3),

        # Average Monthly Sales
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line", style={
                            'fontSize': '18px',
                            'color': colors['accent1'],
                            'marginRight': '8px'
                        }),
                        html.Div([
                            html.H6("Monthly Average", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                            html.H4(
                                f"{monthly_avg:,}",
                                id="monthly-avg-value",
                                style={'fontWeight': 'bold', 'color': colors['text'], 'margin': '0', 'fontSize': '16px'}
                            )
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'padding': '8px'})
            ], style=card_style)
        ], width=3),

        # Revenue
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-dollar-sign", style={
                            'fontSize': '18px',
                            'color': colors['accent1'],
                            'marginRight': '8px'
                        }),
                        html.Div([
                            html.H6("Total Revenue", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                            html.H4(
                                f"${total_revenue:,.2f}",
                                id="revenue-value",
                                style={'fontWeight': 'bold', 'color': colors['text'], 'margin': '0', 'fontSize': '16px'}
                            )
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'padding': '8px'})
            ], style=card_style)
        ], width=3),

        # Top Product
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-trophy", style={
                            'fontSize': '18px',
                            'color': colors['accent1'],
                            'marginRight': '8px'
                        }),
                        html.Div([
                            html.H6("Top Product", style={'fontSize': '10px', 'margin': '0', 'color': '#666'}),
                            html.H4(
                                top_coffee,
                                id="top-product-value",
                                style={'fontWeight': 'bold', 'color': coffee_colors.get(top_coffee, colors['text']),
                                       'margin': '0',
                                       'fontSize': '16px'}
                            )
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'padding': '8px'})
            ], style=card_style)
        ], width=3),
    ], className="mb-2")

def generate_dashboard_view(coffee_filter):
    # 获取过滤后的数据
    filtered_data = filter_data(coffee_filter)
    filtered_sales = filtered_data['filtered_sales']
    pie_data = filtered_data['pie_data']
    heatmap_colors = filtered_data['heatmap_colors']
    total_sales = filtered_data['total_sales']

    # 创建KPI卡片
    kpi_cards_updated = create_kpi_cards(filtered_data)

    # 生成图表
    line_fig = px.line(
        filtered_sales, x='Month', y='Sales', color='Coffee Type',
        color_discrete_map=coffee_colors, markers=True
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
        xaxis=dict(tickmode='array', tickvals=months, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        height=150,
        hovermode="x unified"
    ).update_traces(
        line=dict(width=2),
        marker=dict(size=4),
        hovertemplate='%{y:,.0f} cups'
    )

    pie_fig = px.pie(
        pie_data, values='Sales', names='Type', hole=0.6,
        color='Type', color_discrete_map=coffee_colors
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(size=8)),
        annotations=[dict(text=f"{total_sales:,}", showarrow=False, font=dict(size=14))],
        height=145
    ).update_traces(
        textposition='inside',
        textinfo='percent',
        textfont=dict(size=9),
        hovertemplate='%{label}<br>%{value:,.0f} cups<br>%{percent}'
    )

    bar_fig = px.bar(
        filtered_sales, x='Month', y='Sales', color='Coffee Type',
        barmode='group', color_discrete_map=coffee_colors,
        category_orders={"Month": months}
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
        xaxis=dict(tickfont=dict(size=7)),
        yaxis=dict(tickfont=dict(size=7)),
        height=130,
        bargap=0.15,
        bargroupgap=0.05
    ).update_traces(
        hovertemplate='%{y:,.0f} cups'
    )

    heatmap_fig = px.imshow(
        np.outer(np.array([0.7, 0.6, 0.7, 0.8, 0.9, 1.3, 1.2]),
                 np.array([1.4, 1.6, 1.2, 1.3, 0.8, 0.9, 1.0])) * np.random.uniform(0.8, 1.2, size=(7, 7)),
        x=['7AM', '9AM', '11AM', '1PM', '3PM', '5PM', '7PM'],
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        color_continuous_scale=heatmap_colors
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        height=130,
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title=dict(text=""),
            thicknessmode="pixels", thickness=8,
            lenmode="pixels", len=100,
            tickfont=dict(size=7)
        ),
        xaxis=dict(tickfont=dict(size=7)),
        yaxis=dict(tickfont=dict(size=7))
    )

    dashboard_view_updated = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Annual Coffee Sales Trend",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="sales-trend-chart",
                        figure=line_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '150px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style),

            dbc.Card([
                dbc.CardBody([
                    html.H6("Coffee Sales by Day & Hour",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="heatmap-chart",
                        figure=heatmap_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '130px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style),
        ], width=7),

        dbc.Col([
            # 环形图 - 更紧凑
            dbc.Card([
                dbc.CardBody([
                    html.H6("Share by Coffee Type",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="share-pie-chart",
                        figure=pie_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '145px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style),

            dbc.Card([
                dbc.CardBody([
                    html.H6("Monthly Sales by Coffee Type",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="monthly-bar-chart",
                        figure=bar_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '130px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style)
        ], width=5)
    ], className="mb-1")

    return [kpi_cards_updated, dashboard_view_updated]

def generate_trends_view(coffee_filter):
    filtered_data = filter_data(coffee_filter)

    kpi_cards_updated = create_kpi_cards(filtered_data)

    if coffee_filter == 'all':
        price_data = price_long
    else:
        price_data = price_long[price_long['Coffee Type'] == coffee_filter]

    price_fig = px.line(
        price_data, x='Month', y='Price', color='Coffee Type',
        color_discrete_map=coffee_colors, markers=True,
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
        xaxis=dict(tickmode='array', tickvals=months, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), tickprefix='$'),
        height=150,
        hovermode="x unified"
    ).update_traces(
        line=dict(width=2),
        marker=dict(size=4),
        hovertemplate='$%{y:.2f}'
    )

    season_data = pd.DataFrame({
        'Season': ['Winter', 'Spring', 'Summer', 'Fall']
    })

    if coffee_filter == 'all' or coffee_filter == 'Espresso':
        season_data['Espresso'] = [1050, 980, 920, 1020]
    if coffee_filter == 'all' or coffee_filter == 'Latte':
        season_data['Latte'] = [1550, 1480, 1400, 1500]
    if coffee_filter == 'all' or coffee_filter == 'Cappuccino':
        season_data['Cappuccino'] = [1250, 1200, 1150, 1180]

    season_long = pd.melt(
        season_data,
        id_vars=['Season'],
        value_vars=[col for col in season_data.columns if col != 'Season'],
        var_name='Coffee Type',
        value_name='Consumption'
    )

    seasonal_fig = px.bar(
        season_long,
        x='Season',
        y='Consumption',
        color='Coffee Type',
        barmode='group',
        color_discrete_map=coffee_colors
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        height=150,
        bargap=0.15,
        bargroupgap=0.05
    )

    corr_x = [3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8]
    corr_y = [1100, 1050, 1000, 970, 930, 880, 850]

    if coffee_filter == 'Latte':
        corr_x = [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8]
        corr_y = [1600, 1550, 1500, 1470, 1430, 1380, 1350]
    elif coffee_filter == 'Cappuccino':
        corr_x = [3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
        corr_y = [1300, 1250, 1200, 1170, 1130, 1080, 1050]

    corr_fig = px.scatter(
        x=corr_x, y=corr_y,
        color_discrete_sequence=[coffee_colors.get(coffee_filter, colors['espresso'])],
        labels={"x": "Price ($)", "y": "Sales (cups)"}
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(tickfont=dict(size=8), tickprefix='$'),
        yaxis=dict(tickfont=dict(size=8)),
        height=130,
        showlegend=False
    ).add_shape(
        type="line",
        x0=min(corr_x), y0=max(corr_y),
        x1=max(corr_x), y1=min(corr_y),
        line=dict(color=coffee_colors.get(coffee_filter, colors['espresso']), width=2)
    ).add_annotation(
        x=corr_x[3], y=corr_y[3],
        text="Price Elasticity: -1.2",
        showarrow=False,
        font=dict(size=9)
    )

    demo_fig = px.pie(
        pd.DataFrame({
            'Age Group': ['18-25', '26-35', '36-45', '46-55', '56+'],
            'Percentage': [15, 35, 25, 15, 10]
        }),
        values='Percentage',
        names='Age Group',
        hole=0.4,
        color_discrete_sequence=[colors['espresso'], colors['latte'], colors['cappuccino'],
                                 '#A67B5B', '#8B7355']
    ).update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(size=8)),
        height=130
    ).update_traces(
        textposition='inside',
        textinfo='percent',
        hovertemplate='%{label}<br>%{percent}'
    )
    trends_view_updated = dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Coffee Price Trends",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="price-trend-chart",
                        figure=price_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '150px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style),

            dbc.Card([
                dbc.CardBody([
                    html.H6("Sales vs Price Correlation",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="correlation-chart",
                        figure=corr_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '130px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style),
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Seasonal Consumption Patterns",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="seasonal-chart",
                        figure=seasonal_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '150px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style),

            dbc.Card([
                dbc.CardBody([
                    html.H6("Customer Demographics",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="demographic-chart",
                        figure=demo_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '130px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style)
        ], width=6)
    ], className="mb-1")

    return [kpi_cards_updated, trends_view_updated]

def generate_predictions_view(coffee_filter):

    filtered_data = filter_data(coffee_filter)

    kpi_cards_updated = create_kpi_cards(filtered_data)

    prediction_fig = go.Figure()

    if coffee_filter == 'all' or coffee_filter == 'Espresso':
        prediction_fig.add_trace(go.Scatter(
            x=prediction_months[:12],
            y=espresso_predictions[:12],
            mode='lines+markers',
            name='Espresso',
            line=dict(color=colors['espresso'], width=2)
        ))
        prediction_fig.add_trace(go.Scatter(
            x=prediction_months[11:],
            y=espresso_predictions[11:],
            mode='lines+markers',
            name='Espresso Forecast',
            line=dict(color=colors['espresso'], width=2, dash='dot')
        ))

    if coffee_filter == 'all' or coffee_filter == 'Latte':
        prediction_fig.add_trace(go.Scatter(
            x=prediction_months[:12],
            y=latte_predictions[:12],
            mode='lines+markers',
            name='Latte',
            line=dict(color=colors['latte'], width=2)
        ))
        prediction_fig.add_trace(go.Scatter(
            x=prediction_months[11:],
            y=latte_predictions[11:],
            mode='lines+markers',
            name='Latte Forecast',
            line=dict(color=colors['latte'], width=2, dash='dot')
        ))

    if coffee_filter == 'all' or coffee_filter == 'Cappuccino':
        prediction_fig.add_trace(go.Scatter(
            x=prediction_months[:12],
            y=cappuccino_predictions[:12],
            mode='lines+markers',
            name='Cappuccino',
            line=dict(color=colors['cappuccino'], width=2)
        ))

        prediction_fig.add_trace(go.Scatter(
            x=prediction_months[11:],
            y=cappuccino_predictions[11:],
            mode='lines+markers',
            name='Cappuccino Forecast',
            line=dict(color=colors['cappuccino'], width=2, dash='dot')
        ))

    prediction_fig.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font=dict(color=colors['text'], size=9),
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        height=150,
        hovermode="x unified",
        shapes=[{
            'type': 'line',
            'x0': months[-1],
            'y0': 0,
            'x1': months[-1],
            'y1': 2000,
            'line': {
                'color': 'gray',
                'width': 1,
                'dash': 'dot',
            }
        }],
        annotations=[{
            'x': months[-1],
            'y': 2000,
            'text': 'Forecast Starts',
            'showarrow': False,
            'font': {'size': 8},
            'xanchor': 'left',
            'yanchor': 'top'
        }]
    )

    current_annual = 147862
    forecasted_annual = 156435
    growth_rate = 5.8

    if coffee_filter == 'Espresso':
        current_annual = 42000
        forecasted_annual = 45360
        growth_rate = 8.0
    elif coffee_filter == 'Latte':
        current_annual = 65862
        forecasted_annual = 68825
        growth_rate = 4.5
    elif coffee_filter == 'Cappuccino':
        current_annual = 40000
        forecasted_annual = 42250
        growth_rate = 5.6

    recommendations = []

    if coffee_filter == 'all':
        recommendations = [
            {"icon": "fas fa-lightbulb", "color": colors['warning'],
             "text": "Increase Latte marketing in summer to boost sales"},
            {"icon": "fas fa-chart-line", "color": colors['info'],
             "text": "Espresso shows strongest growth potential"},
            {"icon": "fas fa-tag", "color": colors['success'],
             "text": "Consider 5% price increase for Cappuccino"},
            {"icon": "fas fa-clock", "color": colors['accent2'],
             "text": "Extend morning hours to capture peak demand"}
        ]
    elif coffee_filter == 'Espresso':
        recommendations = [
            {"icon": "fas fa-lightbulb", "color": colors['warning'],
             "text": "Introduce seasonal espresso variations"},
            {"icon": "fas fa-chart-line", "color": colors['info'],
             "text": "7-8% growth potential with targeted marketing"},
            {"icon": "fas fa-tag", "color": colors['success'],
             "text": "Price increase of 7% possible without affecting demand"},
            {"icon": "fas fa-clock", "color": colors['accent2'],
             "text": "Peak demand from 7-9AM and 2-3PM"}
        ]
    elif coffee_filter == 'Latte':
        recommendations = [
            {"icon": "fas fa-lightbulb", "color": colors['warning'],
             "text": "Promote seasonal flavored lattes"},
            {"icon": "fas fa-chart-line", "color": colors['info'],
             "text": "Social media campaigns show 12% engagement"},
            {"icon": "fas fa-tag", "color": colors['success'],
             "text": "Current price point optimal for volume"},
            {"icon": "fas fa-clock", "color": colors['accent2'],
             "text": "Afternoon sales potential with promotions"}
        ]
    elif coffee_filter == 'Cappuccino':
        recommendations = [
            {"icon": "fas fa-lightbulb", "color": colors['warning'],
             "text": "Introduce art cappuccino to premium segment"},
            {"icon": "fas fa-chart-line", "color": colors['info'],
             "text": "Growing at 5.6%, can reach 7% with campaigns"},
            {"icon": "fas fa-tag", "color": colors['success'],
             "text": "Price elasticity allows for 5-7% increase"},
            {"icon": "fas fa-clock", "color": colors['accent2'],
             "text": "Weekend sales highest between 9-11AM"}
        ]

    price_data = []

    if coffee_filter == 'all' or coffee_filter == 'Espresso':
        price_data.append({
            "product": "Espresso",
            "color": colors['espresso'],
            "current": "$3.50",
            "optimal": "$3.75",
            "change": "+4.2%",
            "highlight": True
        })

    if coffee_filter == 'all' or coffee_filter == 'Latte':
        price_data.append({
            "product": "Latte",
            "color": colors['latte'],
            "current": "$4.50",
            "optimal": "$4.50",
            "change": "0%",
            "highlight": False
        })

    if coffee_filter == 'all' or coffee_filter == 'Cappuccino':
        price_data.append({
            "product": "Cappuccino",
            "color": colors['cappuccino'],
            "current": "$4.20",
            "optimal": "$4.40",
            "change": "+3.5%",
            "highlight": True
        })

    predictions_view_updated = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Sales Forecast (Next 4 Months)",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    dcc.Graph(
                        id="predictions-chart",
                        figure=prediction_fig,
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '150px'}
                    )
                ], style={'padding': '8px'})
            ], style=card_style),
            dbc.Card([
                dbc.CardBody([
                    html.H6("Forecasted Revenue Growth",
                            style={'fontSize': '12px', 'margin': '0 0 5px 0', 'fontWeight': 'bold'}),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5(f"${current_annual:,}",
                                            style={'fontSize': '14px', 'fontWeight': 'bold', 'margin': '0',
                                                   'color': colors['text']}),
                                    html.P("Current Annual", style={'fontSize': '9px', 'margin': '0', 'color': '#777'})
                                ], style={'textAlign': 'center'})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H5(f"${forecasted_annual:,}",
                                            style={'fontSize': '14px', 'fontWeight': 'bold', 'margin': '0',
                                                   'color': colors['success']}),
                                    html.P("Forecasted Annual",
                                           style={'fontSize': '9px', 'margin': '0', 'color': '#777'})
                                ], style={'textAlign': 'center'})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H5(f"+{growth_rate}%",
                                            style={'fontSize': '14px', 'fontWeight': 'bold', 'margin': '0',
                                                   'color': colors['success']}),
                                    html.P("Growth Rate", style={'fontSize': '9px', 'margin': '0', 'color': '#777'})
                                ], style={'textAlign': 'center'})
                            ], width=4)
                        ], className="mb-2"),

                        dcc.Graph(
                            figure=go.Figure().add_trace(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=growth_rate,
                                    title={'text': "Growth", 'font': {'size': 10}},
                                    gauge={
                                        'axis': {'range': [0, 10], 'tickwidth': 1, 'tickfont': {'size': 8}},
                                        'bar': {'color': colors['success']},
                                        'steps': [
                                            {'range': [0, 3], 'color': '#EBE3D5'},
                                            {'range': [3, 7], 'color': '#D2B48C'},
                                            {'range': [7, 10], 'color': '#8B4513'}
                                        ],
                                        'threshold': {
                                            'line': {'color': colors['accent2'], 'width': 2},
                                            'thickness': 0.8,
                                            'value': growth_rate
                                        }
                                    }
                                )
                            ).update_layout(
                                height=90,
                                margin=dict(l=5, r=5, t=5, b=5),
                                paper_bgcolor=colors['card_bg'],
                                font=dict(color=colors['text'], size=9)
                            ),
                            config={'displayModeBar': False, 'responsive': True},
                            style={'height': '120px'}
                        )
                    ])
                ], style={'padding': '8px'})
            ], style=card_style)
        ], width=7),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Recommendations",
                            style={'fontSize': '12px', 'margin': '0 0 8px 0', 'fontWeight': 'bold'}),
                    html.Div([
                        *[html.Div([
                            html.I(className=rec["icon"] + " mr-2", style={'color': rec["color"]}),
                            html.Span(rec["text"], style={'fontSize': '11px'})
                        ], className="mb-2", style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'})
                            for rec in recommendations],
                    ], style={'marginBottom': '10px'}),
                    dbc.Button("Generate New Insights",
                               color="light",
                               size="sm",
                               style={
                                   'backgroundColor': colors['sidebar'],
                                   'borderColor': colors['accent1'],
                                   'color': colors['text'],
                                   'fontSize': '10px',
                                   'width': '100%'
                               })
                ], style={'padding': '10px'})
            ], style=card_style),

            # 价格优化
            dbc.Card([
                dbc.CardBody([
                    html.H6("Price Optimization",
                            style={'fontSize': '12px', 'margin': '0 0 8px 0', 'fontWeight': 'bold'}),
                    html.Div([
                        html.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Product",
                                            style={'fontSize': '10px', 'textAlign': 'left', 'paddingRight': '8px',
                                                   'paddingBottom': '6px'}),
                                    html.Th("Current",
                                            style={'fontSize': '10px', 'textAlign': 'right', 'paddingRight': '8px',
                                                   'paddingBottom': '6px'}),
                                    html.Th("Optimal",
                                            style={'fontSize': '10px', 'textAlign': 'right', 'paddingRight': '8px',
                                                   'paddingBottom': '6px'}),
                                    html.Th("∆ Revenue",
                                            style={'fontSize': '10px', 'textAlign': 'right', 'paddingBottom': '6px'})
                                ], style={'borderBottom': '1px solid #ddd'})
                            ]),
                            html.Tbody([
                                *[html.Tr([
                                    html.Td([
                                        html.Div(style={'width': '8px', 'height': '8px', 'borderRadius': '50%',
                                                        'backgroundColor': item["color"], 'display': 'inline-block',
                                                        'marginRight': '5px'}),
                                        item["product"]
                                    ], style={'fontSize': '10px', 'whiteSpace': 'nowrap', 'padding': '8px 0'}),
                                    html.Td(item["current"],
                                            style={'fontSize': '10px', 'textAlign': 'right', 'padding': '8px 0'}),
                                    html.Td(item["optimal"],
                                            style={'fontSize': '10px', 'textAlign': 'right',
                                                   'color': colors['success'] if item["highlight"] else 'inherit',
                                                   'padding': '8px 0'}),
                                    html.Td(item["change"],
                                            style={'fontSize': '10px', 'textAlign': 'right',
                                                   'color': colors['success'] if item["highlight"] else 'inherit',
                                                   'padding': '8px 0'})
                                ], style={'borderBottom': '1px solid #f5f5f5'}) for item in price_data]
                            ])
                        ], style={'width': '100%'}),

                        html.Div([
                            html.Hr(style={'margin': '15px 0 10px 0', 'opacity': '0.3'}),
                            html.Div([
                                html.Small(
                                    f"Suggested price changes for Q2 2025 - {coffee_filter if coffee_filter != 'all' else 'All Products'}",
                                    style={'fontSize': '9px', 'fontStyle': 'italic', 'color': '#777',
                                           'marginBottom': '5px'}),
                            ], style={'textAlign': 'center'}),
                            html.Div([
                                html.Small([
                                    html.I(className="fas fa-info-circle mr-1",
                                           style={'color': colors['accent1'], 'marginRight': '4px'}),
                                ], style={'fontSize': '8px', 'color': '#999', 'display': 'block', 'textAlign': 'center',
                                          'marginTop': '10px'})
                            ])
                        ])
                    ], style={'minHeight': '100px'})
                ], style={'padding': '10px'})
            ], style=dict(card_style, **{'height': 'calc(50% - 5px)'}))
        ], width=5)
    ], className="mb-1")

    return [kpi_cards_updated, predictions_view_updated]


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        /* 添加到app.index_string的<style>标签中 */
            #_dash-dev-tools {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0 !important;
                width: 0 !important;
                position: absolute !important;
                z-index: -1000 !important;
            }
            body {
                overflow: hidden;
                margin: 0;
                padding: 0;
            }
            html {
                overflow: hidden;
            }
            @media (max-height: 800px) {
                .dash-graph {
                    height: 90% !important; 
                }
                .card-body {
                    padding: 4px !important;
                }
            }
            @media (max-height: 700px) {
                .dash-graph {
                    height: 80% !important;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.Script('''
        document.addEventListener('DOMContentLoaded', function() {
            // 根据屏幕高度动态调整缩放
            const vh = window.innerHeight;
            if (vh < 800) {
                document.body.style.zoom = "90%";
            }
            if (vh < 700) {
                document.body.style.zoom = "80%";
            }
        });
    '''),
    sidebar,
    html.Div([
        header,
        content_area
    ], style={
        'width': 'calc(100% - 220px)',
        'float': 'right',
        'overflow': 'hidden'
    })
], style={
    'backgroundColor': colors['background'],
    'fontFamily': '"Raleway", sans-serif',
    'height': '100vh',
    'overflow': 'hidden'
})

@app.callback(
    [Output('nav-dashboard', 'active'),
     Output('nav-trends', 'active'),
     Output('nav-predictions', 'active')],
    [Input('active-view-store', 'data')]
)
def update_nav_active(active_view):
    is_dashboard = active_view == 'dashboard'
    is_trends = active_view == 'trends'
    is_predictions = active_view == 'predictions'

    return is_dashboard, is_trends, is_predictions

@app.callback(
    [Output('filter-all', 'active'),
     Output('filter-espresso', 'active'),
     Output('filter-latte', 'active'),
     Output('filter-cappuccino', 'active')],
    [Input('active-filter-store', 'data')]
)
def update_filter_active(active_filter):
    is_all = active_filter == 'all'
    is_espresso = active_filter == 'Espresso'
    is_latte = active_filter == 'Latte'
    is_cappuccino = active_filter == 'Cappuccino'

    return is_all, is_espresso, is_latte, is_cappuccino

@app.callback(
    [Output('view-content', 'children'),
     Output('active-view-store', 'data'),
     Output('active-view', 'children')],
    [Input('nav-dashboard', 'n_clicks'),
     Input('nav-trends', 'n_clicks'),
     Input('nav-predictions', 'n_clicks')],
    [State('active-view-store', 'data'),
     State('active-filter-store', 'data')]
)
def update_view(dashboard_clicks, trends_clicks, predictions_clicks, active_view, active_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        return generate_dashboard_view(active_filter), 'dashboard', "OVERVIEW"

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'nav-dashboard':
        return generate_dashboard_view(active_filter), 'dashboard', "OVERVIEW"
    elif button_id == 'nav-trends':
        return generate_trends_view(active_filter), 'trends', "TRENDS"
    elif button_id == 'nav-predictions':
        return generate_predictions_view(active_filter), 'predictions', "PREDICTIONS"

    return generate_dashboard_view(active_filter), 'dashboard', "OVERVIEW"

@app.callback(
    Output('active-filter-store', 'data'),
    [Input('filter-espresso', 'n_clicks'),
     Input('filter-latte', 'n_clicks'),
     Input('filter-cappuccino', 'n_clicks'),
     Input('filter-all', 'n_clicks')],
    [State('active-filter-store', 'data')]
)
def update_coffee_filter(espresso_clicks, latte_clicks, cappuccino_clicks, all_clicks, active_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        return active_filter

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'filter-all':
        return 'all'
    elif button_id == 'filter-espresso':
        return 'Espresso'
    elif button_id == 'filter-latte':
        return 'Latte'
    elif button_id == 'filter-cappuccino':
        return 'Cappuccino'

    return active_filter

@app.callback(
    Output('view-content', 'children', allow_duplicate=True),
    [Input('active-filter-store', 'data')],
    [State('active-view-store', 'data')],
    prevent_initial_call=True
)
def update_view_on_filter_change(active_filter, active_view):
    if active_view == 'dashboard':
        return generate_dashboard_view(active_filter)
    elif active_view == 'trends':
        return generate_trends_view(active_filter)
    elif active_view == 'predictions':
        return generate_predictions_view(active_filter)

    return generate_dashboard_view(active_filter)


if __name__ == '__main__':
<<<<<<< HEAD
    app.run_server(debug=False)
=======
    app.run_server(debug=True)
>>>>>>> f65a5e4702dd662a2a010d746c7bd50ccc1139b6
