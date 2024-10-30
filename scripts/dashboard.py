# dashboard.py
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import requests
import plotly.express as px
import pandas as pd

# Initialize the Dash app
app = Dash(__name__)
server = app.server  # Integrate with Flask

# Load data from the Flask API
def get_summary_data():
    response = requests.get("http://127.0.0.1:5000/api/summary")
    return response.json()

def get_trends_data():
    response = requests.get("http://127.0.0.1:5000/api/trends")
    return pd.DataFrame(response.json().items(), columns=['Date', 'Fraud Cases'])

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    # Summary boxes
    html.Div(id='summary-boxes', style={'display': 'flex', 'gap': '20px'}),
    
    # Line chart for fraud trends over time
    dcc.Graph(id='trend-line-chart'),
])

# Callback to update summary boxes
@app.callback(
    Output('summary-boxes', 'children'),
    Input('trend-line-chart', 'id')  # Trigger the callback once on page load
)
def update_summary_boxes(_):
    data = get_summary_data()
    return [
        html.Div(f"Total Transactions: {data['total_transactions']}"),
        html.Div(f"Total Fraud Cases: {data['total_fraud_cases']}"),
        html.Div(f"Fraud Percentage: {data['fraud_percentage']:.2f}%")
    ]

# Callback to update fraud trends chart
@app.callback(
    Output('trend-line-chart', 'figure'),
    Input('trend-line-chart', 'id')
)
def update_trends_chart(_):
    df = get_trends_data()
    fig = px.line(df, x='Date', y='Fraud Cases', title='Fraud Cases Over Time')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
