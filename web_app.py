import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib

# Load the model
model = joblib.load('./artifacts/model_2.pkl')
df = pd.read_csv('./artifacts/feature_importance.csv', index_col=0)

features = df.index.tolist()
importance = df.iloc[:, 0].tolist()

#Initialise the Dash App
app = dash.Dash(__name__)
server = app.server

#Define App Layout
app.layout = html.Div(
    children=[
        html.H1("Churn Eligibility Predictor"),
        html.Label("CustomerID:"),
        dcc.Input(id='CustomerID', type='number', value=0),
        html.Div(),
        html.Label("Gender:"),
        dcc.Dropdown(
            ['Male','Female'],
            value='Male',
            id='Gender'
        ),
        html.Label("Age:"),
        dcc.Slider(0, 100, 1, marks={i: f'${i}' for i in range(0, 101, 10)}, tooltip={"always_visible": True}, value=20, id='Age'),
        html.Label("Income:"),
        dcc.Slider(0, 100000, 1000, marks={i: f'${i}' for i in range(0, 100001, 10000)}, tooltip={"always_visible": True}, value=10000, id='Income'),
        html.Label("TotalPurchase:"),
        dcc.Slider(0, 10000, 100, marks={i: f'${i}' for i in range(0, 10001, 1000)}, tooltip={"always_visible": True}, value=1000, id='TotalPurchase'),
        html.Label("NumOfPurchases:"),
        dcc.Input(id='NumOfPurchases', type='number', value=1),
        html.Label("Location:"),
        dcc.Dropdown(
            ['Urban','Suburban','Rural'], 
            value='Urban',
            id='Location'
        ),
        html.Label("MaritalStatus:"),
        dcc.Dropdown(
            ['Married','Single'],
            value='Married',
            id='MaritalStatus'
        ),
        html.Label("Education:"),
        dcc.Dropdown(
            ['High School',"Bachelor's","Master's",'PhD'],
            value='High School',
            id='Education'
        ),
        html.Label("Subscription Plan:"),
        dcc.Dropdown(
            ['Gold','Bronze','Silver'],
            value='Bronze',
            id='SubscriptionPlan'
        ),
        html.Button('Check Eligibility', id='submit-val', n_clicks=0),
        html.Div(id='output'),
        dcc.Graph(id='graph')
    ]
)
# Define Callback Function for Predictions
@app.callback(
    Output(component_id='output', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    Input(component_id='CustomerID', component_property='value'),
    Input(component_id='Gender', component_property='value'),
    Input(component_id='Age', component_property='value'),
    Input(component_id='Income', component_property='value'),
    Input(component_id='TotalPurchase', component_property='value'),
    Input(component_id='NumOfPurchases', component_property='value'),
    Input(component_id='Location', component_property='value'),
    Input(component_id='MaritalStatus', component_property='value'),
    Input(component_id='Education', component_property='value'),
    Input(component_id='SubscriptionPlan', component_property='value')
)
def update_output(n_clicks, CustomerID, Gender, Age, Income, TotalPurchase, NumOfPurchases,
                    Location, MaritalStatus, Education, SubscriptionPlan):
    if not(n_clicks is None):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'CustomerID': [CustomerID],
            'gender': [Gender],
            'Age': [Age],
            'income': [Income],
            'total_purchase': [TotalPurchase],
            'num_of_purchases': [NumOfPurchases],
            'location': [Location],
            'marital_status': [MaritalStatus],
            'education': [Education],
            'subscription_plan': [SubscriptionPlan]
        })
        input_data.drop(
            columns=['CustomerID','Age'],
            inplace=True
        )
        # Bin income into brackets
        bins = [0, 30000, 50000, 70000, float('inf')]
        labels = ['Low Income', 'Medium Income', 'High Income', 'Very High Income']
        input_data['income_bin'] = pd.cut(input_data['income'], bins=bins, labels=labels, right=False)

        # Added average feature
        input_data['average_purchase'] = round(input_data['total_purchase'] / input_data['num_of_purchases'],0)
        
        input_data = input_data[['gender', 'income', 'income_bin', 'total_purchase', 'num_of_purchases', 'average_purchase', 'location', 'marital_status', 'education', 'subscription_plan']]

        prediction = model.predict(input_data)
        if prediction == 1:
            return html.Div('Churn predicted', style={'color': 'green'})
        else:
            return html.Div('No churn predicted', style={'color': 'red'})
        
# Define callback to update the graph
@app.callback(
    Output('graph', 'figure'),
    [Input('graph', 'id')]
)
def update_graph(_):
    # Create horizontal bar graph
    fig = {
        'data': [
            {
                'x': importance,
                'y': features,
                'type': 'bar',
                'orientation': 'h'
            }
        ],
        'layout': {
            'title': 'Feature Importance',
            'xaxis': {'title': 'Importance'},
            'yaxis': {'title': 'Feature'},
        }
    }
    return fig
#Run the App
if __name__ == '__main__':
    app.run_server(debug=True)
