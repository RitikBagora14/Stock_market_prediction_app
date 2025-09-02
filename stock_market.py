import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


# Dates
START = '2015-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecast App")

# Valid Yahoo Finance tickers for NSE India
stocks = ("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "RELIANCE.NS")

selected_stock = st.selectbox("Choose a stock for forecasting", stocks)

# Prediction period
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(stock):
    data = yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Data loaded successfully!")

# Show raw data
st.subheader("Raw data")
st.write(data.tail())

# def plot_raw_data():
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open Price'))
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],  name='Close Price'))
#     fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)
# plot_raw_data()

def plot_raw_data():
    fig = go.Figure()

    # Add Open price line
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Open'],
        mode='lines',
        name='stock_open',
        line=dict(color='blue')
    ))

    # Add Close price line
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'],
        mode='lines',
        name='stock_close',
        line=dict(color='red')
    ))

    # Layout styling
    fig.update_layout(
        title="Stock Prices Over Time (2015–2021)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        xaxis_rangeslider_visible=True,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()


### Forecasting

df_train = data[['Date','Close']]
df_train.columns = ['ds', 'y']

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader("Forecasted data")
st.write(forecast.tail())

st.write(f"Forecast plot for {n_years} years")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)


st.write("Forecast components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)