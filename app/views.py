from django.shortcuts import render
import ccxt
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.utils
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import re

# Create your views here.

def home(request):
    context = {}
    return render(request, 'app/home.html', context)


def crypto_price(request):
    exchange = ccxt.binance()  # Sử dụng Binance làm sàn giao dịch
    symbol = request.GET.get('symbol', 'BTC/USDT').replace('-', '/')  # Chuyển đổi ký hiệu cho phù hợp với CCXT

    try:
        ticker = exchange.fetch_ticker(symbol)  # Lấy dữ liệu giá crypto
        percentage_change = ((ticker['last'] - ticker['open']) / ticker['open']) * 100  # Tính phần trăm thay đổi giá

        # Chuyển timestamp sang múi giờ UTC
        utc_time = datetime.utcfromtimestamp(ticker['timestamp'] / 1000)
        utc_tz = pytz.timezone('UTC')
        last_updated = utc_time.replace(tzinfo=pytz.utc).astimezone(utc_tz).strftime('%Y-%m-%d %H:%M:%S UTC')

        context = {
            'symbol': symbol.replace('/', '-'),
            'price': ticker['last'],
            'high': ticker['high'],
            'low': ticker['low'],
            'volume': ticker['baseVolume'],
            'percentage_change': round(percentage_change, 2),  # Làm tròn 2 chữ số
            'last_updated': last_updated,
            'error': None
        }
    except Exception as e:
        context = {
            'symbol': symbol,
            'error': f"Không tìm thấy dữ liệu cho {symbol}. Hãy thử lại!"
        }

    return render(request, 'app/index.html', context)


def market_data(request):
    exchange = ccxt.binance({'timeout': 10000})  # Đặt thời gian timeout API là 10s
    
    # Lấy symbol từ request và chuyển đổi sang định dạng chuẩn
    symbol = request.GET.get('symbol', 'BTC-USDT').upper().replace('/', '-')
    if '-' not in symbol:
        symbol = 'BTC-USDT'
    symbol_ccxt = symbol.replace('-', '/')
    
    # Kiểm tra tính hợp lệ của timeframe
    valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
    timeframe = request.GET.get('timeframe', '1h')
    if timeframe not in valid_timeframes:
        timeframe = '1h'
    
    # Lấy limit từ request và đảm bảo ít nhất có 14 dữ liệu cho việc tính RSI
    try:
        limit = int(request.GET.get('limit', 100))
        limit = max(limit + 14, 114)  # Thêm 14 điểm dữ liệu để tính RSI chính xác
    except ValueError:
        limit = 114  # 100 + 14 điểm dữ liệu

    try:
        # Lấy danh sách các cặp giao dịch USDT và loại bỏ cặp trùng lặp
        markets = exchange.load_markets()
        symbols = sorted({
            market.replace('/USDT/USDT', '/USDT').replace('/', '-') 
            for market in markets if '/USDT' in market
        })

        # Lấy dữ liệu OHLCV
        ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe, limit=limit)
        
        # Chuyển dữ liệu OHLCV sang DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Ho_Chi_Minh')
        
        # Tính toán sự thay đổi giá
        df['price_change'] = df['close'].diff()
        
        # Tính toán lợi nhuận và thua lỗ
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Tính toán trung bình lợi nhuận và thua lỗ
        window = 14
        df['avg_gain'] = df['gain'].rolling(window=window, min_periods=window).mean()
        df['avg_loss'] = df['loss'].rolling(window=window, min_periods=window).mean()
        
        # Tính toán RS và RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = np.nan  # Khởi tạo cột RSI với giá trị NaN
        mask = df.index >= window-1  # Tạo mask cho các dòng từ 14 trở đi
        df.loc[mask, 'rsi'] = 100 - (100 / (1 + df.loc[mask, 'rs']))
        
        # Xử lý các trường hợp đặc biệt (chỉ cho các dòng đã có RSI)
        df.loc[mask & (df['avg_loss'] == 0), 'rsi'] = 100  # Nếu không có thua lỗ, RSI = 100
        df.loc[mask & (df['avg_gain'] == 0) & (df['avg_loss'] == 0), 'rsi'] = 50  # Nếu không có thay đổi, RSI = 50
        
        # Chỉ giữ lại số lượng dòng theo yêu cầu ban đầu
        df = df.iloc[-limit+14:]  # Bỏ 14 dòng đầu đã dùng để tính RSI
        
        # Đảm bảo dòng đầu tiên là NaN
        df.iloc[0, df.columns.get_loc('rsi')] = np.nan
        
        # Chuẩn bị dữ liệu bảng
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        table_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi']].to_dict('records')
        
        context = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit-14,  # Trừ đi 14 dòng đã dùng để tính RSI
            'table_data': table_data,
            'symbols': symbols,
            'error': None
        }
    except Exception as e:
        context = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit,
            'symbols': symbols if 'symbols' in locals() else [],
            'table_data': [],
            'error': f"Lỗi khi tải dữ liệu cho {symbol}: {str(e)}. Vui lòng thử lại với một cặp khác."
        }
    
    return render(request, 'app/market_data.html', context)


def fundamental_analysis(request):
    try:
        # Get symbol from request or default to BTC-USDT
        symbol = request.GET.get('symbol', 'BTC-USDT')
        
        # Validate symbol format
        if not re.match(r'^[A-Za-z0-9]+-[A-Za-z0-9]+$', symbol):
            symbol = 'BTC-USDT'
        
        # Initialize exchange
        exchange = ccxt.binance()
        
        # Load available markets and filter out duplicates
        markets = exchange.load_markets()
        symbols = []
        seen_symbols = set()
        for market_symbol in markets.keys():
            if market_symbol.endswith('USDT'):
                # Remove duplicate USDT pairs (e.g., "BTC-USDT-USDT")
                base_symbol = market_symbol.split('/')[0]
                if base_symbol not in seen_symbols:
                    symbols.append(market_symbol)
                    seen_symbols.add(base_symbol)
        symbols.sort()
        
        # Convert symbol format for API calls (replace - with /)
        symbol_ccxt = symbol.replace('-', '/')
        
        # Fetch market data
        ticker = exchange.fetch_ticker(symbol_ccxt)
        orderbook = exchange.fetch_order_book(symbol_ccxt)
        
        # Check if ticker and orderbook data are valid
        if not ticker or not orderbook:
            raise ValueError(f"Invalid data for {symbol}. Please try again later.")
        
        # Calculate fundamental metrics
        market_cap = float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0
        volume_price_ratio = float(ticker['baseVolume']) if ticker['baseVolume'] else 0
        price_change_24h = float(ticker['percentage']) if ticker['percentage'] else 0
        
        # Calculate bid-ask spread
        best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else 0
        best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
        bid_ask_spread = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
        
        # Calculate market health score (0-100)
        # Volume component (30%)
        volume_score = min(100, (volume_price_ratio / 1000)) * 0.3
        
        # Price stability component (30%)
        stability_score = (100 - min(100, abs(price_change_24h))) * 0.3
        
        # Liquidity component (40%)
        liquidity_score = (100 - min(100, bid_ask_spread * 10)) * 0.4
        
        # Combine scores
        market_health = volume_score + stability_score + liquidity_score
        
        # Get circulating supply (this is a placeholder - in reality, you'd need to fetch this from a blockchain explorer)
        circulating_supply = market_cap / ticker['last'] if ticker['last'] > 0 else 0
        
        # Create visualizations
        # Market metrics gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=market_health,
            title={'text': "Market Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        gauge_fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
        
        # Price stability gauge
        stability_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=100 - bid_ask_spread,
            title={'text': "Price Stability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        stability_fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
        
        # Market activity gauge
        activity_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(100, volume_price_ratio / 10),
            title={'text': "Market Activity"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        activity_fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
        
        # Combine all figures
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Market Health", "Price Stability", "Market Activity", "Price Performance"),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                  [{"type": "indicator"}, {"type": "xy"}]]
        )
        
        # Add gauge charts
        for trace in gauge_fig.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in stability_fig.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in activity_fig.data:
            fig.add_trace(trace, row=2, col=1)
        
        # Add price performance line chart
        fig.add_trace(
            go.Scatter(
                x=[ticker['timestamp']] if 'timestamp' in ticker else [0],  # Handling missing timestamp
                y=[ticker['last']],
                mode='lines+markers',
                name='Current Price',
                line=dict(color='blue')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Fundamental Analysis - {symbol}",
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Prepare context
        context = {
            'symbol': symbol,
            'symbols': symbols,
            'plot_data': fig.to_json(),
            'fundamental_metrics': {
                'market_cap': market_cap,
                'volume_price_ratio': volume_price_ratio,
                'price_change_24h': price_change_24h,
                'bid_ask_spread': bid_ask_spread,
                'market_health': market_health,
                'component_scores': {
                    'volume_score': volume_score / 0.3,
                    'stability_score': stability_score / 0.3,
                    'liquidity_score': liquidity_score / 0.4
                },
                'circulating_supply': circulating_supply
            }
        }
        
        return render(request, 'app/fundamental_analysis.html', context)
    except Exception as e:
        return render(request, 'app/fundamental_analysis.html', {
            'error': f'Error analyzing {symbol}: {str(e)}',
            'symbol': symbol,
            'symbols': symbols if 'symbols' in locals() else []
        })





