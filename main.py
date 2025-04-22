# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uvicorn
import json

app = FastAPI(
    title="金融数据API",
    description="基于akshare的金融数据提取API",
    version="1.0.0"
)

# 定义请求模型
class FinancialDataRequest(BaseModel):
    asset_type: str
    asset_code: str
    start_date: str
    end_date: str
    indicators: List[str]

# 首页
@app.get("/")
def read_root():
    return {"message": "欢迎使用金融数据API", "docs_url": "/docs"}

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 获取金融数据API
@app.post("/api/financial_data")
def get_financial_data(request: FinancialDataRequest):
    try:
        # 提取请求参数
        asset_type = request.asset_type
        asset_code = request.asset_code
        start_date = request.start_date
        end_date = request.end_date
        indicators = request.indicators
        
        # 根据资产类型调用不同的处理函数
        if asset_type == "股票":
            result = get_stock_data(asset_code, start_date, end_date, indicators)
        elif asset_type == "基金":
            result = get_fund_data(asset_code, start_date, end_date, indicators)
        elif asset_type == "期货":
            result = get_futures_data(asset_code, start_date, end_date, indicators)
        elif asset_type == "外汇":
            result = get_forex_data(asset_code, start_date, end_date, indicators)
        elif asset_type == "债券":
            result = get_bond_data(asset_code, start_date, end_date, indicators)
        elif asset_type == "指数":
            result = get_index_data(asset_code, start_date, end_date, indicators)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的资产类型: {asset_type}")
        
        # 返回结果
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 股票数据处理函数
def get_stock_data(code, start_date, end_date, indicators):
    results = {}
    
    try:
        # 确保股票代码格式正确
        if code.startswith(('SH', 'SZ', 'BJ')):
            formatted_code = code
        else:
            # 简单规则：6开头为上海，0和3开头为深圳
            if code.startswith('6'):
                formatted_code = f"sh{code}"
            elif code.startswith(('0', '3')):
                formatted_code = f"sz{code}"
            else:
                formatted_code = code
        
        # 获取股票历史数据
        if "历史价格" in indicators or any(i in indicators for i in ["开盘价", "收盘价", "最高价", "最低价", "交易量"]):
            try:
                stock_data = ak.stock_zh_a_hist(symbol=code, 
                                               start_date=start_date,
                                               end_date=end_date, 
                                               adjust="qfq")
                results["历史数据"] = stock_data_to_dict(stock_data)
            except Exception as e1:
                # 尝试另一种格式
                try:
                    stock_data = ak.stock_zh_a_hist(symbol=formatted_code, 
                                                   start_date=start_date,
                                                   end_date=end_date, 
                                                   adjust="qfq")
                    results["历史数据"] = stock_data_to_dict(stock_data)
                except Exception as e2:
                    results["历史数据错误"] = f"原格式错误: {str(e1)}，格式化后错误: {str(e2)}"
        
        # 计算波动率
        if "波动率" in indicators and "历史数据" in results:
            try:
                historical_data = pd.DataFrame(results["历史数据"])
                
                # 价格列
                price_col = '收盘' if '收盘' in historical_data.columns else 'close'
                
                # 计算日收益率
                historical_data['daily_return'] = historical_data[price_col].pct_change()
                
                # 计算20日历史波动率
                volatility = historical_data['daily_return'].rolling(window=20).std() * np.sqrt(252) * 100
                
                # 日期列
                date_col = '日期' if '日期' in historical_data.columns else 'date'
                
                # 构建结果DataFrame
                volatility_df = pd.DataFrame({date_col: historical_data[date_col], '20日波动率(%)': volatility})
                results["波动率"] = stock_data_to_dict(volatility_df.dropna())
            except Exception as e:
                results["波动率错误"] = str(e)
        
        # 计算均线
        if "均线" in indicators and "历史数据" in results:
            try:
                historical_data = pd.DataFrame(results["历史数据"])
                
                # 价格列
                price_col = '收盘' if '收盘' in historical_data.columns else 'close'
                
                # 日期列
                date_col = '日期' if '日期' in historical_data.columns else 'date'
                
                # 计算MA均线
                historical_data['MA5'] = historical_data[price_col].rolling(window=5).mean()
                historical_data['MA10'] = historical_data[price_col].rolling(window=10).mean()
                historical_data['MA20'] = historical_data[price_col].rolling(window=20).mean()
                historical_data['MA60'] = historical_data[price_col].rolling(window=60).mean()
                
                # 构建结果DataFrame
                ma_df = historical_data[[date_col, price_col, 'MA5', 'MA10', 'MA20', 'MA60']].dropna()
                results["均线数据"] = stock_data_to_dict(ma_df)
            except Exception as e:
                results["均线数据错误"] = str(e)
        
        # 计算MACD
        if "MACD" in indicators and "历史数据" in results:
            try:
                historical_data = pd.DataFrame(results["历史数据"])
                
                # 价格列
                price_col = '收盘' if '收盘' in historical_data.columns else 'close'
                
                # 日期列
                date_col = '日期' if '日期' in historical_data.columns else 'date'
                
                # 计算MACD
                close = historical_data[price_col]
                exp1 = close.ewm(span=12, adjust=False).mean()
                exp2 = close.ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = macd - signal
                
                macd_df = pd.DataFrame({
                    date_col: historical_data[date_col],
                    'MACD': macd,
                    'MACD信号线': signal,
                    'MACD柱状': hist
                }).dropna()
                
                results["MACD"] = stock_data_to_dict(macd_df)
            except Exception as e:
                results["MACD错误"] = str(e)
        
        # 计算RSI
        if "RSI" in indicators and "历史数据" in results:
            try:
                historical_data = pd.DataFrame(results["历史数据"])
                
                # 价格列
                price_col = '收盘' if '收盘' in historical_data.columns else 'close'
                
                # 日期列
                date_col = '日期' if '日期' in historical_data.columns else 'date'
                
                # 计算RSI
                delta = historical_data[price_col].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain_14 = gain.rolling(window=14).mean()
                avg_loss_14 = loss.rolling(window=14).mean()
                
                rs_14 = avg_gain_14 / avg_loss_14
                rsi_14 = 100 - (100 / (1 + rs_14))
                
                rsi_df = pd.DataFrame({
                    date_col: historical_data[date_col],
                    'RSI_14': rsi_14
                }).dropna()
                
                results["RSI"] = stock_data_to_dict(rsi_df)
            except Exception as e:
                results["RSI错误"] = str(e)
            
    except Exception as e:
        results["错误"] = str(e)
    
    return results

# 基金数据处理函数
def get_fund_data(code, start_date, end_date, indicators):
    results = {}
    
    try:
        # 获取基金历史净值
        if "历史价格" in indicators or "净值" in indicators:
            fund_data = ak.fund_open_fund_info_em(fund=code, indicator="单位净值走势")
            # 筛选时间范围
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            fund_data = fund_data[(fund_data['净值日期'] >= start_date_dt) & 
                                 (fund_data['净值日期'] <= end_date_dt)]
            
            # 将日期转换回字符串
            fund_data['净值日期'] = fund_data['净值日期'].dt.strftime('%Y-%m-%d')
            
            results["历史净值"] = stock_data_to_dict(fund_data)
        
        # 获取基金信息
        if "基金信息" in indicators:
            fund_info = ak.fund_open_fund_info_em(fund=code, indicator="基金概况")
            results["基金信息"] = stock_data_to_dict(fund_info)
        
        # 计算波动率
        if "波动率" in indicators and "历史净值" in results:
            fund_data = pd.DataFrame(results["历史净值"])
            # 计算日收益率
            fund_data['daily_return'] = fund_data['单位净值'].pct_change()
            # 计算20日历史波动率
            volatility = fund_data['daily_return'].rolling(window=20).std() * np.sqrt(252) * 100
            volatility_df = pd.DataFrame({'净值日期': fund_data['净值日期'], '20日波动率(%)': volatility})
            results["波动率"] = stock_data_to_dict(volatility_df.dropna())
        
    except Exception as e:
        results["错误"] = str(e)
    
    return results

# 期货数据处理函数
def get_futures_data(code, start_date, end_date, indicators):
    results = {}
    
    try:
        # 获取期货历史数据
        if "历史价格" in indicators:
            futures_data = ak.futures_zh_daily_sina(symbol=code)
            # 筛选时间范围
            futures_data['date'] = pd.to_datetime(futures_data['date'])
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            futures_data = futures_data[(futures_data['date'] >= start_date_dt) & 
                                       (futures_data['date'] <= end_date_dt)]
            
            # 将日期转换回字符串
            futures_data['date'] = futures_data['date'].dt.strftime('%Y-%m-%d')
            
            results["历史数据"] = stock_data_to_dict(futures_data)
        
        # 计算波动率
        if "波动率" in indicators and "历史数据" in results:
            futures_data = pd.DataFrame(results["历史数据"])
            # 计算日收益率
            futures_data['daily_return'] = futures_data['close'].pct_change()
            # 计算20日历史波动率
            volatility = futures_data['daily_return'].rolling(window=20).std() * np.sqrt(252) * 100
            volatility_df = pd.DataFrame({'date': futures_data['date'], '20日波动率(%)': volatility})
            results["波动率"] = stock_data_to_dict(volatility_df.dropna())
        
        # 计算均线
        if "均线" in indicators and "历史数据" in results:
            futures_data = pd.DataFrame(results["历史数据"])
            futures_data['MA5'] = futures_data['close'].rolling(window=5).mean()
            futures_data['MA10'] = futures_data['close'].rolling(window=10).mean()
            futures_data['MA20'] = futures_data['close'].rolling(window=20).mean()
            
            ma_df = futures_data[['date', 'close', 'MA5', 'MA10', 'MA20']].dropna()
            results["均线数据"] = stock_data_to_dict(ma_df)
        
    except Exception as e:
        results["错误"] = str(e)
    
    return results

# 外汇数据处理函数
def get_forex_data(code, start_date, end_date, indicators):
    results = {}
    
    try:
        # 获取外汇历史数据
        if "历史价格" in indicators:
            # 判断外汇对类型
            if code.upper() in ["USD/CNY", "USDCNY", "美元人民币"]:
                forex_data = ak.currency_hist()
                # 筛选时间范围
                forex_data['日期'] = pd.to_datetime(forex_data['日期'])
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)
                forex_data = forex_data[(forex_data['日期'] >= start_date_dt) & 
                                       (forex_data['日期'] <= end_date_dt)]
                
                # 将日期转换回字符串
                forex_data['日期'] = forex_data['日期'].dt.strftime('%Y-%m-%d')
                
                results["历史数据"] = stock_data_to_dict(forex_data)
            else:
                # 其他货币对可以使用不同的API
                results["错误"] = f"暂不支持{code}外汇对数据"
                
    except Exception as e:
        results["错误"] = str(e)
    
    return results

# 债券数据处理函数
def get_bond_data(code, start_date, end_date, indicators):
    results = {}
    
    try:
        # 获取债券历史数据
        if "历史价格" in indicators:
            # 判断债券类型
            if code in ["国债", "yield"]:
                bond_data = ak.bond_zh_us_rate()
                results["国债收益率曲线"] = stock_data_to_dict(bond_data)
            else:
                # 尝试获取特定债券
                try:
                    bond_data = ak.bond_zh_hs_cov_daily(symbol=code)
                    # 筛选时间范围
                    bond_data['date'] = pd.to_datetime(bond_data['date'])
                    start_date_dt = pd.to_datetime(start_date)
                    end_date_dt = pd.to_datetime(end_date)
                    bond_data = bond_data[(bond_data['date'] >= start_date_dt) & 
                                         (bond_data['date'] <= end_date_dt)]
                    
                    # 将日期转换回字符串
                    bond_data['date'] = bond_data['date'].dt.strftime('%Y-%m-%d')
                    
                    results["历史数据"] = stock_data_to_dict(bond_data)
                except Exception as e:
                    results["错误"] = f"无法获取债券{code}的数据: {str(e)}"
                    
    except Exception as e:
        results["错误"] = str(e)
    
    return results

# 指数数据处理函数
def get_index_data(code, start_date, end_date, indicators):
    results = {}
    
    try:
        # 获取指数历史数据
        if "历史价格" in indicators:
            # 处理常见的指数代码
            if code in ["000001", "上证指数", "上证综指"]:
                formatted_code = "000001"
            elif code in ["399001", "深证成指"]:
                formatted_code = "399001"
            elif code in ["399006", "创业板指"]:
                formatted_code = "399006"
            else:
                formatted_code = code
            
            index_data = ak.stock_zh_index_daily(symbol=formatted_code)
            # 筛选时间范围
            index_data['date'] = pd.to_datetime(index_data['date'])
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            index_data = index_data[(index_data['date'] >= start_date_dt) & 
                                   (index_data['date'] <= end_date_dt)]
            
            # 将日期转换回字符串
            index_data['date'] = index_data['date'].dt.strftime('%Y-%m-%d')
            
            results["历史数据"] = stock_data_to_dict(index_data)
        
        # 计算波动率
        if "波动率" in indicators and "历史数据" in results:
            index_data = pd.DataFrame(results["历史数据"])
            # 计算日收益率
            index_data['daily_return'] = index_data['close'].pct_change()
            # 计算20日历史波动率
            volatility = index_data['daily_return'].rolling(window=20).std() * np.sqrt(252) * 100
            volatility_df = pd.DataFrame({'date': index_data['date'], '20日波动率(%)': volatility})
            results["波动率"] = stock_data_to_dict(volatility_df.dropna())
            
        # 计算均线
        if "均线" in indicators and "历史数据" in results:
            index_data = pd.DataFrame(results["历史数据"])
            index_data['MA5'] = index_data['close'].rolling(window=5).mean()
            index_data['MA10'] = index_data['close'].rolling(window=10).mean()
            index_data['MA20'] = index_data['close'].rolling(window=20).mean()
            index_data['MA60'] = index_data['close'].rolling(window=60).mean()
            
            ma_df = index_data[['date', 'close', 'MA5', 'MA10', 'MA20', 'MA60']].dropna()
            results["均线数据"] = stock_data_to_dict(ma_df)
            
    except Exception as e:
        results["错误"] = str(e)
    
    return results

# 辅助函数：DataFrame转Dict
def stock_data_to_dict(df):
    """将DataFrame转换为字典列表"""
    # 处理日期列，确保可以被JSON序列化
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d')
    
    return json.loads(df.to_json(orient="records", date_format='iso'))

# 主程序入口
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)