import pandas as pd
import numpy as np
import crawler  
import crawler_finmind # 🌟 新增：引入 FinMind 爬蟲
from tabulate import tabulate
import re
import unicodedata
from tensorflow import keras

# ==========================================
# 模型載入區塊 
# ==========================================
model_filename = 'lstm_trading_model.keras'
try:
    ml_model = keras.models.load_model(model_filename)
    print(f"✅ 成功載入機器學習大腦：{model_filename}")
except FileNotFoundError:
    ml_model = None
    print(f"⚠️ 找不到 {model_filename}，將跳過 ML 濾網功能。")


def _get_numeric_value(df, row_idx, candidates, default=0.0):
    """回傳 DataFrame 某一列的數值，支援中英欄位名與缺欄位安全回退。"""
    for col in candidates:
        if col in df.columns:
            value = df.at[row_idx, col]
            if pd.notna(value):
                return float(value)
    return float(default)


def filter_with_ml(latest_features_dict):
    """傳入最新一週的特徵字典，回傳 (是否建議進場, 預測未來 4 週報酬%)"""
    if ml_model is None:
        return True, 0.0

    feature_cols = [
        '1週報酬率',
        '4週報酬率',
        '8週報酬率',
        '5日均線偏離率',
        '20日均線偏離率',
        '成交量增幅',
        '外資淨買賣超/成交量',
        '融資融券比率'
    ]

    seq_values = pd.DataFrame([latest_features_dict])[feature_cols].to_numpy(dtype=float)
    seq_values = np.tile(seq_values, (5, 1))
    X_new = seq_values.reshape(1, 5, len(feature_cols))

    pred_return = float(ml_model.predict(X_new, verbose=0)[0][0])
    prediction = (pred_return >= 0.0)
    return prediction, pred_return

# ==========================================
# 核心邏輯：判斷某個時間點是否符合進場條件
# ==========================================
def check_conditions(df, i, continuous_weeks=3, min_growth=0.0479, pop_decline_threshold=0.198,
                     corr_window=156, large_corr_thresh=0.6, retail_corr_thresh=-0.6, avg_corr_thresh=0.6, 
                     last_week_threshold=0.179, skip_cond_e=False):
                     
    large_holder_col = '>400張百分比'
    if '>400張百分比' not in df.columns:
        return False, 0, 0, 0, 0
        
    if i < continuous_weeks: return False, 0, 0, 0, 0

    weekly_growth_a = [((df.at[i-j, large_holder_col] - df.at[i-j-1, large_holder_col]) / df.at[i-j-1, large_holder_col]) * 100 if df.at[i-j-1, large_holder_col] > 0 else -np.inf for j in range(continuous_weeks)]
    if not (all(g > 0 for g in weekly_growth_a) and (weekly_growth_a[0] > last_week_threshold)): 
        return False, 0, 0, 0, 0

    weekly_growth_b = [((df.at[i-j, '平均張數/人'] - df.at[i-j-1, '平均張數/人']) / df.at[i-j-1, '平均張數/人']) * 100 if df.at[i-j-1, '平均張數/人'] > 0 else -np.inf for j in range(continuous_weeks)]
    if not all(g > min_growth for g in weekly_growth_b): 
        return False, 0, 0, 0, 0

    pop_decline_pct = ((df.at[i-continuous_weeks, '總股東人數'] - df.at[i, '總股東人數']) / df.at[i-continuous_weeks, '總股東人數']) * 100
    if pop_decline_pct <= pop_decline_threshold: 
        return False, 0, 0, 0, 0

    actual_window = min(corr_window, i + 1)
    x_large = df.loc[i-actual_window+1:i, large_holder_col].reset_index(drop=True)
    x_avg_per_person = df.loc[i-actual_window+1:i, '平均張數/人'].reset_index(drop=True)
    x_shareholders = df.loc[i-actual_window+1:i, '總股東人數'].reset_index(drop=True)
    y_close = df.loc[i-actual_window+1:i, '收盤價'].reset_index(drop=True) 

    corr_val = x_large.corr(y_close)
    avg_corr_val = x_avg_per_person.corr(y_close)
    retail_corr_val = x_shareholders.corr(y_close)

    corr_val = 0.0 if pd.isna(corr_val) else corr_val
    avg_corr_val = 0.0 if pd.isna(avg_corr_val) else avg_corr_val
    retail_corr_val = 0.0 if pd.isna(retail_corr_val) else retail_corr_val

    if corr_val >= large_corr_thresh or avg_corr_val >= avg_corr_thresh or retail_corr_val <= retail_corr_thresh:
        return True, corr_val, retail_corr_val, avg_corr_val, actual_window

    return False, 0, 0, 0, 0

# ==========================================
# 預測與歷史釣魚模組
# ==========================================
def scan_latest_and_history(df, params): 
    stock_id = df['股票代號'].iloc[0]
    i_latest = len(df) - 1
    
    # 1. 嚴格初篩：先用 GA 參數檢查最新一週
    is_triggered, corr, retail_corr, avg_corr, actual_win = check_conditions(df, i_latest, **params)
    if not is_triggered:
        return None, None

    # ==========================================
    # 🌟 2. 初篩通過！動態抓取 FinMind 最新特徵並合併
    # ==========================================
    df['資料日期'] = pd.to_datetime(df['資料日期'])
    start_str = df['資料日期'].min().strftime('%Y-%m-%d')
    end_str = df['資料日期'].max().strftime('%Y-%m-%d')
    
    df_finmind = crawler_finmind.fetch_weekly_chip_data(stock_id, start_str, end_str)
    
    foreign_net = 0
    margin_bal = 0
    short_bal = 0
    
    if df_finmind is not None and not df_finmind.empty:
        df_finmind['date'] = pd.to_datetime(df_finmind['date'])
        df_finmind = df_finmind.sort_values('date')
        df = df.sort_values('資料日期')
        df = pd.merge_asof(df, df_finmind, left_on='資料日期', right_on='date', direction='backward', tolerance=pd.Timedelta(days=3))
        
        if 'Foreign_Buy_Sum' in df.columns and 'Foreign_Sell_Sum' in df.columns:
            foreign_net = df.at[i_latest, 'Foreign_Buy_Sum'] - df.at[i_latest, 'Foreign_Sell_Sum']
            if pd.isna(foreign_net): foreign_net = 0
        if 'MarginPurchaseBalance' in df.columns:
            margin_bal = df.at[i_latest, 'MarginPurchaseBalance']
            if pd.isna(margin_bal): margin_bal = 0
        if 'ShortSaleBalance' in df.columns:
            short_bal = df.at[i_latest, 'ShortSaleBalance']
            if pd.isna(short_bal): short_bal = 0

    try:
        large_growth = ((df.at[i_latest, '>400張百分比'] - df.at[i_latest-4, '>400張百分比']) / df.at[i_latest-4, '>400張百分比']) * 100
        retail_decline = ((df.at[i_latest-4, '總股東人數'] - df.at[i_latest, '總股東人數']) / df.at[i_latest-4, '總股東人數']) * 100
    except:
        large_growth = 0
        retail_decline = 0

    latest_volume = _get_numeric_value(df, i_latest, ['成交量', 'Volume'], 0.0)
    past_volume = _get_numeric_value(df, max(i_latest - 5, 0), ['成交量', 'Volume'], 0.0)
    volume_change_pct = 0.0
    if i_latest >= 5 and past_volume != 0:
        volume_change_pct = ((latest_volume - past_volume) / past_volume) * 100

    latest_features = {
        '1週報酬率': round(float((df.at[i_latest, '收盤價'] - df.at[max(i_latest - 5, 0), '收盤價']) / df.at[max(i_latest - 5, 0), '收盤價']) * 100, 6) if i_latest >= 5 else 0.0,
        '4週報酬率': round(float((df.at[i_latest, '收盤價'] - df.at[max(i_latest - 20, 0), '收盤價']) / df.at[max(i_latest - 20, 0), '收盤價']) * 100, 6) if i_latest >= 20 else 0.0,
        '8週報酬率': round(float((df.at[i_latest, '收盤價'] - df.at[max(i_latest - 40, 0), '收盤價']) / df.at[max(i_latest - 40, 0), '收盤價']) * 100, 6) if i_latest >= 40 else 0.0,
        '5日均線偏離率': round(float((df.at[i_latest, '收盤價'] - df.loc[max(i_latest - 4, 0):i_latest, '收盤價'].mean()) / df.loc[max(i_latest - 4, 0):i_latest, '收盤價'].mean()) * 100, 6) if i_latest >= 4 else 0.0,
        '20日均線偏離率': round(float((df.at[i_latest, '收盤價'] - df.loc[max(i_latest - 19, 0):i_latest, '收盤價'].mean()) / df.loc[max(i_latest - 19, 0):i_latest, '收盤價'].mean()) * 100, 6) if i_latest >= 19 else 0.0,
        '成交量增幅': round(float(volume_change_pct), 6),
        '外資淨買賣超/成交量': round(float(foreign_net / max(latest_volume, 1)), 6) if latest_volume != 0 else 0.0,
        '融資融券比率': round(float(margin_bal / max(short_bal, 1)), 6) if short_bal != 0 else 0.0
    }

    # 3. 呼叫大腦進行最終預測
    ml_pass, ml_prob = filter_with_ml(latest_features)
    # ==========================================

    past_trades = []
    for i_hist in range(4, len(df)-1):
        hist_trigger, _, _, _, _ = check_conditions(df, i_hist, **params)
        
        if hist_trigger:
            buy_price = df.at[i_hist, '收盤價']
            prev_price = buy_price
            consecutive_drops = 0
            exit_k = 0
            weekly_records = [] 

            for k in range(1, len(df) - i_hist):
                curr_price = df.at[i_hist+k, '收盤價']
                week_ret = ((curr_price - prev_price) / prev_price) * 100
                weekly_records.append(f"W{k}: {week_ret:+.1f}%")

                if week_ret < 0:
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0

                prev_price = curr_price
                exit_k = k
                if consecutive_drops >= 2: break

            cum_ret = ((prev_price - buy_price) / buy_price) * 100
            past_trades.append({
                '進場日': df.at[i_hist, '資料日期'].strftime('%Y-%m-%d') if isinstance(df.at[i_hist, '資料日期'], pd.Timestamp) else df.at[i_hist, '資料日期'],
                '持股週數': exit_k,
                '累積報酬': cum_ret,
                '歷程': ", ".join(weekly_records),
                '開局秒出場': consecutive_drops >= 2 and exit_k == 2 
            })

    # 預設建議
    suggestion = '🎯 建議進場'
    hist_summary = "無歷史前例"
    hist_details_str = "無"

    # 第一關：ML 大腦審查
    if not ml_pass:
        suggestion = '❌ ML大腦退件'

    # 第二關：歷史前例審查
    if past_trades:
        avg_ret = np.mean([t['累積報酬'] for t in past_trades])
        bad_starts = sum(1 for t in past_trades if t['開局秒出場'] and t['累積報酬'] < 0)
        
        hist_summary = f"發生 {len(past_trades)} 次, 平均 {avg_ret:+.2f}%"
        
        # 如果歷史回測不佳，覆蓋原有建議
        if avg_ret < 0 or (bad_starts / len(past_trades) >= 0.5):
            suggestion = '❌ 歷史回測不佳'
            
        details_list = []
        for pt in past_trades:
            status = "⚠️ 連跌兩週停損" if pt['開局秒出場'] else "✅波段結算"
            details_list.append(f"[{pt['進場日']}] 總計 {pt['累積報酬']:>+5.1f}% | 軌跡: {pt['歷程']} ({status})")
        hist_details_str = "\n".join(details_list)

    result_dict = {
        '代號': stock_id,
        '發布日': df.at[i_latest, '資料日期'].strftime('%Y-%m-%d') if isinstance(df.at[i_latest, '資料日期'], pd.Timestamp) else df.at[i_latest, '資料日期'],
        f'大戶({actual_win}週)': round(float(corr), 3),
        f'散戶({actual_win}週)': round(float(retail_corr), 3),
        f'均張({actual_win}週)': round(float(avg_corr), 3),
        '收盤價': df.at[i_latest, '收盤價'],
        'ML預測': f"{'✅' if ml_pass else '❌'} ({ml_prob:+.1f}%)",
        '相似型態勝率': hist_summary,
        '歷史走勢明細': hist_details_str, 
        '建議': suggestion
    }

    return result_dict, past_trades

# ==========================================
# 預測總司令 (支援動態傳入參數)
# ==========================================
def get_next_week_recommendations(target_list, params=None):
    if params is None: params = {}
    recommendations = []
    total = len(target_list)

    for i, sid in enumerate(target_list):
        print(f"🔎 掃描預測 [{i + 1}/{total}] {sid}...", end="\r", flush=True) 
        
        df = crawler.get_individual_stock_data(sid)
        if df is None or df.empty:
            continue

        res, past_trades = scan_latest_and_history(df, params)
        
        if res:
            recommendations.append(res)
            print(f"🔎 掃描預測 [{i + 1}/{total}] {sid}... 🔔 發現預測訊號！{' ' * 20}")

    if recommendations:
        return pd.DataFrame(recommendations).sort_values('代號')
    else:
        return pd.DataFrame()


# ==========================================
# 輔助函式：計算中英文混合字串的視覺寬度
# ==========================================
def get_display_width(text):
    """精準計算終端機上的字元寬度 (全形佔2格，半形佔1格)"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)
