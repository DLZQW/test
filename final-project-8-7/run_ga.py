import sys, os
root = os.path.abspath(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

import pandas as pd
import numpy as np
import random
import math
import time
from tqdm import tqdm
import crawler
import matplotlib.pyplot as plt
import json
from backtest import backtest_squeeze_strategy, has_any_ad_signal

MEMORY_CACHE = {}
RECORD_FILE = 'best_params.json'

# ==========================================
# 🌟 效能極速優化區：動態攔截股價硬碟讀取 (Monkey Patching)
# ==========================================
PRICE_CACHE = {}  

original_download = crawler.download_stock_price_history

def fast_cached_download(stock_id, force_update=False):
    if stock_id not in PRICE_CACHE:
        PRICE_CACHE[stock_id] = original_download(stock_id, force_update)
    return PRICE_CACHE[stock_id]

crawler.download_stock_price_history = fast_cached_download
# ==========================================

def preload_data(target_list):
    print(f"正在將 {len(target_list)} 檔股票資料載入記憶體中（同步進行股價預載與黑名單過濾）...")
    valid_targets = []
    
    for sid in target_list:
        if sid not in MEMORY_CACHE:
            # 1. 抓取籌碼資料
            df = crawler.get_individual_stock_data(sid)
            if df is None or df.empty:
                continue
            
            # 2. 🌟 關鍵過濾：直接預先呼叫 yfinance 抓價格
            # 只要抓不到（例如 3682、2888 下市或報錯），這檔股票就直接被丟掉，不會進入有效清單！
            price_df = crawler.download_stock_price_history(sid)
            if price_df is None or price_df.empty:
                print(f"  ⚠️ 排除無效或下市標的: {sid}")
                continue
                
            MEMORY_CACHE[sid] = df
            
        valid_targets.append(sid)
        
    print(f"資料載入完成！剔除下市股後，有效考題共 {len(valid_targets)} 檔。開始執行基因演算法...\n")
    
    # 回傳這份「絕對乾淨」的清單給主程式
    return valid_targets

def load_historical_best():
    if os.path.exists(RECORD_FILE):
        try:
            with open(RECORD_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'fitness': -1.0, 'params': {}, 'stats': {}}

def save_new_best(params, fitness, stats):
    data = {'fitness': fitness, 'params': params, 'stats': stats}
    with open(RECORD_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n[系統通知] 發現破紀錄的參數！已更新並儲存至 {RECORD_FILE}")

def _evaluate_params_on_universe(target_list, params, sample_limit=None):
    all_trades = []
    target_iter = target_list if sample_limit is None else target_list[:sample_limit]

    for sid in target_iter:
        df = MEMORY_CACHE.get(sid)
        if df is None or df.empty:
            continue

        if not has_any_ad_signal(df,
                                 continuous_weeks=params['continuous_weeks'],
                                 min_growth=params['min_growth'],
                                 last_week_threshold=params['last_week_threshold'],
                                 pop_decline_threshold=params['pop_decline_threshold']):
            continue

        trades = backtest_squeeze_strategy(df,
                                          continuous_weeks=params['continuous_weeks'],
                                          min_growth=params['min_growth'],
                                          last_week_threshold=params['last_week_threshold'],
                                          pop_decline_threshold=params['pop_decline_threshold'])

        if trades:
            all_trades.extend(trades)

    if not all_trades:
        return {'fitness': 0.0, 'n_signals': 0, 'avg_return': 0.0, 'win_rate': 0.0}

    trades_df = pd.DataFrame(all_trades).dropna(subset=['週報酬%'])
    if trades_df.empty:
        return {'fitness': 0.0, 'n_signals': 0, 'avg_return': 0.0, 'win_rate': 0.0}

    n = len(trades_df)
    avg_ret = trades_df['週報酬%'].mean()
    win_rate = (trades_df['週報酬%'] > 0).mean() * 100

    fitness = (win_rate / 100.0) * avg_ret * math.sqrt(n)
    return {'fitness': fitness, 'n_signals': n, 'avg_return': avg_ret, 'win_rate': win_rate}

def run_genetic_algorithm(target_list, generations=15, population_size=50, sample_limit=None,
                          retain_top=0.3, mutate_chance=0.05, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    # 放寬參數邊界，避免第一代找不到任何訊號
    param_bounds = {
        'continuous_weeks': (3, 8),
        'min_growth': (0.0, 0.8),
        'last_week_threshold': (0.0, 8.0),
        'pop_decline_threshold': (0.1, 5.0)
    }

    def random_individual():
        return {
            'continuous_weeks': random.randint(param_bounds['continuous_weeks'][0], param_bounds['continuous_weeks'][1]),
            'min_growth': round(random.uniform(*param_bounds['min_growth']), 4),
            'last_week_threshold': round(random.uniform(*param_bounds['last_week_threshold']), 3),
            'pop_decline_threshold': round(random.uniform(*param_bounds['pop_decline_threshold']), 3)
        }

    population = [random_individual() for _ in range(population_size)]
    history_best = []
    history_avg = []

    for gen in range(generations):
        gen_start_time = time.time() # 🌟 1. 碼錶按下去：記錄這一代開始的時間

        print(f"GA 世代 {gen+1}/{generations} - 正在評估 {len(population)} 個個體...")

        scored_pop = []
        for individual in tqdm(population, desc=f"第 {gen+1} 代運算中", leave=False, colour='green'):
            stats = _evaluate_params_on_universe(target_list, individual, sample_limit=sample_limit)
            scored_pop.append((individual, stats['fitness'], stats))

        scored_pop.sort(key=lambda x: x[1], reverse=True)
        best_individual, best_fitness, best_stats = scored_pop[0]
        avg_fitness = sum(x[1] for x in scored_pop) / len(scored_pop)
        
        history_best.append(best_fitness)
        history_avg.append(avg_fitness)

        gen_end_time = time.time()  # 🌟 2. 碼錶按停：記錄運算結束的時間
        elapsed_time = gen_end_time - gen_start_time  # 🌟 3. 計算總共花了幾秒

        # 🌟 4. 把耗時塞進原本印出結果的字串最後面
        print(f"  目前最佳: 分數={best_fitness:.4f} 訊號數={best_stats['n_signals']} 平均報酬={best_stats['avg_return']:.3f}% 勝率={best_stats['win_rate']:.1f}% 參數={best_individual} ⏱️ 耗時: {elapsed_time:.1f} 秒")

        retain_length = max(1, int(retain_top * len(scored_pop)))
        next_gen = [ind for ind, _, _ in scored_pop[:retain_length]]

        while len(next_gen) < population_size:
            def tournament():
                competitors = random.sample(scored_pop, min(3, len(scored_pop)))
                competitors.sort(key=lambda x: x[1], reverse=True)
                return competitors[0][0]

            parent1 = tournament()
            parent2 = tournament()
            child = {}
            for k in parent1.keys():
                child[k] = parent1[k] if random.random() < 0.5 else parent2[k]

            if random.random() < mutate_chance:
                key_to_mutate = random.choice(list(param_bounds.keys()))
                if key_to_mutate == 'continuous_weeks':
                    child['continuous_weeks'] = random.randint(param_bounds['continuous_weeks'][0], param_bounds['continuous_weeks'][1])
                else:
                    lo, hi = param_bounds[key_to_mutate]
                    child[key_to_mutate] = round(random.uniform(lo, hi), 4 if key_to_mutate=='min_growth' else 3)

            child['continuous_weeks'] = int(child['continuous_weeks'])
            next_gen.append(child)

        population = next_gen

    final_scores = []
    for individual in population:
        stats = _evaluate_params_on_universe(target_list, individual, sample_limit=sample_limit)
        final_scores.append((individual, stats['fitness'], stats))

    final_scores.sort(key=lambda x: x[1], reverse=True)

    try:
        plt.figure(figsize=(10, 6))
        gen_x = range(1, generations + 1)
        plt.plot(gen_x, history_best, marker='o', label='Best Fitness', color='red', linewidth=2)
        plt.plot(gen_x, history_avg, marker='x', label='Average Fitness', color='blue', linestyle='--')
        plt.title('GA Evolution: Fitness Convergence')
        plt.xlabel('Generations')
        plt.ylabel('Fitness Score')
        plt.xticks(gen_x)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('ga_evolution.png')
        print("\n[繪圖完成] 演化趨勢圖已儲存為 'ga_evolution.png'")
    except Exception as e:
        print(f"\n[繪圖失敗] 系統環境無法繪製圖表: {e}")

    return final_scores

def main():
    historical_record = load_historical_best()
    if historical_record['fitness'] != -1.0:
        print("="*50)
        print(f"目前歷史最佳紀錄:")
        print(f"分數: {historical_record['fitness']:.4f}")
        print(f"統計: {historical_record['stats']}")
        print(f"參數: {historical_record['params']}")
        print("="*50)
    else:
        print("尚未建立歷史紀錄，這是第一次執行！")

    stock_list = crawler.get_stock_ids(crawler.list_url)
    print(f'市場總股票數: {len(stock_list)}')
    
    # --------------------------------------------------
    # 🎯 自訂測試範圍設定區 (取代原本的隨機亂抽)
    # --------------------------------------------------
    start_index = 0   # 👉 你想從第幾檔股票開始？ (例如: 0 代表第1檔，10 代表第11檔)
    test_count = len(stock_list)   # 👉 總共要往後測試幾檔？ 用 len(stock_list) 讓系統自動抓取全部 (1077檔)
    
    # 防呆：確保選取範圍不會超過全市場的股票總數
    if start_index >= len(stock_list): start_index = 0
    end_index = min(start_index + test_count, len(stock_list))
    
    target_list = stock_list[start_index : end_index]
    print(f'本次選取的考題: 從第 {start_index} 檔開始，共 {len(target_list)} 檔 ({target_list[:5]}...)')

    # --------------------------------------------------
    # 呼叫預載函式，並用回傳的「乾淨清單」覆蓋原本的 target_list
    target_list = preload_data(target_list)
    # --------------------------------------------------

    # 接下來 GA 就只會拿著這份 100% 安全的 target_list 去跑，絕對不會卡住！
    results = run_genetic_algorithm(target_list,
                                   generations=15,       
                                   population_size=50,   
                                   sample_limit=None,    
                                   retain_top=0.3,       
                                   mutate_chance=0.05,   
                                   random_seed=None)

    print('\n本次執行的 Top 5 結果:')
    for idx, (ind, fit, stats) in enumerate(results[:5]):
        print(f"Top {idx+1}: fitness={fit:.4f} signals={stats['n_signals']} avg_ret={stats['avg_return']:.3f}% win_rate={stats['win_rate']:.1f}% params={ind}")

    best_ind, best_fit, best_stats = results[0]
    if best_fit > historical_record['fitness']:
        save_new_best(best_ind, best_fit, best_stats)
    else:
        print(f"\n[系統通知] 本次最佳分數 ({best_fit:.4f}) 未能超越歷史紀錄 ({historical_record['fitness']:.4f})。")

if __name__ == '__main__':
    main()
