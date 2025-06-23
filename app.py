import streamlit as st
from scipy.stats import poisson
import numpy as np # 数値計算の安定性向上のため追加

# --- 定義データ ---
# ジャグラー各機種の設定ごとの確率情報
# 数値は全て1/X.Xの場合のX.X、または%の場合の小数（例: 0.27%は0.0027）
# 画像「ジャグラー設定さ.jpg」から抽出したデータ
JUGGLER_GAME_DATA = {
    "マイジャグラーV": {
        "BB確率": {1: 273.1, 2: 270.8, 3: 266.4, 4: 262.4, 5: 254.0, 6: 246.0},
        "RB確率": {1: 409.6, 2: 385.5, 3: 336.1, 4: 304.0, 5: 286.8, 6: 270.8},
        "ボーナス合算確率": {1: 163.8, 2: 159.1, 3: 148.6, 4: 140.4, 5: 132.8, 6: 125.5},
        "ブドウ確率": {1: 5.90, 2: 5.85, 3: 5.80, 4: 5.78, 5: 5.76, 6: 5.76},
        "単独BB確率": {1: 420.10, 2: 381.90, 3: 363.82, 4: 356.20, 5: 356.20, 6: 356.20},
        "単独RB確率": {1: 655.36, 2: 595.78, 3: 404.54, 4: 376.64, 5: 348.60, 6: 341.33},
        "チェリー重複BB確率": {1: 1365.33, 2: 1365.33, 3: 496.48, 4: 404.54, 5: 390.10, 6: 327.68},
        "チェリー重複RB確率": {1: 1092.27, 2: 1092.27, 3: 1365.33, 4: 1365.33, 5: 1337.47, 6: 1299.93},
    },
    "ファンキージャグラー2": {
        "BB確率": {1: 266.4, 2: 259.0, 3: 256.0, 4: 249.2, 5: 240.1, 6: 219.9},
        "RB確率": {1: 439.8, 2: 407.1, 3: 366.1, 4: 322.8, 5: 299.3, 6: 262.1},
        "ボーナス合算確率": {1: 165.9, 2: 158.3, 3: 150.7, 4: 140.6, 5: 133.2, 6: 119.6},
        "ブドウ確率": {1: 5.94, 2: 5.93, 3: 5.88, 4: 5.83, 5: 5.75, 6: 5.66},
        "単独BB確率": {1: 404.54, 2: 397.19, 3: 394.80, 4: 383.25, 5: 374.49, 6: 334.37},
        "単独RB確率": {1: 630.15, 2: 585.14, 3: 512.00, 4: 448.88, 5: 404.54, 6: 352.34},
        "チェリー重複BB確率": {1: 1424.7, 2: 1365.3, 3: 1365.3, 4: 1365.3, 5: 1285.0, 6: 1260.3},
        "チェリー重複RB確率": {1: 1456.36, 2: 1337.47, 3: 1285.02, 4: 1149.75, 5: 1149.75, 6: 1024.00},
    },
    "アイムジャグラーEX": {
        "BB確率": {1: 273.1, 2: 269.7, 3: 266.7, 4: 259.0, 5: 255.0, 6: 255.0},
        "RB確率": {1: 439.8, 2: 399.6, 3: 331.0, 4: 315.1, 5: 255.0, 6: 255.0},
        "ボーナス合算確率": {1: 168.5, 2: 161.0, 3: 148.6, 4: 142.2, 5: 128.5, 6: 127.5},
        "ブドウ確率": {1: 6.04, 2: 6.02, 3: 6.02, 4: 5.92, 5: 5.84, 6: 5.78},
        "単独BB確率": {1: 431.16, 2: 422.81, 3: 422.81, 4: 417.43, 5: 417.43, 6: 407.06},
        "単独RB確率": {1: 630.15, 2: 574.88, 3: 474.90, 4: 448.88, 5: 364.09, 6: 300.69},
        "チェリー重複BB確率": {1: 1129.93, 2: 1129.93, 3: 1092.27, 4: 1092.27, 5: 1092.27, 6: 1092.27},
        "チェリー重複RB確率": {1: 1456.36, 2: 1310.72, 3: 1092.27, 4: 1057.03, 5: 851.12, 6: 851.12},
    },
    "ゴーゴージャグラー3": {
        "BB確率": {1: 259.0, 2: 258.0, 3: 257.0, 4: 254.0, 5: 247.3, 6: 234.9},
        "RB確率": {1: 354.2, 2: 332.7, 3: 306.2, 4: 268.6, 5: 247.3, 6: 234.9},
        "ボーナス合算確率": {1: 149.6, 2: 145.3, 3: 139.7, 4: 130.5, 5: 123.7, 6: 117.4},
        "ブドウ確率": {1: 6.25, 2: 6.20, 3: 6.15, 4: 6.07, 5: 6.00, 6: 5.92},
        "単独BB確率": {1: 394.8, 2: 392.4, 3: 392.4, 4: 387.8, 5: 381.0, 6: 364.1},
        "単独RB確率": {1: 489.10, 2: 452.00, 3: 436.90, 4: 381.00, 5: 339.60, 6: 327.70},
        "チェリー重複BB確率": {1: 1456.4, 2: 1456.4, 3: 1456.4, 4: 1394.4, 5: 1394.4, 6: 1310.7},
        "チェリー重複RB確率": {1: 1424.70, 2: 1310.70, 3: 1170.30, 4: 1110.80, 5: 1024.00, 6: 936.20},
    },
}

# --- カスタムCSS（ジャグラー版 - シンプルなデザイン） ---
CUSTOM_CSS_JUGGLER = """
<style>
/* 全体背景色をStreamlitのテーマに依存させる（デフォルトの白/黒）*/
/* メインコンテンツの背景色や影、角丸のみを設定 */
[data-testid="stAppViewBlockContainer"] {
    background-color: #FFFFFF; /* 明るい背景色 */
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0, 150, 0, 0.2); /* GOGOランプのような緑の光の影 */
    margin: 20px auto;
    max-width: 700px;
}

/* タイトル */
h1, h2, h3 {
    color: #FF1493; /* ジャグラーらしいピンク色 */
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    font-family: 'Arial Black', Gadget, sans-serif;
    text-align: center;
}

/* セクション区切りの破線 */
hr {
    border-top: 2px dashed #FFD700; /* GOGOランプの黄色 */
}

/* ナンバーインプット、セレクトボックスなどの入力フィールド */
.stNumberInput > div > div > input, 
.stSelectbox > div > div > button {
    background-color: #F0F0F0; /* 明るいグレーの背景 */
    color: #333333; /* 暗い文字 */
    border: 2px solid #00AA00; /* 緑の枠線 */
    border-radius: 8px;
    box-shadow: 0 0 8px rgba(0, 255, 0, 0.1); /* 緑の光る影 */
    font-weight: bold;
}

/* ボタン */
.stButton > button {
    background-color: #00AA00; /* GOGOランプの緑色 */
    color: white;
    border: 2px solid #00FF00;
    border-radius: 10px;
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
    font-weight: bold;
    padding: 10px 20px;
    transition: all 0.3s ease;
    display: block;
    margin: 15px auto;
}
.stButton > button:hover {
    background-color: #00FF00; /* ホバーで明るい緑 */
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.7), 0 0 30px rgba(0, 200, 0, 0.5);
    transform: translateY(-2px);
}

/* st.infoのスタイル（ヒントボックス） */
.stAlert {
    background-color: rgba(255, 255, 0, 0.8); /* 黄色の半透明 */
    color: #333333; /* 黒文字 */
    border-left: 5px solid #FFA500; /* オレンジ色の線 */
    border-radius: 8px;
}
</style>
"""


def calculate_likelihood(observed_count, total_count, target_rate_value, is_probability_rate=True):
    """
    実測値と解析値から尤度を計算する。
    target_rate_value: 1/X形式の場合のX、または%形式の小数。
    is_probability_rate: Trueなら確率（%表示の小数）、Falseなら分母（1/XのX）
    """
    if total_count <= 0: # 試行回数がゼロ以下なら計算に影響を与えない
        return 1.0
    
    # 観測回数もゼロなら影響を与えない（データがないのと同じ）
    if observed_count <= 0 and total_count > 0:
        # ただし、解析値が0%なのに観測値が0なら尤度が高い
        if (is_probability_rate and target_rate_value <= 1e-10) or \
           (not is_probability_rate and target_rate_value == float('inf')): # 分母無限大=確率0
           return 1.0 # 観測0で解析値も0なら尤度高い

    # target_rate_value が極端に小さい/大きい場合のNaN/inf回避
    if not isinstance(target_rate_value, (int, float)) or target_rate_value <= 1e-10:
        # if total_count > 0 and observed_count > 0: # 観測があるのに確率が0なら極めて低い
        #     return 1e-10
        # else:
        #     return 1.0 # 観測がないか、確率0でも観測0なら中立
        return 1.0 if observed_count == 0 else 1e-10 # 確率0なら観測0で尤度1、観測>0で尤度極小

    if is_probability_rate: # %形式の確率の場合
        expected_value = total_count * target_rate_value
    else: # 1/X形式の分母の場合
        expected_value = total_count / target_rate_value
    
    # expected_valueが0以下または非有限な値の場合のハンドリング
    if not np.isfinite(expected_value) or expected_value < 1e-10: 
        return 1.0 if observed_count == 0 else 1e-10

    # ポアソン分布のPMF (確率質量関数) を使用して尤度を計算
    # observed_countを整数に丸めることで、floatが原因のエラーを防ぐ
    observed_count_int = int(round(observed_count))

    try:
        likelihood = poisson.pmf(observed_count_int, expected_value)
    except ValueError: # scipy.stats.poisson.pmfが不正な入力でValueErrorを出す場合
        # 例: observed_count_int が極端に大きい/小さい、または expected_value が特殊な値
        return 1e-10 # エラーが発生した場合は非常に低い尤度を返す（その状況は統計的にありえない）
    except OverflowError: # 数値が大きすぎて計算できない場合
        return 1e-10
    
    # 尤度がゼロになることを避けるため、非常に小さい値を下限とする
    return max(likelihood, 1e-10)


# --------------------------------------------------------------------------------------
# Contextual Scoring (周辺状況スコアリング) ロジック
# --------------------------------------------------------------------------------------
def apply_contextual_score(overall_likelihoods, context_inputs):
    """
    周辺状況の入力に基づいて、各設定の尤度を調整する
    context_inputs: Streamlit UIから直接取得した辞書 (get()で安全にアクセス)
    """
    adjusted_likelihoods = overall_likelihoods.copy() # 元の尤度をコピーして調整

    # 1. 今日はイベント日か
    event_day_factor = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    if context_inputs.get('is_event_day_radio', '不明') == "はい":
        event_day_factor = {1: 0.9, 2: 1.0, 3: 1.1, 4: 1.2, 5: 1.5, 6: 2.0} # 高設定ほど期待度UP
    elif context_inputs.get('is_event_day_radio', '不明') == "いいえ":
        event_day_factor = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 6: 0.5} # 高設定ほど期待度DOWN
    
    # 2. 今日はジャグラーに設定が入っていることが期待できる日なのか
    juggler_expect_factor = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    if context_inputs.get('juggler_expect_day_radio', '不明') == "はい":
        juggler_expect_factor = {1: 0.9, 2: 1.0, 3: 1.1, 4: 1.3, 5: 1.8, 6: 2.5} # さらに高設定期待度UP
    elif context_inputs.get('juggler_expect_day_radio', '不明') == "いいえ":
        juggler_expect_factor = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 6: 0.4} # さらに高設定期待度DOWN

    # 3. 末尾イベントなどやっているか & 自分の座っている末尾は期待できるか
    tail_event_factor = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    if context_inputs.get('is_tail_event_radio', '不明') == "はい" and context_inputs.get('is_my_tail_expected_radio', '不明') == "はい":
        tail_event_factor = {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5, 5: 2.0, 6: 3.0} # 末尾合致で高設定に強い影響
    elif context_inputs.get('is_tail_event_radio', '不明') == "はい" and context_inputs.get('is_my_tail_expected_radio', '不明') == "いいえ":
        tail_event_factor = {1: 0.8, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 6: 0.5} # 末尾不一致で高設定にペナルティ
    
    # 4. ジャグラー系の取材は入っているか
    coverage_factor = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    if context_inputs.get('has_juggler_coverage_radio', '不明') == "はい":
        coverage_factor = {1: 0.9, 2: 1.0, 3: 1.1, 4: 1.3, 5: 1.8, 6: 2.5} # 取材で高設定期待度UP
    elif context_inputs.get('has_juggler_coverage_radio', '不明') == "いいえ":
        coverage_factor = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 6: 0.5} # 取材なしで高設定期待度DOWN

    # 5. 店がジャグラーに設定6を過去に使っている可能性はあるか
    store_s6_history_factor = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    if context_inputs.get('store_s6_history_radio', '不明') == "はい":
        store_s6_history_factor = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.2, 5: 1.5, 6: 3.0} # 設定6の可能性を強くする
    elif context_inputs.get('store_s6_history_radio', '不明') == "いいえ":
        store_s6_history_factor = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.8, 5: 0.5, 6: 0.1} # 設定6の可能性を大幅に下げる

    # 全ての要因を掛け合わせて尤度を調整
    for setting in range(1, 7):
        adjusted_likelihoods[setting] *= (
            event_day_factor[setting] *
            juggler_expect_factor[setting] *
            tail_event_factor[setting] *
            coverage_factor[setting] *
            store_s6_history_factor[setting]
        )
        # 0になるのを防ぐ
        adjusted_likelihoods[setting] = max(adjusted_likelihoods[setting], 1e-10)

    return adjusted_likelihoods


def predict_setting(game_type, data_inputs, context_inputs): # context_inputsを追加
    overall_likelihoods = {setting: 1.0 for setting in range(1, 7)} # 各設定の総合尤度を1.0で初期化

    # 選択された機種のデータ
    current_game_data = JUGGLER_GAME_DATA.get(game_type)
    if not current_game_data:
        return "選択された機種のデータが見つかりません。入力を見直してください。"

    # データが一つも入力されていない場合のチェック
    if data_inputs.get('total_game_count', 0) == 0: # 総ゲーム数のみで判断
        return "データが入力されていません。推測を行うには、少なくとも総ゲーム数を入力してください。"

    total_game_count = data_inputs.get('total_game_count', 0) # 総ゲーム数
    
    # --- 確率系の要素の計算 ---
    
    # BB, RB, ボーナス合算確率
    # ボーナス総回数 (BIG+REG) が分母となるもの (単独BB/RB)
    at_first_hit_count = data_inputs.get('at_first_hit_count', 0) # UIからのボーナス総回数
    
    for bonus_type_key in ["BB確率", "RB確率", "ボーナス合算確率"]:
        # UI入力キーを動的に取得
        ui_input_key = ""
        if bonus_type_key == "BB確率": ui_input_key = "bb_count"
        elif bonus_type_key == "RB確率": ui_input_key = "reg_count"
        elif bonus_type_key == "ボーナス合算確率": ui_input_key = "at_first_hit_count"
        
        observed_count = data_inputs.get(ui_input_key, 0)
        if total_game_count > 0 and observed_count >= 0:
            for setting, rate_val in current_game_data.get(bonus_type_key, {}).items(): 
                likelihood = calculate_likelihood(observed_count, total_game_count, rate_val, is_probability_rate=False)
                overall_likelihoods[setting] *= likelihood

    # ブドウ確率
    if total_game_count > 0 and data_inputs.get('budou_count', 0) >= 0:
        for setting, rate_val in current_game_data.get("ブドウ確率", {}).items(): 
            likelihood = calculate_likelihood(data_inputs['budou_count'], total_game_count, rate_val, is_probability_rate=False)
            overall_likelihoods[setting] *= likelihood

    # 単独BB/RB確率 (ボーナス総回数を分母に)
    for bonus_type_key in ["単独BB確率", "単独RB確率"]:
        ui_input_key = ""
        if bonus_type_key == "単独BB確率": ui_input_key = "tandoku_bb_count"
        elif bonus_type_key == "単独RB確率": ui_input_key = "tandoku_rb_count"

        observed_count = data_inputs.get(ui_input_key, 0)
        if at_first_hit_count > 0 and observed_count >= 0: # ボーナス総回数が分母
             for setting, rate_val in current_game_data.get(bonus_type_key, {}).items(): 
                likelihood = calculate_likelihood(observed_count, at_first_hit_count, rate_val, is_probability_rate=False)
                overall_likelihoods[setting] *= likelihood

    # チェリー重複BB/RB確率 (チェリー総回数を分母に)
    cherry_total_count = data_inputs.get('cherry_count', 0)
    if cherry_total_count > 0: # チェリー総回数を分母に
        for bonus_type_key in ["チェリー重複BB確率", "チェリー重複RB確率"]:
            ui_input_key = ""
            if bonus_type_key == "チェリー重複BB確率": ui_input_key = "cherry_choufuku_bb_count"
            elif bonus_type_key == "チェリー重複RB確率": ui_input_key = "cherry_choufuku_rb_count"
            
            observed_count = data_inputs.get(ui_input_key, 0)
            if observed_count >= 0:
                for setting, rate_val in current_game_data.get(bonus_type_key, {}).items(): 
                    likelihood = calculate_likelihood(observed_count, cherry_total_count, rate_val, is_probability_rate=False)
                    overall_likelihoods[setting] *= likelihood
    
    # --- 周辺状況スコアリングを適用 ---
    adjusted_likelihoods = apply_contextual_score(overall_likelihoods, context_inputs)

    # --- 最終結果の処理 ---
    total_overall_likelihood_sum = sum(adjusted_likelihoods.values())
    if total_overall_likelihood_sum == 0: 
        return "データが不足しているか、矛盾しているため、推測が困難です。入力値を見直してください。"

    normalized_probabilities = {s: (p / total_overall_likelihood_sum) * 100 for s, p in adjusted_likelihoods.items()}

    predicted_setting = max(normalized_probabilities, key=normalized_probabilities.get)
    max_prob_value = normalized_probabilities[predicted_setting]

    result_str = f"## ✨ 推測される設定: 設定{predicted_setting} (確率: 約{max_prob_value:.2f}%) ✨\n\n"
    result_str += "--- 各設定の推測確率 ---\n"
    for setting, prob in sorted(normalized_probabilities.items(), key=lambda item: item[1], reverse=True):
        result_str += f"  - 設定{setting}: 約{prob:.2f}%\n"
    
    return result_str


# --- Streamlit UI 部分 ---

st.set_page_config(
    page_title="ジャグラー 設定判別ツール",
    layout="centered",
    initial_sidebar_state="collapsed", # サイドバーはデフォルトで閉じる
    page_icon="🎰" 
)

# カスタムCSSの注入 (背景画像なし、UI要素のスタイリングのみ残す)
st.markdown(CUSTOM_CSS_JUGGLER, unsafe_allow_html=True)


st.title("🎰 ジャグラー 設定判別ツール 🎰")

st.markdown(
    """
    ジャグラーの設定判別に特化したツールです。
    遊技の参考に活用してください！
    ---
    """
)

# --- 機種選択セクション ---
st.header("▼機種選択▼")
game_type_options = list(JUGGLER_GAME_DATA.keys())
selected_game_type = st.selectbox("機種を選択してください", game_type_options, key="selected_game_type")
st.markdown("---")

# --- データ入力セクション ---
st.header("▼データ入力▼")
st.markdown("各判別要素の累計値を入力してください。")

st.subheader(f"1. {selected_game_type} の基本データ 🎯") # 選択機種名を表示
st.markdown(f"選択機種の各項目に累計値を入力してください。")
with st.container(border=True): # コンテナで囲んで視覚的にグループ化
    col1, col2, col3 = st.columns(3)
    with col1:
        total_game_count = st.number_input("総ゲーム数", min_value=0, value=0, help="総回転数を入力します。", key="total_game_count")
        
    with col2:
        bb_count = st.number_input("BIG回数", min_value=0, value=0, key="bb_count")
        reg_count = st.number_input("REG回数", min_value=0, value=0, key="reg_count")
    with col3:
        budou_count = st.number_input("ブドウ回数", min_value=0, value=0, key="budou_count")
        cherry_count = st.number_input("チェリー総回数", min_value=0, value=0, key="cherry_count")
    
    st.markdown("---")
    st.markdown("##### 💡 その他の詳細データ（集計していれば入力）")
    col_koyaku1, col_koyaku2, col_koyaku3 = st.columns(3)
    with col_koyaku1:
        st.markdown("###### チェリー重複ボーナス")
        cherry_choufuku_bb_count = st.number_input("└ 重複BIG回数", min_value=0, value=0, key="cherry_choufuku_bb_count")
        cherry_choufuku_rb_count = st.number_input("└ 重複REG回数", min_value=0, value=0, key="cherry_choufuku_rb_count") 
    with col_koyaku2:
        st.markdown("###### 単独ボーナス")
        tandoku_bb_count = st.number_input("└ 単独BIG回数", min_value=0, value=0, key="tandoku_bb_count")
        tandoku_rb_count = st.number_input("└ 単独REG回数", min_value=0, value=0, key="tandoku_rb_count")
    with col_koyaku3:
        # ボーナス総回数は合算で取得可能だが、念のため表示
        st.markdown("###### ボーナス総回数")
        st.markdown(f"<p style='font-size:1.2em; font-weight:bold;'>{bb_count + reg_count} 回</p>", unsafe_allow_html=True)
    

st.markdown("---")

# --- 周辺状況スコアリング入力 ---
st.header("▼周辺状況入力（設定推測の精度向上に！）▼")
st.markdown("ホールの状況やイベント情報を入力することで、推測結果の精度を高めます。")
st.markdown("※入力しなくても推測可能です。")

with st.container(border=True):
    col_context1, col_context2 = st.columns(2)
    with col_context1:
        is_event_day = st.radio("今日はイベント日ですか？", ["不明", "はい", "いいえ"], key="is_event_day_radio")
        juggler_expect_day = st.radio("今日はジャグラーに設定が入っていることが期待できる日ですか？", ["不明", "はい", "いいえ"], key="juggler_expect_day_radio")
        has_juggler_coverage = st.radio("ジャグラー系の取材は入っていますか？", ["不明", "はい", "いいえ"], key="has_juggler_coverage_radio")
    with col_context2:
        is_tail_event = st.radio("末尾イベントなどやっていますか？", ["不明", "はい", "いいえ"], key="is_tail_event_radio")
        # is_my_tail_expected は is_tail_event が「はい」の場合のみ表示
        if is_tail_event == "はい":
            is_my_tail_expected = st.radio("└ 自分の座っている末尾は期待できますか？", ["不明", "はい", "いいえ"], key="is_my_tail_expected_radio")
        else:
            is_my_tail_expected = "不明" # 末尾イベントでない場合は「不明」として扱い影響させない
            st.markdown("_(末尾イベントでないため無関係)_") # 条件分岐で表示
        store_s6_history = st.radio("店がジャグラーに設定6を過去に使っている可能性は？", ["不明", "はい", "いいえ"], key="store_s6_history_radio")

st.markdown("---")


# --- 推測実行ボタン ---
st.subheader("▼結果表示▼")
st.markdown("全てのデータ入力が終わったら、以下のボタンをクリックしてください。")
result_button_clicked = st.button("✨ 推測結果を表示 ✨", type="primary")

if result_button_clicked:
    # predict_setting関数に渡す入力データを収集
    user_inputs_for_prediction = {
        'total_game_count': total_game_count,
        'at_first_hit_count': bb_count + reg_count, # ボーナス総回数をここで計算して渡す
        'bb_count': bb_count,
        'reg_count': reg_count,
        'budou_count': budou_count,
        'cherry_count': cherry_count,
        'cherry_choufuku_bb_count': cherry_choufuku_bb_count,
        'cherry_choufuku_rb_count': cherry_choufuku_rb_count,
        'tandoku_bb_count': tandoku_bb_count,
        'tandoku_rb_count': tandoku_rb_count,
    }

    context_inputs = {
        'is_event_day': is_event_day,
        'juggler_expect_day': juggler_expect_day,
        'is_tail_event': is_tail_event,
        'is_my_tail_expected': is_my_tail_expected, 
        'has_juggler_coverage': has_juggler_coverage,
        'store_s6_history': store_s6_history,
    }
    
    result_content = predict_setting(selected_game_type, user_inputs_for_prediction, context_inputs)
    st.markdown(result_content)