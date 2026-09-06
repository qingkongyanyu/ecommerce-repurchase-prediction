import numpy as np
import pandas as pd
import random
from scipy.stats import lognorm
import os

# ===================== 基础配置 =====================
np.random.seed(42)
random.seed(42)
SAMPLE_NUM = 8500  # 样本总量
MISS_RATIO = 0.008  # 缺失值比例
SAVE_DIR = "dataset"

# 创建存储文件夹
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ===================== 1. 用户基础画像 =====================
user_id = np.arange(1, SAMPLE_NUM + 1)
user_age = np.random.randint(16, 66, size=SAMPLE_NUM)
user_gender = np.random.binomial(1, 0.48, size=SAMPLE_NUM)
member_level = np.random.choice([0, 1, 2, 3], size=SAMPLE_NUM, p=[0.55, 0.25, 0.15, 0.05])
register_days = np.random.randint(1, 1801, size=SAMPLE_NUM)
device_type = np.random.choice([0, 1, 2, 3], size=SAMPLE_NUM, p=[0.42, 0.38, 0.12, 0.08])
city_tier = np.random.choice([0, 1, 2, 3, 4], size=SAMPLE_NUM, p=[0.12, 0.22, 0.30, 0.24, 0.12])
is_vip = np.random.binomial(1, 0.18, size=SAMPLE_NUM)

# ===================== 2. 消费频次特征 =====================
total_order_num = lognorm.rvs(s=1.2, scale=8, size=SAMPLE_NUM).astype(int)
month_order_avg = total_order_num / (register_days / 30)
week_visit_cnt = lognorm.rvs(s=1.0, scale=5, size=SAMPLE_NUM).astype(int)
cart_add_cnt = lognorm.rvs(s=1.3, scale=12, size=SAMPLE_NUM).astype(int)
collect_goods_num = lognorm.rvs(s=1.1, scale=6, size=SAMPLE_NUM).astype(int)
return_order_num = np.random.poisson(0.8, size=SAMPLE_NUM)
refund_times = np.random.poisson(0.3, size=SAMPLE_NUM)
cross_shop_order = lognorm.rvs(s=0.9, scale=4, size=SAMPLE_NUM).astype(int)
promo_order_ratio = np.clip(np.random.normal(0.45, 0.22, size=SAMPLE_NUM), 0, 1)
silence_days = np.random.randint(0, 366, size=SAMPLE_NUM)

# ===================== 3. 金额消费价值特征 =====================
total_consume = lognorm.rvs(s=1.6, scale=150, size=SAMPLE_NUM)
avg_order_price = total_consume / (total_order_num + 1e-6)
max_single_order = lognorm.rvs(s=1.4, scale=200, size=SAMPLE_NUM)
min_single_order = lognorm.rvs(s=0.8, scale=15, size=SAMPLE_NUM)
discount_save_money = lognorm.rvs(s=1.3, scale=40, size=SAMPLE_NUM)
coupon_use_cnt = lognorm.rvs(s=1.0, scale=7, size=SAMPLE_NUM).astype(int)
redbag_income = lognorm.rvs(s=1.1, scale=25, size=SAMPLE_NUM)
high_price_goods_ratio = np.clip(np.random.normal(0.3, 0.25, size=SAMPLE_NUM), 0, 1)
low_price_goods_ratio = np.clip(np.random.normal(0.5, 0.2, size=SAMPLE_NUM), 0, 1)
logistics_extra_cost = lognorm.rvs(s=0.9, scale=12, size=SAMPLE_NUM)

# ===================== 4. 交互行为特征 =====================
good_comment_cnt = lognorm.rvs(s=1.1, scale=4, size=SAMPLE_NUM).astype(int)
bad_comment_cnt = np.random.poisson(0.4, size=SAMPLE_NUM)
live_watch_duration = lognorm.rvs(s=1.5, scale=60, size=SAMPLE_NUM)
short_video_click = lognorm.rvs(s=1.2, scale=10, size=SAMPLE_NUM).astype(int)
consult_service_cnt = np.random.poisson(0.6, size=SAMPLE_NUM)
share_product_cnt = lognorm.rvs(s=0.9, scale=3, size=SAMPLE_NUM).astype(int)
popup_click_rate = np.clip(np.random.normal(0.25, 0.18, size=SAMPLE_NUM), 0, 1)
search_word_cnt = lognorm.rvs(s=1.3, scale=15, size=SAMPLE_NUM).astype(int)

# ===================== 构建复购标签（业务逻辑关联） =====================
# 构建复购得分，正向指标越高、静默/退货越低，越容易复购
score = (
    total_order_num * 0.12 +
    total_consume * 0.0008 +
    cart_add_cnt * 0.06 +
    coupon_use_cnt * 0.08 +
    good_comment_cnt * 0.05 +
    live_watch_duration * 0.001 -
    silence_days * 0.02 -
    return_order_num * 0.15 -
    refund_times * 0.2
)
# 概率映射，生成0/1标签
prob = 1 / (1 + np.exp(-(score - np.median(score)) / np.std(score)))
repurchase_label = np.random.binomial(1, prob, size=SAMPLE_NUM)

# ===================== 整合所有字段 =====================
data_dict = {
    # 用户画像8列
    "user_id": user_id,
    "user_age": user_age,
    "user_gender": user_gender,
    "member_level": member_level,
    "register_days": register_days,
    "device_type": device_type,
    "city_tier": city_tier,
    "is_vip": is_vip,
    # 消费频次10列
    "total_order_num": total_order_num,
    "month_order_avg": month_order_avg,
    "week_visit_cnt": week_visit_cnt,
    "cart_add_cnt": cart_add_cnt,
    "collect_goods_num": collect_goods_num,
    "return_order_num": return_order_num,
    "refund_times": refund_times,
    "cross_shop_order": cross_shop_order,
    "promo_order_ratio": promo_order_ratio,
    "silence_days": silence_days,
    # 消费金额10列
    "total_consume": total_consume,
    "avg_order_price": avg_order_price,
    "max_single_order": max_single_order,
    "min_single_order": min_single_order,
    "discount_save_money": discount_save_money,
    "coupon_use_cnt": coupon_use_cnt,
    "redbag_income": redbag_income,
    "high_price_goods_ratio": high_price_goods_ratio,
    "low_price_goods_ratio": low_price_goods_ratio,
    "logistics_extra_cost": logistics_extra_cost,
    # 交互行为8列
    "good_comment_cnt": good_comment_cnt,
    "bad_comment_cnt": bad_comment_cnt,
    "live_watch_duration": live_watch_duration,
    "short_video_click": short_video_click,
    "consult_service_cnt": consult_service_cnt,
    "share_product_cnt": share_product_cnt,
    "popup_click_rate": popup_click_rate,
    "search_word_cnt": search_word_cnt,
    # 预测标签
    "repurchase_label": repurchase_label
}
df = pd.DataFrame(data_dict)

# ===================== 随机添加缺失值 =====================
feature_cols = df.columns[1:-1]  # 除user_id、标签外所有特征
for col in feature_cols:
    mask = np.random.choice([True, False], size=SAMPLE_NUM, p=[MISS_RATIO, 1-MISS_RATIO])
    df.loc[mask, col] = np.nan

# ===================== 保存数据集 =====================
csv_path = os.path.join(SAVE_DIR, "ecommerce_repurchase_8500.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"数据集生成完成！共{df.shape[0]}行 {df.shape[1]}列")
print(f"文件保存路径：{csv_path}")
print("前5行预览：")
print(df.head())
print("\n数据基本统计信息：")
print(df.describe())
print("\n缺失值统计：")
print(df.isnull().sum())
print("\n复购标签分布：")
print(df["repurchase_label"].value_counts())