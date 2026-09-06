# -*- coding: utf-8 -*-
"""
数据分析图表生成脚本
基于真实数据集，使用 matplotlib + seaborn 生成一套分析报告图表，
保存至 output/image/，并输出关键统计汇总 JSON 与《数据分析报告.md》骨架。
运行：python tools/generate_analysis.py
"""

import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 加入项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.config import DATASET_FILE, RANDOM_SEED, IMAGE_DIR, REPORT_DIR
from backend.core.data_clean import clean_dataset
from backend.core.dim_reduce import perform_pca, perform_factor_analysis
from backend.core.model_train import ModelTrainer
from sklearn.metrics import roc_curve, auc

# ---------------- 全局绘图配置 ----------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='Microsoft YaHei')

# 项目配色（深蓝科技风）
C_BLUE   = '#0a6cff'
C_CYAN   = '#00d4ff'
C_ORANGE = '#fa8c16'
C_GREEN  = '#52c41a'
C_RED    = '#f5222d'
C_PURPLE = '#722ed1'
C_GRAY   = '#8c8c8c'
PALETTE  = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED, C_CYAN]

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def savefig(fig, name, dpi=150):
    path = os.path.join(IMAGE_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {name}')
    return path


# ---------------- 数据加载 ----------------
print('[1/6] 加载数据...')
raw = pd.read_csv(DATASET_FILE, encoding='utf-8-sig')
biz = raw.dropna().copy()  # 业务图表用完整行

# 用于模型/降维的标准化数据（真实清洗管线）
clean_result = clean_dataset()
cleaned = clean_result['df_cleaned']
feature_cols = [c for c in cleaned.columns if c not in ['user_id', 'repurchase_label']]
X = cleaned[feature_cols].values
y = cleaned['repurchase_label'].values

n = len(raw)
repurchase = int(raw['repurchase_label'].sum())
churn = n - repurchase
rate = repurchase / n
print(f'  总样本 {n}，复购 {repurchase} ({rate:.1%})，流失 {churn}')


# ============================================================
#  一、样本结构与复购标签
# ============================================================
print('[2/6] 样本结构与标签分布...')
fig, ax = plt.subplots(figsize=(7, 5))
vc = raw['repurchase_label'].value_counts()
counts = [vc.get(0, 0), vc.get(1, 0)]
bars = ax.bar(['不复购 (0)', '复购 (1)'], counts,
              color=[C_ORANGE, C_GREEN], width=0.45, edgecolor='white')
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
            f'{c}\n({c / n * 100:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('用户数')
ax.set_title(f'电商用户复购标签分布（样本总量 {n}）', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(counts) * 1.18)
savefig(fig, '01_复购标签分布.png')


def group_rate_fig(df, group_col, names, title, fname):
    """分组：用户数 / 复购用户 / 复购率 复合图"""
    g = df.groupby(group_col).agg(total=('user_id', 'count'),
                                  repurchase=('repurchase_label', 'sum'))
    g['rate'] = g['repurchase'] / g['total']
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - 0.2, g['total'], width=0.4, label='总用户', color=C_GRAY, alpha=0.85)
    b2 = ax.bar(x + 0.2, g['repurchase'], width=0.4, label='复购用户', color=C_GREEN)
    ax2 = ax.twinx()
    ax2.plot(x, g['rate'] * 100, color=C_ORANGE, marker='o', linewidth=2, label='复购率')
    ax2.set_ylabel('复购率 (%)', color=C_ORANGE)
    ax2.set_ylim(0, max(g['rate'] * 100) * 1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(int(i), int(i)) for i in g.index], fontsize=11)
    ax.set_ylabel('用户数')
    ax.set_title(title, fontsize=14, fontweight='bold')
    for i, r in zip(x, g['rate']):
        ax2.annotate(f'{r * 100:.1f}%', (i, g["rate"][i] * 100),
                     textcoords='offset points', xytext=(0, 6), ha='center',
                     fontsize=10, color=C_ORANGE, fontweight='bold')
    # 图例合并
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc='upper right', framealpha=0.9)
    savefig(fig, fname)


MEMBER_NAMES = {0: '普通', 1: '银卡', 2: '金卡', 3: '钻石'}
CITY_NAMES = {0: '五线', 1: '四线', 2: '三线', 3: '二线', 4: '一线'}
DEVICE_NAMES = {0: '安卓', 1: '苹果', 2: 'PC', 3: '小程序'}

group_rate_fig(biz, 'member_level', MEMBER_NAMES,
               '不同会员等级的复购情况', '02_会员等级复购率.png')
group_rate_fig(biz, 'city_tier', CITY_NAMES,
               '不同城市层级的复购情况', '03_城市层级复购率.png')
group_rate_fig(biz, 'device_type', DEVICE_NAMES,
               '不同设备端的复购情况', '04_设备类型复购率.png')


# VIP 对比
fig, ax = plt.subplots(figsize=(7, 5))
vip = biz.groupby('is_vip').apply(lambda d: pd.Series({
    'users': len(d), 'rate': d['repurchase_label'].mean()}))
vip = vip.reset_index()
x = np.arange(2)
bars = ax.bar(x, vip['rate'] * 100, width=0.45,
              color=[C_GRAY, C_CYAN], edgecolor='white')
for b, (_, r) in zip(bars, vip.iterrows()):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
            f'{r["rate"] * 100:.1f}%\n(n={int(r["users"])})', ha='center',
            fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['非付费用户', '付费会员 (VIP)'])
ax.set_ylabel('复购率 (%)')
ax.set_ylim(0, 80)
ax.set_title('付费会员与非会员的复购率对比', fontsize=14, fontweight='bold')
savefig(fig, '05_VIP复购率对比.png')


# ============================================================
#  二、消费与行为分布
# ============================================================
print('[3/6] 消费与行为分布...')

# 消费金额箱线图（按复购标签分组，对数刻度）
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(x='repurchase_label', y='total_consume', data=biz,
            palette=[C_ORANGE, C_GREEN], width=0.5, ax=ax)
ax.set_yscale('log')
ax.set_xticklabels(['不复购', '复购'])
ax.set_xlabel('是否复购')
ax.set_ylabel('累计消费金额（对数刻度）')
ax.set_title('复购与非复购用户累计消费金额分布', fontsize=14, fontweight='bold')
savefig(fig, '06_消费金额箱线图.png')

# 消费金额直方图
fig, ax = plt.subplots(figsize=(9, 5))
for lab, col in [(0, C_ORANGE), (1, C_GREEN)]:
    d = biz[biz['repurchase_label'] == lab]['total_consume']
    ax.hist(d, bins=40, alpha=0.55, color=col, label=f'{"不复购" if lab == 0 else "复购"} (n={len(d)})')
ax.set_xscale('log')
ax.set_xlabel('累计消费金额（对数刻度）')
ax.set_ylabel('用户数')
ax.set_title('累计消费金额分布（按复购标签）', fontsize=14, fontweight='bold')
ax.legend()
savefig(fig, '07_消费金额分布直方图.png')

# 复购 vs 流失均消对比
avg_r = biz[biz['repurchase_label'] == 1]['total_consume'].mean()
avg_c = biz[biz['repurchase_label'] == 0]['total_consume'].mean()
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(['流失用户', '复购用户'], [avg_c, avg_r],
              color=[C_ORANGE, C_GREEN], width=0.45, edgecolor='white')
for b, v in zip(bars, [avg_c, avg_r]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
            f'¥{v:,.0f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('平均消费金额（元）')
ax.set_title(f'复购用户平均消费比流失用户高 {(avg_r / avg_c - 1) * 100:.0f}%',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(avg_c, avg_r) * 1.18)
savefig(fig, '08_复购与流失用户均消对比.png')

# 静默天数分布
fig, ax = plt.subplots(figsize=(9, 5))
for lab, col in [(0, C_ORANGE), (1, C_GREEN)]:
    d = biz[biz['repurchase_label'] == lab]['silence_days']
    ax.hist(d, bins=30, alpha=0.6, color=col,
            label=f'{"不复购" if lab == 0 else "复购"} (中位 {d.median():.0f}天)')
ax.axvline(30, color=C_RED, linestyle='--', linewidth=1.5, label='静默阈值 30 天')
ax.set_xlabel('最近一次下单距今天数（静默天数）')
ax.set_ylabel('用户数')
ax.set_title('静默天数分布（按复购标签）', fontsize=14, fontweight='bold')
ax.legend()
savefig(fig, '09_静默天数分布.png')


# ============================================================
#  三、特征相关性
# ============================================================
print('[4/6] 特征相关性热力图...')
num_cols = [c for c in biz.columns
            if c not in ['user_id', 'repurchase_label'] and pd.api.types.is_numeric_dtype(biz[c])]
corr = biz[num_cols].corr()
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=False,
            square=True, linewidths=0.3, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('36 维特征 Pearson 相关性热力图', fontsize=15, fontweight='bold')
ax.tick_params(axis='x', rotation=90, labelsize=9)
ax.tick_params(axis='y', labelsize=9)
savefig(fig, '10_特征相关性热力图.png')


# ============================================================
#  四、降维分析：PCA + 因子分析
# ============================================================
print('[5/6] 降维分析（PCA / 因子分析）...')

pca_res = perform_pca(X, random_state=RANDOM_SEED)
evr = pca_res['explained_variance_ratio']
cum = pca_res['cumulative_variance_ratio']
n_pc = pca_res['n_components']

# 碎石图
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(1, len(evr) + 1)
ax.bar(x, evr, alpha=0.75, color=C_BLUE, label='个体方差贡献率')
ax.plot(x, cum, marker='o', color=C_ORANGE, linewidth=2, label='累计贡献率')
ax.axhline(0.95, color=C_GREEN, linestyle=':', alpha=0.8, label='95% 阈值')
ax.axvline(n_pc, color=C_RED, linestyle='--', alpha=0.7,
           label=f'选择 {n_pc} 个主成分')
ax.set_xlabel('主成分序号')
ax.set_ylabel('方差贡献率')
ax.set_title(f'PCA 碎石图（累计方差 {cum[-1] * 100:.1f}%）',
             fontsize=14, fontweight='bold')
ax.legend()
savefig(fig, '11_PCA碎石图.png')

# 因子载荷气泡图（气泡大小 = 公因子方差，颜色 = 特征分组）
fa_res = perform_factor_analysis(X, n_factors=10, random_state=RANDOM_SEED)
loadings = fa_res['loadings']
communality = fa_res['communality']

# 特征分组（四大类）
FEATURE_GROUPS = {
    '用户画像': ['user_age', 'user_gender', 'member_level', 'register_days', 'device_type', 'city_tier', 'is_vip'],
    '消费频次': ['total_order_num', 'month_order_avg', 'week_visit_cnt', 'cart_add_cnt', 'collect_goods_num',
                'return_order_num', 'refund_times', 'cross_shop_order', 'promo_order_ratio', 'silence_days'],
    '消费金额': ['total_consume', 'avg_order_price', 'max_single_order', 'min_single_order', 'discount_save_money',
                'coupon_use_cnt', 'redbag_income', 'high_price_goods_ratio', 'low_price_goods_ratio', 'logistics_extra_cost'],
    '交互行为': ['good_comment_cnt', 'bad_comment_cnt', 'live_watch_duration', 'short_video_click',
                'consult_service_cnt', 'share_product_cnt', 'popup_click_rate', 'search_word_cnt'],
}
GROUP_COLORS = {
    '用户画像': C_BLUE,
    '消费频次': C_CYAN,
    '消费金额': C_ORANGE,
    '交互行为': C_PURPLE,
}
feat_group = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

# 气泡大小：公因子方差（共同度）越大说明该特征被公因子解释得越充分
cmin, cmax = communality.min(), communality.max()
sizes = 60 + 900 * (communality - cmin) / (cmax - cmin + 1e-9)
colors = [GROUP_COLORS.get(feat_group.get(name, '其他'), C_GRAY) for name in feature_cols]

fig, ax = plt.subplots(figsize=(11, 8.5))
ax.scatter(loadings[:, 0], loadings[:, 1], s=sizes, c=colors, alpha=0.72,
           edgecolors='white', linewidths=1.2, zorder=3)
ax.axhline(0, color=C_GRAY, linestyle='--', alpha=0.5)
ax.axvline(0, color=C_GRAY, linestyle='--', alpha=0.5)
ax.set_xlabel('因子 1 载荷（金额 / 消费强度维度）')
ax.set_ylabel('因子 2 载荷（活跃 / 交互维度）')
ax.set_title('因子载荷气泡图（颜色 = 特征分组，气泡大小 = 公因子方差）',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 图例一：特征分组
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=9, label=g) for g, c in GROUP_COLORS.items()]
leg1 = ax.legend(handles=handles, loc='upper left', title='特征分组', framealpha=0.95)
ax.add_artist(leg1)
# 图例二：气泡大小
size_marks = [plt.scatter([], [], s=s, color=C_GRAY, alpha=0.5)
              for s in [60, 400, 900]]
leg2 = ax.legend(size_marks, ['共同度低', '中等', '共同度高'],
                 loc='lower right', title='气泡大小', framealpha=0.95)
savefig(fig, '12_因子载荷图.png')


# ============================================================
#  五、建模与评估
# ============================================================
print('[6/6] 建模评估（SVM / 随机森林）...')
trainer = ModelTrainer(random_state=RANDOM_SEED)
trainer.feature_names = feature_cols
trainer.split_data(X, y)
trainer.train_svm()
trainer.train_random_forest()

svm_m = trainer.svm_metrics
rf_m = trainer.rf_metrics

# ROC 曲线
fig, ax = plt.subplots(figsize=(8, 6))
for model, name, col in [(trainer.svm_model, 'SVM', C_BLUE),
                         (trainer.rf_model, '随机森林', C_GREEN)]:
    prob = model.predict_proba(trainer.X_test)[:, 1]
    fpr, tpr, _ = roc_curve(trainer.y_test, prob)
    a = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, linewidth=2.2,
            label=f'{name} (AUC = {a:.3f})')
ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle='--', linewidth=1,
        label='随机猜想 (AUC = 0.5)')
ax.set_xlabel('假阳性率 (FPR)')
ax.set_ylabel('真阳性率 (TPR)')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_title('SVM 与 随机森林 ROC 曲线对比', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
savefig(fig, '13_ROC曲线对比.png')

# 特征重要性
imp_df = trainer.get_feature_importance(top_n=10)
fig, ax = plt.subplots(figsize=(8, 6))
imp_sorted = imp_df.sort_values('importance')
bars = ax.barh(imp_sorted['feature'], imp_sorted['importance'],
               color=C_BLUE, alpha=0.85)
for bar, val in zip(bars, imp_sorted['importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('特征重要性')
ax.set_title('随机森林 Top10 特征重要性', fontsize=14, fontweight='bold')
savefig(fig, '14_特征重要性Top10.png')

# 指标对比
metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metric_names = ['准确率', '精确率', '召回率', 'F1', 'AUC']
svm_vals = [svm_m.get(m, 0) for m in metrics_list]
rf_vals = [rf_m.get(m, 0) for m in metrics_list]
x = np.arange(len(metrics_list))
fig, ax = plt.subplots(figsize=(8.5, 5.5))
w = 0.35
b1 = ax.bar(x - w / 2, svm_vals, w, label='SVM', color=C_BLUE)
b2 = ax.bar(x + w / 2, rf_vals, w, label='随机森林', color=C_GREEN)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylabel('得分')
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title('SVM 与 随机森林评估指标对比', fontsize=14, fontweight='bold')
savefig(fig, '15_模型指标对比.png')


# ============================================================
#  汇总统计 JSON
# ============================================================
summary = {
    'total_users': n,
    'repurchase_users': repurchase,
    'churn_users': churn,
    'repurchase_rate': round(rate, 4),
    'avg_consume': round(biz['total_consume'].mean(), 2),
    'median_consume': round(biz['total_consume'].median(), 2),
    'avg_consume_repurchase': round(avg_r, 2),
    'avg_consume_non_repurchase': round(avg_c, 2),
    'avg_consume_gap_pct': round((avg_r / avg_c - 1) * 100, 1),
    'coupon_usage_rate': round(float((biz['coupon_use_cnt'] > 0).mean()), 4),
    'silence_users': int((biz['silence_days'] > 30).sum()),
    'silence_rate': round(float((biz['silence_days'] > 30).mean()), 4),
    'median_silence_days': int(biz['silence_days'].median()),
    'pca_components': int(n_pc),
    'pca_cumulative_variance': round(float(cum[-1]), 4),
    'factor_components': int(fa_res['n_factors']),
    'svm_metrics': {k: round(float(v), 4) for k, v in svm_m.items() if k != 'confusion_matrix'},
    'rf_metrics': {k: round(float(v), 4) for k, v in rf_m.items() if k != 'confusion_matrix'},
    'top10_features': imp_df.head(10)[['feature', 'importance']].round(4).to_dict('records'),
}
summary_path = os.path.join(REPORT_DIR, 'analysis_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f'\n统计汇总已保存：{summary_path}')
print('全部图表已生成至 output/image/')
