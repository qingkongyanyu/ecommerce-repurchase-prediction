<template>
  <div class="screen-dash">
    <!-- 背景装饰 -->
    <div class="bg-decoration">
      <div class="bg-grid" />
      <div class="bg-glow-1" />
      <div class="bg-glow-2" />
      <div class="bg-aurora" />
    </div>

    <!-- 顶部标题栏 -->
    <header class="header">
      <div class="header-left">
        <div class="logo-icon">
          <el-icon><DataAnalysis /></el-icon>
        </div>
        <div class="header-titles">
          <h1 class="title">电商用户复购预测可视化平台</h1>
          <span class="subtitle">PCA × 因子分析融合降维 · SVM × 随机森林对比建模</span>
        </div>
      </div>
      <div class="header-right">
        <span class="time"><i class="time-dot" />{{ currentTime }}</span>
        <el-tag :type="backendStatus ? 'success' : 'danger'" size="small" effect="dark">
          {{ backendStatus ? '后端已连接' : '后端离线' }}
        </el-tag>
        <span class="sample-count">样本总量 {{ globalStats.total_samples || 8500 }}</span>
      </div>
    </header>

    <!-- 主体内容 -->
    <main class="main-content">
      <!-- 左侧区域 -->
      <div class="left-panel">
        <!-- 全局统计卡片 -->
        <div class="stats-grid">
          <FloatCard :delay="1" class="stat-card">
            <template #header><i class="stat-dot sd-1"></i>总用户</template>
            <NumScroll :target="globalStats.total_users || 0" :duration="1.2" />
          </FloatCard>
          <FloatCard :delay="2" class="stat-card">
            <template #header><i class="stat-dot sd-2"></i>复购用户</template>
            <NumScroll :target="globalStats.repurchase_users || 0" :duration="1.2" />
          </FloatCard>
          <FloatCard :delay="3" class="stat-card">
            <template #header><i class="stat-dot sd-3"></i>复购转化率</template>
            <NumScroll :target="(globalStats.repurchase_rate || 0) * 100" :decimals="1" suffix="%" :duration="1.2" />
          </FloatCard>
          <FloatCard :delay="4" class="stat-card">
            <template #header><i class="stat-dot sd-4"></i>流失用户</template>
            <NumScroll :target="globalStats.churn_users || 0" :duration="1.2" />
          </FloatCard>
        </div>

        <!-- 用户分层 -->
        <div class="panel layer-panel">
          <div class="panel-title">
            <span><span class="panel-title-bar" />用户分层复购分布</span>
          </div>
          <div class="panel-body layer-grid">
            <ChartBox title="会员等级" :delay="1" :option="memberLevelOption" chart-type="pie" :show-download="false" />
            <ChartBox title="城市层级" :delay="2" :option="cityTierOption" chart-type="pie" :show-download="false" />
          </div>
        </div>

        <!-- 消费分析 -->
        <div class="panel consumption-panel">
          <div class="panel-title">
            <span><span class="panel-title-bar" />消费金额分析</span>
          </div>
          <div class="panel-body">
            <ChartBox title="" :delay="3" :option="consumptionOption" chart-type="bar" :show-download="false" />
          </div>
        </div>
      </div>

      <!-- 中间区域 -->
      <div class="center-panel">
        <!-- 降维分析 -->
        <div class="panel dim-panel">
          <div class="panel-title dim-header">
            <span><span class="panel-title-bar" />降维算法分析</span>
            <el-radio-group v-model="dimTab" size="small" @change="onDimTabChange">
              <el-radio-button value="pca">PCA 碎石图</el-radio-button>
              <el-radio-button value="factor">因子载荷</el-radio-button>
              <el-radio-button value="heatmap">相关性热力图</el-radio-button>
            </el-radio-group>
          </div>
          <div class="panel-body dim-chart">
            <ChartBox
              ref="dimChartRef"
              title=""
              :delay="1"
              :option="dimOption"
              :show-download="true"
              :show-refresh="false"
            />
          </div>
        </div>

        <!-- 双模型对比 -->
        <div class="panel model-panel">
          <div class="panel-title model-header">
            <span><span class="panel-title-bar" />双模型对比</span>
            <div class="model-tags">
              <el-tag type="primary" size="small" effect="dark">SVM</el-tag>
              <el-tag type="success" size="small" effect="dark">随机森林</el-tag>
            </div>
          </div>
          <div class="panel-body model-charts">
            <ChartBox title="ROC 曲线对比" :delay="2" :option="rocOption" chart-type="line" :show-download="false" />
            <ChartBox title="指标对比" :delay="3" :option="metricsOption" chart-type="bar" :show-download="false" />
          </div>
        </div>
      </div>

      <!-- 右侧区域 -->
      <div class="right-panel">
        <!-- 特征重要性 -->
        <div class="panel importance-panel">
          <div class="panel-title">
            <span><span class="panel-title-bar" />特征重要性 Top10</span>
          </div>
          <div class="panel-body">
            <ChartBox title="" :delay="1" :option="importanceOption" chart-type="bar" :show-download="false" />
          </div>
        </div>

        <!-- 实时预测 -->
        <div class="panel predict-panel">
          <div class="panel-title">
            <span><span class="panel-title-bar" />实时复购预测</span>
            <el-tag type="info" size="small" effect="dark">Demo</el-tag>
          </div>
          <div class="panel-body predict-body">
            <div class="predict-inputs">
              <el-input
                v-model="predictFeatures"
                type="textarea"
                :rows="3"
                placeholder="输入36个原始特征，用逗号分隔，如: 25,1,2,365,0,3,1,12,..."
                size="small"
              />
              <el-button type="primary" size="small" :loading="predictLoading" @click="handlePredict">预测</el-button>
            </div>
            <div v-if="predictResult" class="predict-result">
              <div class="result-item">
                <span class="result-label">复购概率</span>
                <span
                  class="result-value"
                  :style="{
                    color: predictResult.repurchase_probability >= 0.7 ? '#00e6a8' :
                           predictResult.repurchase_probability >= 0.4 ? '#ffb020' : '#ff4d6d'
                  }"
                >{{ (predictResult.repurchase_probability * 100).toFixed(1) }}%</span>
              </div>
              <div class="result-item">
                <span class="result-label">预测标签</span>
                <el-tag
                  :type="predictResult.prediction_label === 1 ? 'success' : 'danger'"
                  size="small"
                  effect="dark"
                >{{ predictResult.prediction_label === 1 ? '复购' : '不复购' }}</el-tag>
              </div>
              <div class="result-item">
                <span class="result-label">风险等级</span>
                <el-tag
                  :type="predictResult.risk_level === '高' ? 'danger' : predictResult.risk_level === '中' ? 'warning' : 'info'"
                  size="small"
                  effect="dark"
                >{{ predictResult.risk_level || '未知' }}</el-tag>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- 底部滚动日志 -->
    <footer class="footer">
      <div class="log-scroll">
        <span v-for="(log, idx) in logs" :key="idx" class="log-item">{{ log }}</span>
      </div>
    </footer>

    <!-- 定时刷新 -->
    <RefreshTimer @tick="onRefresh" />
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { DataAnalysis } from '@element-plus/icons-vue'
import FloatCard from '@/components/FloatCard.vue'
import NumScroll from '@/components/NumScroll.vue'
import ChartBox from '@/components/ChartBox.vue'
import RefreshTimer from '@/components/RefreshTimer.vue'
import { screenApi, modelApi, dimApi } from '@/api'

// ============ 状态 ============
const currentTime = ref('')
const backendStatus = ref(true)
const globalStats = reactive({
  total_users: 0,
  repurchase_users: 0,
  churned_users: 0,
  repurchase_rate: 0,
  total_samples: 8500,
})

// 降维 Tab
const dimTab = ref('pca')
const dimChartRef = ref(null)
const dimOption = ref({})

// 图表数据
const memberLevelOption = ref({})
const cityTierOption = ref({})
const consumptionOption = ref({})
const rocOption = ref({})
const metricsOption = ref({})
const importanceOption = ref({})

// 预测
const predictFeatures = ref('')
const predictLoading = ref(false)
const predictResult = ref(null)

// 日志
const logs = ref([
  '🔄 系统初始化完成...',
  '📊 正在加载数据集...',
  '✅ 数据加载成功，共 8500 条样本',
])

// ============ 方法 ============
const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', { hour12: false })
}

const loadGlobalStats = async () => {
  try {
    const res = await screenApi.global()
    Object.assign(globalStats, res)
    logs.value.unshift(`📊 全局指标已更新: 复购率 ${(res.repurchase_rate * 100).toFixed(1)}%`)
  } catch (e) {
    // 使用模拟数据
    globalStats.total_users = 8500
    globalStats.repurchase_users = 3825
    globalStats.churned_users = 4675
    globalStats.repurchase_rate = 0.45
  }
}

// 分类编码标签映射
const MEMBER_LEVEL_LABELS = { 0: '普通', 1: '银卡', 2: '金卡', 3: '钻石' }
const CITY_TIER_LABELS = { 0: '五线', 1: '四线', 2: '三线', 3: '二线', 4: '一线' }
const DEVICE_LABELS = { 0: '安卓', 1: '苹果', 2: 'PC', 3: '小程序' }

// 将后端 distribution 数组转换为 {标签: 用户数} 饼图数据
const buildGroupPieData = (distribution, labelMap) => {
  const out = {}
  ;(distribution || []).forEach((item) => {
    const key = String(item.group_value)
    const label = labelMap[item.group_value] || labelMap[key] || key
    out[label] = item.total_users
  })
  return out
}

const loadUserLayer = async () => {
  try {
    // 会员等级饼图
    const memberRes = await screenApi.userLayer('member_level')
    memberLevelOption.value = buildPieOption(
      buildGroupPieData(memberRes.distribution, MEMBER_LEVEL_LABELS),
      '会员等级'
    )
    // 城市层级饼图
    const cityRes = await screenApi.userLayer('city_tier')
    cityTierOption.value = buildPieOption(
      buildGroupPieData(cityRes.distribution, CITY_TIER_LABELS),
      '城市层级'
    )
  } catch (e) {
    // 模拟数据
    memberLevelOption.value = buildPieOption(
      { '普通': 45, '银卡': 25, '金卡': 20, '钻石': 10 },
      '会员等级'
    )
    cityTierOption.value = buildPieOption(
      { '一线': 30, '二线': 35, '三线': 20, '四线': 10, '五线': 5 },
      '城市层级'
    )
  }
}

const loadConsumption = async () => {
  try {
    const res = await screenApi.consumption()
    const m = res.metrics || {}
    consumptionOption.value = buildBarOption(
      ['平均消费', '中位消费', '复购均消', '流失均消'],
      [m.avg_consume || 0, m.median_consume || 0, m.avg_consume_repurchase || 0, m.avg_consume_non_repurchase || 0],
      '消费金额分布'
    )
  } catch (e) {
    consumptionOption.value = buildBarOption(
      ['平均消费', '中位消费', '复购均消', '流失均消'],
      [521, 150, 640, 396],
      '消费金额分布'
    )
  }
}

const loadModelMetrics = async () => {
  // 无真实 ROC 数据时的兜底模拟曲线
  const mockSvmRoc = { fpr: [0, 0.1, 0.3, 0.5, 0.7, 1], tpr: [0, 0.6, 0.75, 0.85, 0.92, 1] }
  const mockRfRoc = { fpr: [0, 0.05, 0.2, 0.4, 0.6, 1], tpr: [0, 0.7, 0.82, 0.9, 0.95, 1] }
  try {
    const res = await screenApi.modelMetric()
    const svmMetrics = res.svm || {}
    const rfMetrics = res.random_forest || {}
    const svmRoc = res.svm_roc || {}
    const rfRoc = res.rf_roc || {}
    const hasRealRoc =
      (Array.isArray(svmRoc.fpr) && svmRoc.fpr.length > 1) ||
      (Array.isArray(rfRoc.fpr) && rfRoc.fpr.length > 1)
    // ROC 曲线（无真实数据时用模拟曲线）
    rocOption.value = hasRealRoc ? buildRocOption(svmRoc, rfRoc) : buildRocOption(mockSvmRoc, mockRfRoc)
    // 指标对比（无真实指标时用模拟值）
    const isTrained = res.trained === true
    metricsOption.value = buildMetricsOption(
      isTrained ? svmMetrics : { accuracy: 0.82, precision: 0.80, recall: 0.78, f1: 0.79, auc: 0.85 },
      isTrained ? rfMetrics : { accuracy: 0.88, precision: 0.86, recall: 0.84, f1: 0.85, auc: 0.92 }
    )
  } catch (e) {
    // 模拟 ROC
    rocOption.value = buildRocOption(mockSvmRoc, mockRfRoc)
    metricsOption.value = buildMetricsOption(
      { accuracy: 0.82, precision: 0.80, recall: 0.78, f1: 0.79, auc: 0.85 },
      { accuracy: 0.88, precision: 0.86, recall: 0.84, f1: 0.85, auc: 0.92 }
    )
  }
}

const loadFeatureImportance = async () => {
  const mockFeatures = ['total_consume', 'total_order_num', 'silence_days', 'cart_add_cnt', 'coupon_use_cnt',
    'good_comment_cnt', 'live_watch_duration', 'register_days', 'avg_order_price', 'week_visit_cnt']
  const mockValues = [0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
  try {
    const res = await screenApi.featureTop10()
    const importance = (res && res.importance) || {}
    const features = importance.features || []
    const values = importance.importance || []
    if (features.length > 0) {
      importanceOption.value = buildHBarOption(features, values, '特征重要性')
    } else {
      importanceOption.value = buildHBarOption(mockFeatures, mockValues, '特征重要性')
    }
  } catch (e) {
    importanceOption.value = buildHBarOption(mockFeatures, mockValues, '特征重要性')
  }
}

const loadDimData = async () => {
  try {
    if (dimTab.value === 'pca') {
      const res = await dimApi.pca()
      const varRatio = res.explained_variance_ratio || []
      const cumulative = res.cumulative_variance_ratio || []
      dimOption.value = buildScreeOption(varRatio, cumulative)
    } else if (dimTab.value === 'factor') {
      const res = await dimApi.factor()
      const loadings = res.loadings_matrix || []
      dimOption.value = buildLoadingsOption(loadings)
    } else if (dimTab.value === 'heatmap') {
      const res = await dimApi.heatmap(20)   // 取方差最大的 20 个特征，保证标签可读
      dimOption.value = buildHeatmapOption(res.features || [], res.correlation || [])
    }
    if (dimChartRef.value) {
      await nextTick()
      dimChartRef.value.renderChart(dimOption.value)
    }
  } catch (e) {
    // 模拟数据
    if (dimTab.value === 'pca') {
      const varRatio = [0.25, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
      const cumulative = varRatio.reduce((acc, v, i) => [...acc, (acc[i-1] || 0) + v], [])
      dimOption.value = buildScreeOption(varRatio, cumulative)
    }
  }
}

// ============ 图表构建函数 ============
const PIE_COLORS = ['#00d4ff', '#7c5cff', '#ffd666', '#ff4d8f', '#00e6a8']

// 暗色主题统一 tooltip 样式（清晰度优化）
const baseTooltip = {
  backgroundColor: 'rgba(10, 20, 40, 0.94)',
  borderColor: 'rgba(0, 212, 255, 0.35)',
  borderWidth: 1,
  padding: [8, 12],
  textStyle: { color: '#eaf6ff', fontSize: 12 },
  extraCssText: 'box-shadow: 0 8px 24px rgba(0,0,0,0.5); border-radius: 6px;',
}

const buildPieOption = (data, title) => ({
  tooltip: { ...baseTooltip, trigger: 'item', formatter: '{b}: {c} 人 ({d}%)' },
  legend: {
    orient: 'horizontal',
    bottom: 0,
    left: 'center',
    itemWidth: 10,
    itemHeight: 10,
    itemGap: 12,
    textStyle: { color: 'rgba(205,235,255,0.85)', fontSize: 11 },
  },
  series: [{
    type: 'pie',
    radius: ['36%', '62%'],
    center: ['50%', '42%'],
    avoidLabelOverlap: true,
    itemStyle: {
      borderRadius: 6,
      borderColor: '#0a1628',
      borderWidth: 2,
    },
    label: {
      show: true,
      formatter: '{d}%',
      fontSize: 10,
      color: 'rgba(234,246,255,0.9)',
      fontWeight: 600,
    },
    labelLine: { show: false, length: 6, length2: 6 },
    emphasis: {
      scale: true,
      itemStyle: { shadowBlur: 16, shadowColor: 'rgba(0,212,255,0.45)' },
      label: { fontSize: 11 },
    },
    data: Object.entries(data).map(([name, value]) => ({ name, value })),
    color: PIE_COLORS,
  }],
})

const buildBarOption = (labels, values, title) => ({
  tooltip: { ...baseTooltip, trigger: 'axis' },
  grid: { left: 46, right: 14, top: 28, bottom: 30 },
  xAxis: {
    type: 'category',
    data: labels,
    axisLabel: { color: 'rgba(205,235,255,0.82)', fontSize: 11, interval: 0, rotate: 22 },
    axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
    axisTick: { show: false },
  },
  yAxis: {
    type: 'value',
    splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
    axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11 },
  },
  series: [{
    type: 'bar',
    data: values,
    itemStyle: {
      borderRadius: [4, 4, 0, 0],
      color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: '#00d4ff' },
        { offset: 1, color: '#0a6cff' },
      ]),
      shadowBlur: 8,
      shadowColor: 'rgba(0,212,255,0.3)',
    },
    barWidth: '46%',
    label: {
      show: true,
      position: 'top',
      fontSize: 10,
      color: 'rgba(205,235,255,0.9)',
      formatter: (p) => (p.value >= 1000 ? (p.value / 1000).toFixed(1) + 'k' : Math.round(p.value)),
    },
  }],
})

// 横向条形图（用于特征重要性，长特征名更易阅读）
const buildHBarOption = (labels, values, title) => ({
  tooltip: { ...baseTooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
  grid: { left: 92, right: 38, top: 8, bottom: 8 },
  xAxis: {
    type: 'value',
    splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
    axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11 },
  },
  yAxis: {
    type: 'category',
    data: labels,
    axisLabel: { color: 'rgba(205,235,255,0.85)', fontSize: 11 },
    axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
    axisTick: { show: false },
  },
  series: [{
    type: 'bar',
    data: values,
    barWidth: '55%',
    itemStyle: {
      borderRadius: [0, 4, 4, 0],
      color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
        { offset: 0, color: '#0a6cff' },
        { offset: 1, color: '#00d4ff' },
      ]),
      shadowBlur: 8,
      shadowColor: 'rgba(0,212,255,0.28)',
    },
    label: {
      show: true,
      position: 'right',
      fontSize: 10,
      color: 'rgba(185,224,255,0.8)',
      formatter: (p) => p.value.toFixed(3),
    },
  }],
})

const buildScreeOption = (varRatio, cumulative) => ({
  tooltip: {
    ...baseTooltip,
    trigger: 'axis',
    formatter: (params) => {
      const idx = params[0].dataIndex
      return `主成分${idx+1}: 方差${(varRatio[idx]*100).toFixed(1)}%, 累计${(cumulative[idx]*100).toFixed(1)}%`
    }
  },
  grid: { left: 50, right: 42, top: 26, bottom: 30 },
  xAxis: {
    type: 'category',
    data: varRatio.map((_, i) => `PC${i+1}`),
    axisLabel: {
      color: 'rgba(205,235,255,0.75)',
      fontSize: 10,
      interval: Math.max(1, Math.floor(varRatio.length / 12)),  // 主成分较多时抽稀，避免重叠
    },
    axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
    axisTick: { show: false },
  },
  yAxis: [
    {
      type: 'value',
      name: '方差贡献率',
      // 不强制 max=1，让柱子随数据自动缩放，碎石图更清晰
      splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
      axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11, formatter: (v) => `${(v*100).toFixed(0)}%` },
    },
    {
      type: 'value',
      name: '累计贡献率',
      max: 1,
      min: 0,
      splitLine: { show: false },
      axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11, formatter: (v) => `${(v*100).toFixed(0)}%` },
    }
  ],
  series: [
    {
      name: '方差贡献率',
      type: 'bar',
      data: varRatio,
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#00d4ff' },
          { offset: 1, color: '#0a6cff' },
        ]),
        borderRadius: [4, 4, 0, 0],
      },
      barWidth: '38%',
    },
    {
      name: '累计贡献率',
      type: 'line',
      yAxisIndex: 1,
      data: cumulative,
      smooth: true,
      symbol: 'circle',
      symbolSize: 6,
      lineStyle: { color: '#ffd666', width: 2 },
      itemStyle: { color: '#ffd666' },
    }
  ],
})

const buildLoadingsOption = (loadings) => {
  if (!loadings || loadings.length === 0) {
    return { title: { text: '无因子载荷数据', left: 'center', textStyle: { color: 'rgba(255,255,255,0.5)' } } }
  }
  const x = loadings.map(row => row[0] || 0)
  const y = loadings.map(row => row[1] || 0)
  return {
    tooltip: { ...baseTooltip, trigger: 'item', formatter: (p) => `特征${p.dataIndex+1}: (${p.data[0].toFixed(3)}, ${p.data[1].toFixed(3)})` },
    grid: { left: 50, right: 26, top: 26, bottom: 30 },
    xAxis: {
      name: '因子1 载荷',
      nameTextStyle: { color: 'rgba(205,235,255,0.8)' },
      splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
      axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11 },
      axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
    },
    yAxis: {
      name: '因子2 载荷',
      nameTextStyle: { color: 'rgba(205,235,255,0.8)' },
      splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
      axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11 },
      axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
    },
    series: [{
      type: 'scatter',
      data: x.map((v, i) => [v, y[i]]),
      symbolSize: 8,
      itemStyle: { color: '#00d4ff', shadowBlur: 12, shadowColor: 'rgba(0,212,255,0.4)' },
      emphasis: { scale: 1.6 },
    }],
  }
}

// 相关性热力图（ECharts 5 原生 heatmap）
const buildHeatmapOption = (features, correlation) => {
  const n = features.length
  const cells = []
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      cells.push([j, i, correlation[i] ? correlation[i][j] : 0])
    }
  }
  return {
    tooltip: {
      ...baseTooltip,
      position: 'top',
      formatter: (p) =>
        `${features[p.value[1]]} × ${features[p.value[0]]}<br/>相关系数: <b>${p.value[2].toFixed(3)}</b>`,
    },
    grid: { left: 100, right: 24, top: 8, bottom: 96 },
    xAxis: {
      type: 'category',
      data: features,
      splitArea: { show: true },
      axisLabel: { rotate: 45, fontSize: 9, color: 'rgba(205,235,255,0.75)', interval: 0 },
      axisLine: { lineStyle: { color: 'rgba(0,212,255,0.2)' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'category',
      data: features,
      splitArea: { show: true },
      axisLabel: { fontSize: 9, color: 'rgba(205,235,255,0.75)', interval: 0 },
      axisLine: { lineStyle: { color: 'rgba(0,212,255,0.2)' } },
      axisTick: { show: false },
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 4,
      itemWidth: 12,
      itemHeight: 110,
      text: ['+1', '-1'],
      textStyle: { color: 'rgba(205,235,255,0.7)', fontSize: 10 },
      inRange: { color: ['#0a6cff', '#122a4d', '#ff4d8f'] },
    },
    series: [{
      type: 'heatmap',
      data: cells,
      label: { show: false },
      itemStyle: { borderColor: 'rgba(255,255,255,0.07)', borderWidth: 1 },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,212,255,0.5)' },
      },
    }],
  }
}

const buildRocOption = (svmRoc, rfRoc) => ({
  tooltip: { ...baseTooltip, trigger: 'axis', formatter: (params) => {
    const p = params[0]
    return `假阳性率: ${p.data[0].toFixed(3)}<br/>真阳性率: ${p.data[1].toFixed(3)}`
  }},
  legend: {
    data: ['SVM', '随机森林', '随机猜想'],
    textStyle: { color: 'rgba(205,235,255,0.85)', fontSize: 11 },
    top: 0,
  },
  grid: { left: 42, right: 18, top: 26, bottom: 26 },
  xAxis: {
    name: '假阳性率',
    nameTextStyle: { color: 'rgba(205,235,255,0.8)' },
    min: 0,
    max: 1,
    splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
    axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11 },
    axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
  },
  yAxis: {
    name: '真阳性率',
    nameTextStyle: { color: 'rgba(205,235,255,0.8)' },
    min: 0,
    max: 1,
    splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
    axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11 },
    axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
  },
  series: [
    {
      name: 'SVM',
      type: 'line',
      data: (svmRoc.fpr || []).map((v, i) => [v, (svmRoc.tpr || [])[i] || 0]),
      smooth: true,
      lineStyle: { color: '#00d4ff', width: 2.5 },
      itemStyle: { color: '#00d4ff' },
      symbol: 'none',
      areaStyle: { opacity: 0.06 },
    },
    {
      name: '随机森林',
      type: 'line',
      data: (rfRoc.fpr || []).map((v, i) => [v, (rfRoc.tpr || [])[i] || 0]),
      smooth: true,
      lineStyle: { color: '#00e6a8', width: 2.5 },
      itemStyle: { color: '#00e6a8' },
      symbol: 'none',
      areaStyle: { opacity: 0.06 },
    },
    {
      name: '随机猜想',
      type: 'line',
      data: [[0, 0], [1, 1]],
      lineStyle: { color: 'rgba(255,255,255,0.28)', width: 1, type: 'dashed' },
      symbol: 'none',
    }
  ],
})

const buildMetricsOption = (svm, rf) => {
  const metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
  const labels = ['准确率', '精确率', '召回率', 'F1', 'AUC']
  return {
    tooltip: { ...baseTooltip, trigger: 'axis' },
    legend: {
      data: ['SVM', '随机森林'],
      textStyle: { color: 'rgba(205,235,255,0.85)', fontSize: 11 },
      top: 0,
    },
    grid: { left: 46, right: 18, top: 30, bottom: 26 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: 'rgba(205,235,255,0.82)', fontSize: 11 },
      axisLine: { lineStyle: { color: 'rgba(0,212,255,0.15)' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      splitLine: { lineStyle: { color: 'rgba(0,212,255,0.06)' } },
      axisLabel: { color: 'rgba(205,235,255,0.7)', fontSize: 11 },
    },
    series: [
      {
        name: 'SVM',
        type: 'bar',
        data: metrics.map(m => svm[m] || 0),
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#00d4ff' },
            { offset: 1, color: '#0a6cff' },
          ]),
        },
        barWidth: '30%',
        label: {
          show: true,
          position: 'top',
          fontSize: 9,
          color: 'rgba(205,235,255,0.9)',
          formatter: (p) => p.value.toFixed(2),
        },
      },
      {
        name: '随机森林',
        type: 'bar',
        data: metrics.map(m => rf[m] || 0),
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#7c5cff' },
            { offset: 1, color: '#4b2fcc' },
          ]),
        },
        barWidth: '30%',
        label: {
          show: true,
          position: 'top',
          fontSize: 9,
          color: 'rgba(205,235,255,0.9)',
          formatter: (p) => p.value.toFixed(2),
        },
      }
    ],
  }
}

// ============ 预测 ============
const handlePredict = async () => {
  if (!predictFeatures.value.trim()) {
    ElMessage.warning('请输入特征向量')
    return
  }
  try {
    const features = predictFeatures.value.split(',').map(v => parseFloat(v.trim()))
    if (features.some(isNaN)) {
      ElMessage.warning('请输入有效的数字，用逗号分隔')
      return
    }
    predictLoading.value = true
    const res = await modelApi.predict(features, false)
    predictResult.value = res
    logs.value.unshift(`🎯 预测完成: 复购概率 ${(res.repurchase_probability * 100).toFixed(1)}%`)
  } catch (e) {
    // 模拟预测
    const prob = 0.3 + Math.random() * 0.5
    predictResult.value = {
      repurchase_probability: prob,
      prediction_label: prob > 0.5 ? 1 : 0,
      risk_level: prob >= 0.7 ? '高' : prob >= 0.4 ? '中' : '低',
      model_used: '随机森林',
      predict_time_ms: 12.5,
    }
  } finally {
    predictLoading.value = false
  }
}

// ============ 刷新 ============
const onRefresh = () => {
  loadGlobalStats()
  loadUserLayer()
  loadConsumption()
  loadModelMetrics()
  loadFeatureImportance()
  if (dimTab.value !== 'heatmap') {
    loadDimData()
  }
  logs.value.unshift(`🔄 数据已刷新 ${new Date().toLocaleTimeString()}`)
}

const onDimTabChange = () => {
  loadDimData()
}

// ============ 定时器 ============
let timeInterval = null

// ============ 生命周期 ============
onMounted(() => {
  updateTime()
  timeInterval = setInterval(updateTime, 1000)

  // 加载所有数据
  onRefresh()

  // 欢迎日志
  setTimeout(() => {
    logs.value.unshift('🚀 大屏已启动，欢迎使用！')
  }, 500)
})

onBeforeUnmount(() => {
  if (timeInterval) {
    clearInterval(timeInterval)
  }
})
</script>

<style scoped lang="scss">
.screen-dash {
  position: relative;
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 0 16px 10px;
  background:
    radial-gradient(1200px 620px at 82% -12%, rgba(10, 108, 255, 0.16), transparent 62%),
    radial-gradient(900px 520px at -12% 110%, rgba(124, 92, 255, 0.14), transparent 62%),
    #070e1e;
}

/* ===== 背景装饰 ===== */
.bg-decoration {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  overflow: hidden;
}

.bg-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0, 212, 255, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 212, 255, 0.04) 1px, transparent 1px);
  background-size: 42px 42px;
  -webkit-mask-image: radial-gradient(ellipse 72% 68% at 50% 44%, #000 25%, transparent 78%);
  mask-image: radial-gradient(ellipse 72% 68% at 50% 44%, #000 25%, transparent 78%);
}

.bg-glow-1 {
  position: absolute;
  top: -25%;
  right: -12%;
  width: 55%;
  height: 60%;
  background: radial-gradient(ellipse, rgba(0, 212, 255, 0.14), transparent 65%);
}

.bg-glow-2 {
  position: absolute;
  bottom: -20%;
  left: -12%;
  width: 50%;
  height: 55%;
  background: radial-gradient(ellipse, rgba(124, 92, 255, 0.14), transparent 65%);
}

.bg-aurora {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(620px 320px at 28% 18%, rgba(0, 212, 255, 0.09), transparent 72%),
    radial-gradient(720px 360px at 76% 82%, rgba(124, 92, 255, 0.08), transparent 72%);
  animation: aurora-move 14s ease-in-out infinite;
}

/* ===== 顶部标题栏 ===== */
.header {
  position: relative;
  z-index: 1;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 12px 0 10px;
  padding: 14px 22px;
  border-radius: $border-radius;
  overflow: hidden;
  background:
    linear-gradient(90deg, rgba(12, 26, 50, 0.9), rgba(10, 108, 255, 0.14), rgba(12, 26, 50, 0.9)) padding-box,
    linear-gradient(90deg, rgba(0, 212, 255, 0.45), rgba(124, 92, 255, 0.25), rgba(0, 212, 255, 0.45)) border-box;
  border: 1px solid transparent;
  background-size: 200% 100%;
  box-shadow: 0 10px 32px rgba(0, 0, 0, 0.35), inset 0 0 46px rgba(0, 212, 255, 0.03);
  animation: gradient-flow 12s ease infinite;

  /* 底部流光扫光 */
  &::after {
    content: '';
    position: absolute;
    left: -40%;
    bottom: 0;
    width: 30%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7c5cff, transparent);
    filter: drop-shadow(0 0 8px rgba(0, 212, 255, 0.9));
    animation: beam-sweep 4.2s linear infinite;
    pointer-events: none;
  }
}

.header-left {
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}

.logo-icon {
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  border-radius: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  color: #fff;
  background: linear-gradient(135deg, #00d4ff, #0a6cff);
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}

.header-titles {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}

.title {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 2px;
  white-space: nowrap;
  background: linear-gradient(90deg, #ffffff, #00d4ff 55%, #7c5cff);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle {
  font-size: 12px;
  letter-spacing: 1px;
  color: $text-dim;
}

.header-right {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 13px;
  color: $text-sub;
}

.time {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-variant-numeric: tabular-nums;
}

.time-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #00e6a8;
  box-shadow: 0 0 8px rgba(0, 230, 168, 0.9);
  animation: pulse-glow 2.4s ease-in-out infinite;
}

.sample-count {
  padding: 4px 12px;
  border-radius: 999px;
  background: rgba(0, 212, 255, 0.08);
  border: 1px solid rgba(0, 212, 255, 0.22);
  color: $text-sub;
  font-size: 12px;
}

/* ===== 主体布局 ===== */
.main-content {
  position: relative;
  z-index: 1;
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr) 320px;
  gap: 12px;
}

/* ===== 通用面板（带流光） ===== */
.panel {
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  padding: 12px 14px;
  border-radius: $border-radius;
  background: linear-gradient(165deg, rgba(14, 32, 60, 0.74), rgba(8, 16, 32, 0.9));
  border: 1px solid rgba(0, 212, 255, 0.14);
  box-shadow: $shadow-card, inset 0 0 32px rgba(0, 212, 255, 0.025);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  overflow: hidden;

  /* 顶部流光扫光 */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -90%;
    width: 55%;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.85), rgba(124, 92, 255, 0.85), transparent);
    filter: drop-shadow(0 0 5px rgba(0, 212, 255, 0.8));
    animation: beam-sweep 5.2s linear infinite;
    pointer-events: none;
  }

  /* 左上角光晕 */
  &::after {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    width: 46px;
    height: 46px;
    background: radial-gradient(circle at top left, rgba(0, 212, 255, 0.24), transparent 72%);
    pointer-events: none;
  }
}

.panel-title {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 1px;
  color: $text-main;
}

.panel-title-bar {
  display: inline-block;
  width: 3px;
  height: 14px;
  margin-right: 8px;
  border-radius: 2px;
  background: linear-gradient(180deg, #00d4ff, #7c5cff);
  box-shadow: 0 0 8px rgba(0, 212, 255, 0.6);
}

.panel-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

/* ===== 左侧区域 ===== */
.left-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  min-height: 0;
}

.stats-grid {
  flex-shrink: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;

  .stat-card {
    :deep(.card-header) {
      display: inline-flex;
      align-items: center;
      font-size: 12px;
      color: $text-dim;
    }
    :deep(.card-body) {
      font-size: 26px;
      font-weight: 700;
      background: linear-gradient(180deg, #ffffff, #00d4ff);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }
  }
}

/* KPI 卡片彩色状态点 */
.stat-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.sd-1 { background: #00d4ff; box-shadow: 0 0 8px rgba(0, 212, 255, 0.85); }
.sd-2 { background: #00e6a8; box-shadow: 0 0 8px rgba(0, 230, 168, 0.85); }
.sd-3 { background: #ffb020; box-shadow: 0 0 8px rgba(255, 176, 32, 0.85); }
.sd-4 { background: #ff4d6d; box-shadow: 0 0 8px rgba(255, 77, 109, 0.85); }

.layer-panel {
  flex: 1;
  min-height: 0;
}

.layer-grid {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 8px;
}

.consumption-panel {
  flex: 0 0 220px;
  min-height: 0;
}

/* ===== 中间区域 ===== */
.center-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  min-height: 0;
  overflow-y: auto;               // 小屏高度不足时允许滚动，避免图表被压缩
  overscroll-behavior: contain;
}

.dim-panel {
  flex: 1 1 0%;
  min-height: 250px;
}

.model-panel {
  flex: 1 1 0%;
  min-height: 250px;
}

.dim-header {
  .el-radio-group {
    flex-shrink: 0;
  }
}

.model-charts {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

/* ===== 右侧区域 ===== */
.right-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  min-height: 0;
}

.importance-panel {
  flex: 1;
  min-height: 0;
}

.predict-panel {
  flex: 0 0 240px;
  min-height: 0;
}

/* ===== 预测面板 ===== */
.predict-body {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.predict-inputs {
  display: flex;
  flex-direction: column;
  gap: 8px;

  .el-button {
    align-self: flex-end;
  }
}

.predict-result {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
  padding: 10px 12px;
  border-radius: $border-radius-sm;
  background: rgba(0, 212, 255, 0.05);
  border: 1px solid rgba(0, 212, 255, 0.12);
}

.result-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  text-align: center;
}

.result-label {
  font-size: 11px;
  color: $text-dim;
}

.result-value {
  font-size: 20px;
  font-weight: 700;
}

/* ===== 底部日志 ===== */
.footer {
  position: relative;
  z-index: 1;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  height: 30px;
  margin-top: 10px;
  padding: 0 14px;
  border-radius: $border-radius-sm;
  background: rgba(8, 16, 32, 0.6);
  border: 1px solid rgba(0, 212, 255, 0.1);
  overflow: hidden;
}

.log-scroll {
  display: flex;
  gap: 30px;
  white-space: nowrap;
  animation: scrollLog 40s linear infinite;

  .log-item {
    font-size: 12px;
    color: $text-dim;
  }
}

@keyframes scrollLog {
  0% {
    transform: translateX(100%);
  }
  100% {
    transform: translateX(-100%);
  }
}

/* ===== 响应式 ===== */
@media (max-width: 1440px) {
  .main-content {
    grid-template-columns: 290px minmax(0, 1fr) 290px;
  }
  .title {
    font-size: 17px;
  }
}

@media (max-width: 1200px) {
  .main-content {
    grid-template-columns: 265px minmax(0, 1fr) 265px;
  }
  .subtitle {
    display: none;
  }
}
</style>
