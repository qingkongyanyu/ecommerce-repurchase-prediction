<template>
  <div class="chart-box-wrapper">
    <FloatCard
      :delay="delay"
      :hover-scale="true"
      :width="width"
      :height="height"
      :bg-color="bgColor"
      :border-color="borderColor"
    >
      <template #header>
        <div class="chart-header">
          <span class="chart-title">{{ title }}</span>
          <div class="chart-actions">
            <el-button
              v-if="showDownload"
              size="small"
              text
              @click="handleDownload"
            >
              <el-icon><Download /></el-icon>
            </el-button>
            <el-button
              v-if="showRefresh"
              size="small"
              text
              :loading="refreshing"
              @click="handleRefresh"
            >
              <el-icon><Refresh /></el-icon>
            </el-button>
          </div>
        </div>
      </template>
      <div class="chart-container" ref="chartRef" />
    </FloatCard>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import { Download, Refresh } from '@element-plus/icons-vue'
import FloatCard from './FloatCard.vue'

const props = defineProps({
  title: {
    type: String,
    default: '图表',
  },
  width: {
    type: String,
    default: '100%',
  },
  height: {
    type: String,
    default: '100%',
  },
  delay: {
    type: Number,
    default: 1,
  },
  bgColor: {
    type: String,
    default: 'rgba(10, 22, 40, 0.75)',
  },
  borderColor: {
    type: String,
    default: 'rgba(64, 169, 255, 0.25)',
  },
  showDownload: {
    type: Boolean,
    default: true,
  },
  showRefresh: {
    type: Boolean,
    default: true,
  },
  // ECharts 配置
  option: {
    type: Object,
    default: () => ({}),
  },
  // 是否自动渲染
  autoRender: {
    type: Boolean,
    default: true,
  },
  // 图表类型（用于默认配置）
  chartType: {
    type: String,
    default: 'line',
    validator: (v) => ['line', 'bar', 'pie', 'scatter', 'heatmap', 'funnel'].includes(v),
  },
})

const emit = defineEmits(['refresh', 'rendered'])

const chartRef = ref(null)
let chartInstance = null
const refreshing = ref(false)

// 获取 ECharts 实例
const getChart = () => {
  if (!chartInstance) {
    chartInstance = echarts.init(chartRef.value, 'dark', {
      renderer: 'canvas',
      devicePixelRatio: window.devicePixelRatio || 1, // 高 DPI 屏幕渲染更清晰
    })
  }
  return chartInstance
}

// 渲染图表
const renderChart = (option = null) => {
  const chart = getChart()
  const opt = option || props.option
  if (opt && Object.keys(opt).length > 0) {
    chart.setOption(opt, true)
    chart.resize()
    emit('rendered')
  }
}

// 自适应
const resize = () => {
  if (chartInstance) {
    chartInstance.resize()
  }
}

// 下载图片
const handleDownload = () => {
  const chart = getChart()
  const url = chart.getDataURL({
    type: 'png',
    pixelRatio: 2,
    backgroundColor: '#0a1628',
  })
  const link = document.createElement('a')
  link.download = `${props.title}.png`
  link.href = url
  link.click()
}

// 刷新
const handleRefresh = async () => {
  refreshing.value = true
  emit('refresh')
  await nextTick()
  refreshing.value = false
}

// 清理实例
const dispose = () => {
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
}

onMounted(() => {
  if (props.autoRender) {
    renderChart()
  }
  // 监听窗口 resize
  window.addEventListener('resize', resize)
  // 使用 ResizeObserver 更精确
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => resize())
    if (chartRef.value) {
      ro.observe(chartRef.value)
    }
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  dispose()
})

watch(
  () => props.option,
  (newOpt) => {
    if (newOpt && Object.keys(newOpt).length > 0) {
      renderChart(newOpt)
    }
  },
  { deep: true }
)

// 暴露方法给父组件
defineExpose({
  renderChart,
  resize,
  getChart,
  dispose,
})
</script>

<style scoped lang="scss">
.chart-box-wrapper {
  flex: 1;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.chart-title {
  font-size: 13px;
  font-weight: 500;
  color: rgba(234, 246, 255, 0.9);
  letter-spacing: 0.5px;
}

.chart-actions {
  display: flex;
  gap: 4px;

  .el-button {
    color: rgba(255, 255, 255, 0.4);
    padding: 4px 8px;

    &:hover {
      color: #00d4ff;
    }
  }
}

/* 图表容器填满卡片主体（FloatCard 的 card-body 为 flex:1） */
.chart-container {
  width: 100%;
  height: 100%;
  min-height: 0;
}
</style>