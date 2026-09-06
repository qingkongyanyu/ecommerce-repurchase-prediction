// 前端 API 封装
import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建 axios 实例
const apiClient = axios.create({
  baseURL: '/api',
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 可在此添加 loading 或 token
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    const message = error.response?.data?.message || error.message || '请求失败'
    ElMessage.error(message)
    return Promise.reject(error)
  }
)

// ============ 数据管理接口 ============
export const dataApi = {
  // 加载数据集概览
  load: () => apiClient.get('/data/load'),
  // 预览数据（分页）
  preview: (page = 1, pageSize = 10) =>
    apiClient.post('/data/preview', { page, page_size: pageSize }),
  // 数据清洗
  clean: () => apiClient.post('/data/clean'),
  // 导出数据
  export: (format = 'csv', useCleaned = true) =>
    apiClient.post('/data/export', { format, use_cleaned: useCleaned }),
  // 下载文件
  download: (path) => `/api/data/download?path=${encodeURIComponent(path)}`,
}

// ============ 降维分析接口 ============
export const dimApi = {
  // PCA 分析
  pca: (nComponents = null, returnComponents = false) =>
    apiClient.get('/dim/pca', {
      params: { n_components: nComponents, return_components: returnComponents },
    }),
  // 因子分析
  factor: (nFactors = null, returnScores = false) =>
    apiClient.get('/dim/factor', {
      params: { n_factors: nFactors, return_scores: returnScores },
    }),
  // 融合降维
  fusion: (pcaComponents = null, factorComponents = 10) =>
    apiClient.get('/dim/fusion', {
      params: { pca_components: pcaComponents, factor_components: factorComponents },
    }),
  // 相关性热力图
  heatmap: (maxFeatures = 30) =>
    apiClient.get('/dim/heatmap', { params: { max_features: maxFeatures } }),
  // 降维摘要
  summary: () => apiClient.get('/dim/summary'),
}

// ============ 模型训练接口 ============
export const modelApi = {
  // 训练模型
  train: (modelType, useFusion = true, params = {}) =>
    apiClient.post('/model/train', { model_type: modelType, use_fusion_features: useFusion, ...params }),
  // 双模型对比
  compare: (useFusion = true) =>
    apiClient.get('/model/compare', { params: { use_fusion: useFusion } }),
  // 预测
  predict: (features, useFusion = false) =>
    apiClient.post('/model/predict', { features, use_fusion_features: useFusion }),
  // 特征重要性
  importance: (useFusion = true) =>
    apiClient.get('/model/importance', { params: { use_fusion: useFusion } }),
  // 导出报告
  report: () => apiClient.get('/model/report'),
}

// ============ 大屏专用接口 ============
export const screenApi = {
  // 全局指标
  global: () => apiClient.get('/screen/global'),
  // 用户分层（会员、城市、设备），groupBy 取值：member_level / city_tier / device_type
  userLayer: (groupBy = 'member_level') =>
    apiClient.get('/screen/user-layer', { params: { group_by: groupBy } }),
  // 模型实时指标
  modelMetric: () => apiClient.get('/screen/model-metrics'),
  // Top10 特征重要性
  featureTop10: () => apiClient.get('/screen/feature-importance'),
  // 消费指标统计
  consumption: () => apiClient.get('/screen/consumption'),
  // 全部大屏数据（一次请求获取所有）
  all: () => apiClient.get('/screen/all'),
}

export default {
  dataApi,
  dimApi,
  modelApi,
  screenApi,
}
