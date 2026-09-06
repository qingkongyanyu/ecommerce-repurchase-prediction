// 前端路由配置
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'ScreenDash',
    component: () => import('@/views/ScreenDash.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
