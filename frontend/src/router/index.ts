import { createRouter, createWebHistory } from 'vue-router'
import LibraryView from '../views/LibraryView.vue'
import HomeView from '../views/HomeView.vue'
import ProgressView from '../views/ProgressView.vue'
import ChatView from '../views/ChatView.vue'

/**
 * 路由配置：
 * /         书架（教材列表，点击进入对话）
 * /ingest   添加教材（上传 → 解析，原首页逻辑）
 * /progress 摄入进度（task_id 经 query 传递）
 * /chat     多轮对话（textbook 经 query 传递）
 */
const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'library', component: LibraryView },
    { path: '/ingest', name: 'ingest', component: HomeView },
    { path: '/progress', name: 'progress', component: ProgressView },
    { path: '/chat', name: 'chat', component: ChatView },
  ],
})

export default router
