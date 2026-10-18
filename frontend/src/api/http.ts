/* ============================================================
   axios 请求层：统一 baseURL、注入 X-User-Id、统一错误处理
   - 所有网络请求收敛到此实例，组件内禁止直接 fetch
   - baseURL 来自 .env（VITE_API_BASE_URL），部署改环境变量即可
   ============================================================ */

import axios from 'axios'
import { getUserId } from '../utils/user'

/** 后端 API 基础地址（Vite 从 .env 注入，兜底 localhost:8000） */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/** 共享 axios 实例：自动附加匿名设备身份头，错误统一转成中文 Error */
export const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 教材解析/问答为长任务，超时放宽
})

// 请求拦截：附带 X-User-Id（匿名设备身份，后端据此软隔离）
http.interceptors.request.use((config) => {
  config.headers.set('X-User-Id', getUserId())
  return config
})

// 响应拦截：非 2xx 统一提取后端 detail 字段，抛带中文信息的 Error
http.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err?.response?.data?.detail
    const status = err?.response?.status
    const msg = typeof detail === 'string' ? detail : `请求失败：HTTP ${status ?? '未知'}`
    return Promise.reject(new Error(msg))
  },
)