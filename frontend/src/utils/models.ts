/* ============================================================
   对话模型配置：本地偏好（localStorage）
   - 切换/新增模型：模型名 + 兼容地址 + 密钥存本地
   - 未选中任何自定义模型 = 使用服务端默认模型（请求不带模型字段）
   - 密钥敏感，仅存当前浏览器本地，不落服务器
   ============================================================ */

import type { ModelConfig } from '../api/types'

/** localStorage 密钥：已保存的模型列表 */
const MODELS_KEY = 'gradassist_models'
/** localStorage 密钥：当前选中模型 id（空 = 服务端默认） */
const ACTIVE_KEY = 'gradassist_active_model_id'

/** 读取全部已保存模型（非法数据兜底为空数组） */
export function listModels(): ModelConfig[] {
  try {
    const raw = localStorage.getItem(MODELS_KEY)
    const arr = raw ? (JSON.parse(raw) as ModelConfig[]) : []
    return Array.isArray(arr) ? arr : []
  } catch {
    return []
  }
}

/** 持久化模型列表 */
function saveModels(list: ModelConfig[]): void {
  localStorage.setItem(MODELS_KEY, JSON.stringify(list))
}

/** 新增一个模型（生成 id 后持久化），返回落地后的模型 */
export function addModel(input: Omit<ModelConfig, 'id'>): ModelConfig {
  const cfg: ModelConfig = { id: crypto.randomUUID(), ...input }
  const list = listModels()
  list.push(cfg)
  saveModels(list)
  return cfg
}

/** 删除指定 id 的模型；若删的是当前选中则清空选中（回到服务端默认） */
export function removeModel(id: string): void {
  saveModels(listModels().filter((m) => m.id !== id))
  if (getActiveModelId() === id) setActiveModelId(null)
}

/** 当前选中模型 id（空串/null 均视为服务端默认） */
export function getActiveModelId(): string | null {
  return localStorage.getItem(ACTIVE_KEY) || null
}

/** 设置当前选中模型 id（传 null 回到服务端默认） */
export function setActiveModelId(id: string | null): void {
  if (id) localStorage.setItem(ACTIVE_KEY, id)
  else localStorage.removeItem(ACTIVE_KEY)
}

/** 当前选中的模型实体；未选中（服务端默认）返回 null */
export function getActiveModel(): ModelConfig | null {
  const id = getActiveModelId()
  if (!id) return null
  return listModels().find((m) => m.id === id) ?? null
}