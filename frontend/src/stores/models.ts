/* ============================================================
   模型配置 store：对话模型本地偏好（Pinia 化 models.ts）
   - 跨组件共享模型列表 + 当前选中模型
   - 底层 local偏好存储见 utils/models.ts，store 负责响应式状态
   ============================================================ */

import { defineStore } from 'pinia'
import type { ModelConfig } from '../api/types'
import {
  addModel,
  getActiveModelId,
  listModels,
  removeModel,
  setActiveModelId,
} from '../utils/models'

export const useModelsStore = defineStore('models', {
  state: () => ({
    /** 已保存的模型列表 */
    models: [] as ModelConfig[],
    /** 当前选中模型 id（null = 服务端默认） */
    activeModelId: null as string | null,
  }),
  getters: {
    /** 当前选中的模型实体；未选中（服务端默认）返回 null */
    activeModel: (s): ModelConfig | null =>
      s.activeModelId ? (s.models.find((m) => m.id === s.activeModelId) ?? null) : null,
    /** 当前模型展示名 */
    activeLabel: (s): string => {
      if (!s.activeModelId) return '默认模型'
      return s.models.find((m) => m.id === s.activeModelId)?.label ?? s.activeModelId
    },
  },
  actions: {
    /** 从 localStorage 加载（应用挂载时调用一次） */
    init() {
      this.models = listModels()
      this.activeModelId = getActiveModelId()
    },
    /** 切换选中模型；传 null 回到服务端默认 */
    select(id: string | null) {
      setActiveModelId(id)
      this.activeModelId = id
    },
    /** 新增一个模型并立即选中，返回落地后的模型 */
    add(input: Omit<ModelConfig, 'id'>): ModelConfig {
      const cfg = addModel(input)
      this.models = listModels()
      setActiveModelId(cfg.id)
      this.activeModelId = cfg.id
      return cfg
    },
    /** 删除指定模型（删当前选中则自动回到服务端默认） */
    remove(id: string) {
      removeModel(id)
      this.models = listModels()
      this.activeModelId = getActiveModelId()
    },
  },
})