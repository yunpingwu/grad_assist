/* ============================================================
   用户身份 store：匿名设备身份 + 恢复码（Pinia 化 user.ts）
   - 跨组件共享 user_id / 恢复码 opt-in 状态
   - 底层仍走 localStorage（见 utils/user.ts），store 负责响应式状态
   ============================================================ */

import { defineStore } from 'pinia'
import {
  createUserId,
  getStoredUserId,
  hasRecoveryOptIn,
  setRecoveryOptIn,
  setUserId,
} from '../utils/user'

export const useUserStore = defineStore('user', {
  state: () => ({
    /** 当前设备 user_id（同时充当恢复码） */
    userId: '' as string,
    /** 是否已显式拥有恢复码（创建过或粘贴过） */
    hasCode: false as boolean,
  }),
  getters: {
    /** 恢复码文案：未拥有时为空 */
    recoveryCode: (s) => (s.hasCode ? s.userId : ''),
  },
  actions: {
    /** 依据 localStorage 初始化（应用挂载时调用一次） */
    init() {
      this.hasCode = hasRecoveryOptIn()
      if (this.hasCode) this.userId = getStoredUserId() || createUserId()
    },
    /** 创建恢复码：生成并持久化，随后转为可复制展示 */
    create() {
      if (this.hasCode) return
      this.userId = createUserId()
      setRecoveryOptIn()
      this.hasCode = true
    },
    /** 粘贴恢复码找回历史：写入 localStorage 并覆盖当前身份 */
    restore(code: string): { ok: boolean; error?: string } {
      const v = code.trim()
      if (!v) return { ok: false, error: '请输入恢复码' }
      setUserId(v)
      setRecoveryOptIn()
      this.userId = v
      this.hasCode = true
      return { ok: true }
    },
  },
})