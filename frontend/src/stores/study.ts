/* ============================================================
   学习上下文 store：当前教材等跨页面共享状态
   - 书架点击教材 → 写入 currentTextbook → 对话页据此初始化
   - 聊天内部的会话 / 消息流等局部状态仍由 ChatView 自己管理
   ============================================================ */

import { defineStore } from 'pinia'

export const useStudyStore = defineStore('study', {
  state: () => ({
    /** 当前打开的教材完整名（与后端过滤/检索精确匹配） */
    currentTextbook: '' as string,
  }),
  actions: {
    /** 进入某教材：记录当前教材名 */
    setTextbook(name: string) {
      this.currentTextbook = name
    },
  },
})