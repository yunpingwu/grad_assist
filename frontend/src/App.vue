<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { NConfigProvider, NDialogProvider, NMessageProvider } from 'naive-ui'
import { useUserStore } from './stores/user'
import { API_BASE_URL } from './api/http'
import { paperInkTheme } from './theme/naive'

/** 根组件：全局页眉 + 路由出口 + 页脚 */

const userStore = useUserStore()

/** 复制成功反馈（短暂显示「已复制」） */
const copied = ref(false)
/** 「恢复历史」弹层 */
const restoreOpen = ref(false)
const restoreInput = ref('')
const restoreError = ref('')
let copyTimer: number | undefined

onMounted(() => {
  // 依据显式 opt-in 判定是否展示恢复码；请求所需的匿名 id 可能被子组件挂载时自动生成，
  // 故不能以 userId 是否为空判断，只能看用户是否显式创建/粘贴过
  userStore.init()
  document.addEventListener('click', onDocClick)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
})

/** 点击弹层以外的区域时收起弹层 */
function onDocClick() {
  restoreOpen.value = false
}

/** 点击「创建恢复码」：生成并持久化，随后转为可复制的恢复码展示 */
function createRecoveryCode(): void {
  userStore.create()
}

/** 复制文本：优先 Clipboard API，非安全上下文(非 https/localhost)退回 execCommand */
async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* 继续走兜底 */
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

/** 点击恢复码：复制并给出短暂反馈 */
function copyCurrent(): void {
  void (async () => {
    const ok = await copyText(userStore.userId)
    copied.value = ok
    window.clearTimeout(copyTimer)
    copyTimer = window.setTimeout(() => (copied.value = false), 1500)
  })()
}

/** 开合「恢复历史」弹层 */
function toggleRestore(): void {
  restoreOpen.value = !restoreOpen.value
  if (restoreOpen.value) restoreError.value = ''
}

/** 粘贴恢复码找回历史：写入 localStorage 并覆盖当前身份 */
function applyRestore(): void {
  const r = userStore.restore(restoreInput.value)
  if (!r.ok) {
    restoreError.value = r.error ?? '恢复码无效'
    return
  }
  restoreInput.value = ''
  restoreOpen.value = false
}
</script>

<template>
  <n-config-provider :theme-overrides="paperInkTheme">
    <n-message-provider>
      <n-dialog-provider>
        <div class="app">
          <header class="site-header">
            <router-link class="brand" to="/">
              <span class="brand-glyph" aria-hidden="true">G</span>
              <span class="brand-name">GradAssist</span>
              <span class="brand-sep">/</span>
              <span class="brand-sub">教材问答</span>
            </router-link>
            <nav class="header-actions">
              <!-- 创建恢复码：未拥有时展示按钮，创建后变为可复制的恢复码 -->
              <button v-if="!userStore.hasCode" class="create-btn" type="button" @click="createRecoveryCode">
                创建恢复码
              </button>
              <button v-else class="code-chip" type="button" :title="userStore.userId" @click="copyCurrent">
                <span class="code-chip-label">{{ copied ? '已复制 ✓' : '恢复码' }}</span>
                <span class="code-chip-id">{{ userStore.userId }}</span>
              </button>

              <!-- 恢复历史：粘贴其他设备的恢复码找回历史 -->
              <div class="restore-wrap">
                <button class="restore-btn" type="button" @click.stop="toggleRestore">恢复历史</button>
                <div v-if="restoreOpen" class="restore-panel" @click.stop>
                  <div class="restore-panel-head">
                    <span class="restore-title">恢复历史</span>
                    <button class="restore-close" type="button" aria-label="关闭" @click="restoreOpen = false">✕</button>
                  </div>
                  <label class="restore-label" for="restore-input">粘贴其他设备的恢复码，找回该设备的教材与会话历史</label>
                  <div class="restore-input-row">
                    <input
                      id="restore-input"
                      v-model="restoreInput"
                      class="restore-input"
                      type="text"
                      placeholder="粘贴恢复码"
                      :class="{ 'is-error': restoreError }"
                      @keydown.enter="applyRestore"
                    />
                    <button class="restore-apply" type="button" @click="applyRestore">恢复</button>
                  </div>
                  <p v-if="restoreError" class="restore-error">{{ restoreError }}</p>
                </div>
              </div>

              <router-link class="add-btn" to="/ingest">＋ 添加教材</router-link>
              <a class="header-link" :href="`${API_BASE_URL}/docs`" target="_blank" rel="noopener">
                API 文档 <span aria-hidden="true">↗</span>
              </a>
            </nav>
          </header>

          <router-view />
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.site-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px clamp(20px, 5vw, 64px);
  border-bottom: 1px solid var(--line);
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.brand-glyph {
  align-self: center;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  background: var(--accent);
  color: var(--surface);
  font-family: var(--font-display);
  font-weight: 900;
  font-size: 17px;
  transform: rotate(-4deg);
}

.brand-name {
  font-family: var(--font-display);
  font-weight: 900;
  font-size: 19px;
  letter-spacing: 0.02em;
}

.brand-sep {
  color: var(--ink-3);
}

.brand-sub {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.14em;
  color: var(--ink-2);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* ---------- 创建恢复码 / 恢复码展示 ---------- */
.create-btn {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--accent);
  background: transparent;
  border: 1px dashed var(--accent);
  padding: 7px 14px;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background 0.2s var(--ease-out), color 0.2s var(--ease-out), transform 0.2s var(--ease-out);
}

.create-btn:hover {
  background: var(--accent-soft);
  transform: translateY(-1px);
}

.code-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  max-width: min(320px, 40vw);
  padding: 6px 12px;
  border: 1px dashed var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 12px;
  transition: border-color 0.2s var(--ease-out), color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.code-chip:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-soft);
}

.code-chip-label {
  color: var(--accent);
  white-space: nowrap;
}

.code-chip-id {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ink-2);
}

/* ---------- 恢复历史 ---------- */
.restore-wrap {
  position: relative;
}

.restore-btn {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--ink-2);
  background: transparent;
  border: 1px solid var(--line);
  padding: 7px 14px;
  border-radius: var(--radius);
  cursor: pointer;
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.restore-btn:hover {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

.restore-panel {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: min(320px, 78vw);
  padding: 14px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
  z-index: 30;
}

.restore-panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.restore-title {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.1em;
  color: var(--ink-3);
}

.restore-close {
  border: none;
  background: transparent;
  color: var(--ink-3);
  font-size: 12px;
  cursor: pointer;
  padding: 2px 6px;
}

.restore-close:hover {
  color: var(--accent);
}

.restore-label {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 8px;
}

.restore-input-row {
  display: flex;
  gap: 8px;
}

.restore-input {
  flex: 1;
  min-width: 0;
  padding: 7px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--ink-1);
  font-family: var(--font-mono);
  font-size: 12px;
  outline: none;
  transition: border-color 0.2s var(--ease-out);
}

.restore-input:focus {
  border-color: var(--accent);
}

.restore-input.is-error {
  border-color: var(--danger, #d9534f);
}

.restore-apply {
  padding: 7px 12px;
  border: none;
  border-radius: var(--radius);
  background: var(--accent);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 12px;
  cursor: pointer;
  transition: background 0.2s var(--ease-out);
}

.restore-apply:hover {
  background: var(--accent-deep);
}

.restore-error {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--danger, #d9534f);
}

/* ---------- 其他页眉入口 ---------- */
.add-btn {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  color: #fff;
  background: var(--accent);
  padding: 7px 14px;
  border-radius: var(--radius);
  transition: background 0.2s var(--ease-out), transform 0.2s var(--ease-out);
}

.add-btn:hover {
  background: var(--accent-deep);
  transform: translateY(-1px);
}

.header-link {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--ink-2);
  padding: 6px 12px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.header-link:hover {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}
</style>