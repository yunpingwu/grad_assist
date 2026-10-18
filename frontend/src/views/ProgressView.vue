<script setup lang="ts">
/**
 * 进度页 · 教材解析实时监控
 * 上传后跳转本页并携带 textbook_path，本页发起 POST /resolve 以 SSE 流接收实时进度
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { resolveTextbooks } from '../api/textbook'
import type { ResolveHandlers } from '../api/textbook'
import type { OffsetEntry, TaskEvent, TaskState, TocEventData } from '../api/types'
import { clearIngestTask, getIngestTask, saveIngestTask } from '../utils/ingestTask'
import ProgressRing from '../components/ProgressRing.vue'
import StageChecklist from '../components/StageChecklist.vue'
import TocTree from '../components/TocTree.vue'

const route = useRoute()
const router = useRouter()

/** 从路由 query 取教材目录路径 */
const rawPath = route.query.path
const textbookPath = Array.isArray(rawPath) ? rawPath[0] ?? '' : rawPath ?? ''

/** 恢复进入：仅带 task_id（无 path）时凭 checkpoint 断点续跑 / 重看偏移确认 */
const rawTaskParam = route.query.task_id
const resumeTaskId = Array.isArray(rawTaskParam) ? rawTaskParam[0] ?? '' : rawTaskParam ?? ''

/** 任务状态 / 进度 / 当前消息 */
const status = ref<TaskState>('pending')
const progress = ref(0)
const message = ref('正在建立解析连接…')
const errorMessage = ref('')

/** 续跑用 task_id（后端 info 事件回传） */
const taskId = ref('')
/** 是否停在章节页码偏移确认 */
const awaitingOffset = ref(false)
/** 待确认的偏移条目（可编辑） */
const offsetEntries = ref<OffsetEntry[]>([])
/** 是否正在提交偏移确认 */
const submittingOffset = ref(false)

/** 进度事件日志 */
const logs = ref<TaskEvent[]>([])
const logBody = ref<HTMLElement | null>(null)

/** 右侧面板当前页签：目录树（默认）/ 实时日志 */
const activeTab = ref<'toc' | 'log'>('toc')
/** 已到达的目录树（按教材 upsert） */
const tocBooks = ref<TocEventData[]>([])
/** 续跑时 langgraph 会重放节点代码、原样重发上一条 ask_offset；用上次提交的偏移值签名去重 */
const lastOffsetSignature = ref('')

/** 偏移条目集合的稳定签名（教材名:偏移 排序拼接） */
function offsetSignature(entries: OffsetEntry[]): string {
  return entries
    .map((e) => `${e.textbook_name}:${e.offset}`)
    .sort()
    .join('|')
}

/** 收到 toc 事件：同教材覆盖、新教材追加 */
function upsertToc(data: TocEventData): void {
  const idx = tocBooks.value.findIndex((b) => b.textbook === data.textbook)
  if (idx >= 0) tocBooks.value.splice(idx, 1, data)
  else tocBooks.value.push(data)
}

/** 终态结果（成功 / 失败） */
const finalState = ref<'succeeded' | 'failed' | null>(null)
const isDone = computed(() => finalState.value !== null)

/** 状态中文标签 */
const STATUS_LABEL: Record<TaskState, string> = {
  pending: '等待任务启动',
  running: '解析进行中',
  succeeded: '解析完成',
  failed: '解析失败',
}

/** 日志上限 */
const MAX_LOGS = 200

/** 时间戳格式化（HH:MM:SS） */
function fmtTs(ts: number): string {
  const d = new Date(ts * 1000)
  const p = (n: number): string => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/** 追加日志（去重相邻重复消息）并自动滚动到底部 */
function pushLog(ev: TaskEvent): void {
  const last = logs.value[logs.value.length - 1]
  if (last && last.message === ev.message) return
  logs.value.push(ev)
  if (logs.value.length > MAX_LOGS) logs.value.shift()
}

/** 日志变化后滚动到底部 */
watch(
  () => logs.value.length,
  () => {
    void nextTick(() => {
      if (logBody.value) logBody.value.scrollTop = logBody.value.scrollHeight
    })
  },
)

/** 应用一次进度事件（SSE message） */
function applyEvent(ev: TaskEvent): void {
  status.value = ev.status
  progress.value = ev.progress
  if (ev.message) {
    message.value = ev.message
    pushLog({ ...ev, ts: ev.ts ?? Date.now() / 1000 })
  }
}

/** 构造 SSE 事件回调（初始解析与偏移确认续跑共用） */
function buildHandlers(): ResolveHandlers {
  return {
    onInfo: (d) => {
      taskId.value = d.task_id
      status.value = 'running'
      message.value = d.message
      // 偏移确认已被服务端接受（续跑流首个 info）：收起确认面板；
      // 关闭只由此事件驱动，不能放在提交请求的 finally——ask_offset 也发生在流尾部，
      // finally 会误关刚弹出的下一轮确认面板，且整条流要等解析结束才关闭
      awaitingOffset.value = false
      submittingOffset.value = false
      // 断线重进：后端在首帧附带目录树快照，立即恢复常驻展示
      for (const toc of d.tocs ?? []) upsertToc(toc)
      // 落盘进行中任务：离开本页后可从上传页凭 task_id 找回（恢复进入时沿用已存路径）
      saveIngestTask({ taskId: d.task_id, textbookPath: textbookPath || getIngestTask()?.textbookPath || '' })
      pushLog({ ts: Date.now() / 1000, status: 'running', message: d.message, progress: d.progress ?? 0, data: {} })
    },
    onMessage: (ev) => applyEvent(ev),
    onAskOffset: (d) => {
      // langgraph 续跑会重放节点代码并原样重发上一条 ask_offset：与刚提交的值一致则忽略
      if (offsetSignature(d.offsets ?? []) === lastOffsetSignature.value) return
      offsetEntries.value = (d.offsets ?? []).map((o) => ({ textbook_name: o.textbook_name, offset: o.offset }))
      awaitingOffset.value = true
      message.value = d.message ?? '请确认章节页码偏移'
    },
    onToc: (d) => upsertToc(d),
    onDone: () => {
      finalState.value = 'succeeded'
      status.value = 'succeeded'
      // 解析完成：清除进行中任务，上传页不再展示恢复入口
      clearIngestTask()
    },
    onError: (d) => {
      errorMessage.value = d.message
      finalState.value = 'failed'
      status.value = 'failed'
      pushLog({ ts: Date.now() / 1000, status: 'failed', message: d.message, progress: progress.value, data: {} })
    },
  }
}

/** 发起解析：POST /resolve 以 SSE 流接收实时进度；恢复进入时只带 task_id 续跑 */
async function startResolve(): Promise<void> {
  status.value = 'running'
  try {
    if (!textbookPath && resumeTaskId) {
      await resolveTextbooks('', buildHandlers(), { taskId: resumeTaskId })
    } else {
      await resolveTextbooks(textbookPath, buildHandlers())
    }
  } catch (err) {
    errorMessage.value = (err as Error).message
    finalState.value = 'failed'
    status.value = 'failed'
  }
}

/** 确认偏移后续跑：POST /resolve 带 task_id + offsets；面板收起由续跑流的 info 事件驱动（见 buildHandlers） */
async function confirmOffsets(): Promise<void> {
  submittingOffset.value = true
  lastOffsetSignature.value = offsetSignature(offsetEntries.value)
  try {
    finalState.value = null
    errorMessage.value = ''
    status.value = 'running'
    await resolveTextbooks('', buildHandlers(), { taskId: taskId.value, offsets: offsetEntries.value })
  } catch (err) {
    // 提交失败（HTTP 错误/断连）：面板保留，允许修正后重试
    errorMessage.value = (err as Error).message
    finalState.value = 'failed'
    status.value = 'failed'
  } finally {
    submittingOffset.value = false
  }
}

onMounted(() => {
  if (textbookPath || resumeTaskId) void startResolve()
})

onBeforeUnmount(() => {
  /* 无 EventSource / 轮询需清理 */
})
</script>


<template>
  <main class="progress-page">
    <template v-if="textbookPath || resumeTaskId">
      <!-- 顶栏 -->
      <header class="topbar">
        <button class="back" @click="router.push('/ingest')">← 返回上传</button>
        <span class="mode-badge">{{ isDone ? '已结束' : '解析中' }}</span>
      </header>

      <!-- 状态区 -->
      <section class="status-hero" :class="finalState ?? ''">
        <ProgressRing :progress="progress" :state="status" />
        <div class="status-text">
          <p class="status-label" :class="status">{{ STATUS_LABEL[status] }}</p>
          <h2 class="status-message">{{ message }}</h2>
          <p v-if="errorMessage" class="status-error">{{ errorMessage }}</p>
        </div>
        <div class="status-actions">
          <button v-if="isDone" class="ghost" @click="router.push('/ingest')">继续上传教材</button>
        </div>
      </section>

      <!-- 章节页码偏移确认 -->
      <section v-if="awaitingOffset" class="offset-panel">
        <div class="offset-head">
          <h3 class="offset-title">确认章节页码偏移</h3>
          <span class="offset-rule">OFFSET CHECK</span>
        </div>
        <p class="offset-hint">后端自动计算的偏移量可能偏差较大，请逐本核对并修正（offset = 物理页码 − 印刷页码）。</p>
        <ol class="offset-list">
          <li v-for="item in offsetEntries" :key="item.textbook_name" class="offset-row">
            <span class="offset-name">{{ item.textbook_name }}</span>
            <input v-model.number="item.offset" type="number" step="1" class="offset-input" />
          </li>
        </ol>
        <div class="offset-actions">
          <button class="cta" :disabled="submittingOffset" @click="confirmOffsets">
            {{ submittingOffset ? '提交中…' : '确认并继续解析' }}
          </button>
        </div>
      </section>

      <!-- 阶段 + 日志 -->
      <section class="detail-grid">
        <div class="panel">
          <div class="panel-head">
            <h3 class="panel-title">流水线</h3>
            <span class="panel-rule">PIPELINE</span>
          </div>
          <StageChecklist :progress="progress" :state="status" :final="finalState" />
        </div>

        <div class="panel detail-panel" :class="{ 'log-mode': activeTab === 'log' }">
          <div class="panel-head">
            <div class="panel-tabs" role="tablist" aria-label="面板视图">
              <button
                class="panel-tab"
                :class="{ active: activeTab === 'toc' }"
                role="tab"
                :aria-selected="activeTab === 'toc'"
                @click="activeTab = 'toc'"
              >目录树</button>
              <button
                class="panel-tab"
                :class="{ active: activeTab === 'log' }"
                role="tab"
                :aria-selected="activeTab === 'log'"
                @click="activeTab = 'log'"
              >实时日志</button>
            </div>
            <span class="panel-rule"><i v-if="activeTab === 'log'" class="live-dot" />{{ activeTab === 'log' ? 'LIVE' : 'TOC' }}</span>
          </div>
          <TocTree v-if="activeTab === 'toc'" :books="tocBooks" />
          <ol v-else ref="logBody" class="log-body">
            <li v-for="(log, i) in logs" :key="i" class="log-line" :class="log.status">
              <span class="log-ts">{{ fmtTs(log.ts) }}</span>
              <span class="log-state">{{ log.status }}</span>
              <span class="log-msg">{{ log.message }}</span>
            </li>
            <li v-if="!logs.length" class="log-empty">等待进度事件…</li>
          </ol>
        </div>
      </section>
    </template>

    <!-- 缺少教材路径 -->
    <section v-else class="missing">
      <p class="missing-code">404 · TEXTBOOK</p>
      <h1 class="missing-title">缺少教材路径</h1>
      <p class="missing-desc">请先上传教材并创建解析任务，从上传页进入本页面。</p>
      <button class="cta" @click="router.push('/ingest')">返回上传页 →</button>
    </section>
  </main>
</template>

<style scoped>
.progress-page {
  flex: 1;
  width: min(1180px, 100%);
  margin: 0 auto;
  padding: clamp(20px, 4vw, 44px) clamp(20px, 5vw, 64px) 80px;
  animation: rise 0.6s var(--ease-out) both;
}

/* ---------- 顶栏 ---------- */
.topbar {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  margin-bottom: 28px;
}

.back {
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--ink-2);
  font-size: 12px;
  padding: 8px 14px;
  border-radius: var(--radius);
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out);
}

.back:hover {
  color: var(--accent);
  border-color: var(--accent);
}

.task-chip {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  border: 1px dashed var(--ink-3);
  background: none;
  padding: 8px 14px;
  border-radius: var(--radius);
  transition: border-color 0.2s var(--ease-out);
}

.task-chip:hover {
  border-color: var(--accent);
}

.chip-label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.18em;
  color: var(--accent);
}

.task-chip code {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--ink);
}

.copied {
  font-size: 11px;
  color: var(--ok);
}

.mode-badge {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.16em;
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--ok-soft);
  color: var(--ok);
}

.mode-badge.sse {
  background: var(--ok-soft);
  color: var(--ok);
}

.mode-badge.poll {
  background: var(--accent-soft);
  color: var(--accent);
  animation: blink 1.4s var(--ease-out) infinite;
}

.mode-badge.idle {
  background: var(--paper-deep);
  color: var(--ink-2);
}

/* ---------- 状态区 ---------- */
.status-hero {
  display: flex;
  align-items: center;
  gap: clamp(20px, 4vw, 44px);
  padding: clamp(24px, 4vw, 40px);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  box-shadow: 0 1px 0 rgba(33, 29, 22, 0.04), 0 18px 40px -28px rgba(33, 29, 22, 0.3);
  flex-wrap: wrap;
}

.status-text {
  flex: 1;
  min-width: 240px;
}

.status-label {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
}

.status-label.pending {
  color: var(--ink-3);
}

.status-label.running {
  color: var(--accent);
}

.status-label.succeeded {
  color: var(--ok);
}

.status-label.failed {
  color: var(--bad);
}

.status-message {
  margin-top: 10px;
  font-family: var(--font-display);
  font-weight: 700;
  font-size: clamp(1.15rem, 2.4vw, 1.5rem);
  line-height: 1.6;
}

.status-error {
  margin-top: 10px;
  font-size: 13px;
  color: var(--bad);
}

.status-actions {
  display: flex;
  gap: 10px;
  flex-direction: column;
}

.ghost {
  border: 1px solid var(--line);
  background: none;
  color: var(--ink-2);
  font-size: 12px;
  padding: 9px 16px;
  border-radius: var(--radius);
  white-space: nowrap;
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.ghost:hover:not(:disabled) {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

.ghost:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 终态横幅色调 */
.status-hero.succeeded {
  border-color: rgba(39, 98, 60, 0.4);
  background: linear-gradient(135deg, var(--surface), var(--ok-soft));
}

.status-hero.failed {
  border-color: rgba(163, 39, 31, 0.4);
  background: linear-gradient(135deg, var(--surface), var(--bad-soft));
}

/* ---------- 阶段 + 日志 ---------- */
.detail-grid {
  margin-top: 24px;
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
  gap: 24px;
}

.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 22px;
  display: flex;
  flex-direction: column;
}

.panel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--ink);
  margin-bottom: 8px;
}

.panel-title {
  font-family: var(--font-display);
  font-weight: 900;
  font-size: 16px;
  letter-spacing: 0.06em;
}

.panel-rule {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.24em;
  color: var(--ink-3);
}

.live-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  animation: blink 1.4s var(--ease-out) infinite;
}

/* 右侧面板页签（目录树 / 实时日志） */
.panel-tabs {
  display: inline-flex;
  gap: 4px;
}

.panel-tab {
  border: 1px solid transparent;
  background: none;
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 14px;
  letter-spacing: 0.06em;
  color: var(--ink-3);
  padding: 4px 12px;
  border-radius: var(--radius);
  transition: color 0.2s var(--ease-out), background 0.2s var(--ease-out), border-color 0.2s var(--ease-out);
}

.panel-tab:hover {
  color: var(--accent);
}

.panel-tab.active {
  color: var(--accent);
  background: var(--accent-soft);
  border-color: rgba(194, 59, 34, 0.35);
}

/* 日志页签下沿用墨色终端底 */
.detail-panel.log-mode {
  background: var(--ink);
  border-color: var(--ink);
}

.detail-panel.log-mode .panel-head {
  border-bottom-color: #3a332a;
}

.detail-panel.log-mode .panel-tab {
  color: #b3a998;
}

.detail-panel.log-mode .panel-tab:hover {
  color: #e8825f;
}

.detail-panel.log-mode .panel-tab.active {
  color: #e8825f;
  background: rgba(232, 130, 95, 0.12);
  border-color: rgba(232, 130, 95, 0.4);
}

.detail-panel.log-mode .panel-rule {
  color: #b3a998;
}

.log-body {
  flex: 1;
  min-height: 300px;
  max-height: 430px;
  overflow-y: auto;
  padding: 10px 2px 6px;
  list-style: none;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 2;
}

.log-line {
  display: flex;
  gap: 12px;
  align-items: baseline;
  animation: rise 0.3s var(--ease-out) both;
}

.log-ts {
  color: #a39986;
  flex-shrink: 0;
}

.log-state {
  width: 74px;
  flex-shrink: 0;
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.1em;
}

.log-line.running .log-state {
  color: #e8825f;
}

.log-line.succeeded .log-state,
.log-line.pending .log-state {
  color: #7fbf8a;
}

.log-line.failed .log-state {
  color: #e06a62;
}

.log-msg {
  color: #d8cfba;
  word-break: break-all;
}

.log-empty {
  color: #a39986;
  font-style: italic;
}

/* ---------- 缺少 task_id ---------- */
.missing {
  text-align: center;
  padding: clamp(48px, 10vw, 120px) 20px;
}

.missing-code {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.3em;
  color: var(--accent);
}

.missing-title {
  margin-top: 14px;
  font-family: var(--font-display);
  font-weight: 900;
  font-size: clamp(1.8rem, 4vw, 2.6rem);
}

.missing-desc {
  margin: 14px auto 30px;
  max-width: 42ch;
  color: var(--ink-2);
  font-size: 14px;
  line-height: 1.8;
}

.cta {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  border: none;
  background: var(--accent);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.1em;
  padding: 13px 24px;
  border-radius: var(--radius);
  transition: background 0.25s var(--ease-out), transform 0.25s var(--ease-out);
}

.cta:hover {
  background: var(--accent-deep);
  transform: translateY(-1px);
}

/* ---------- 章节页码偏移确认 ---------- */
.offset-panel {
  margin-top: 24px;
  background: var(--surface);
  border: 1px solid var(--accent);
  border-radius: 4px;
  padding: 22px;
  animation: rise 0.4s var(--ease-out) both;
}

.offset-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--ink);
  margin-bottom: 14px;
}

.offset-title {
  font-family: var(--font-display);
  font-weight: 900;
  font-size: 16px;
  letter-spacing: 0.06em;
}

.offset-rule {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.24em;
  color: var(--accent);
}

.offset-hint {
  color: var(--ink-2);
  font-size: 13px;
  line-height: 1.7;
  margin-bottom: 16px;
}

.offset-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 18px;
}

.offset-row {
  display: flex;
  align-items: center;
  gap: 14px;
}

.offset-name {
  flex: 1;
  min-width: 0;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--ink);
  word-break: break-all;
}

.offset-input {
  width: 110px;
  padding: 8px 10px;
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--ink);
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  text-align: right;
}

.offset-input:focus {
  outline: none;
  border-color: var(--accent);
}

.offset-actions {
  display: flex;
  justify-content: flex-end;
}

/* ---------- 动效 ---------- */
@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(14px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@keyframes blink {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.45;
  }
}

/* ---------- 响应式 ---------- */
@media (max-width: 860px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }

  .status-hero {
    flex-direction: column;
    align-items: flex-start;
  }

  .status-actions {
    flex-direction: row;
  }
}
</style>
