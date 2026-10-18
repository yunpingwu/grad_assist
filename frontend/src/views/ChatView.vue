<script setup lang="ts">
/**
 * 对话页 · 针对一本教材的多轮问答（两栏布局）
 * - 点击教材进入 = 自动新建会话（前端 crypto.randomUUID() 生成 session_id）
 * - 左侧：该教材的历史会话列表（GET /sessions，最近在前），点击任意会话加载历史并继续提问
 * - 右侧：消息流 + 输入框；答案支持 markdown 图片引用（![]() → <img>）与换行
 * - 历史限制：后端只返回最近 50 轮，前端再切片兜底，避免超长会话卡爆渲染
 */
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useDialog, useMessage } from 'naive-ui'
import { deleteSession, fetchHistory, listFiles, listSessions, readProductFile, streamChat } from '../api/chat'
import type { ClarifyQuestion, ProductFile } from '../api/chat'
import type { HistoryMessage, SessionInfo, StepItem } from '../api/types'
import { renderMarkdown } from '../utils/markdown'
import { useModelsStore } from '../stores/models'
import { useStudyStore } from '../stores/study'

const route = useRoute()
const router = useRouter()
const modelsStore = useModelsStore()
const studyStore = useStudyStore()
const message = useMessage()
const dialog = useDialog()

/** 教材完整名（路由 query 传入，与后端过滤/检索精确匹配） */
const textbookName = computed(() => {
  const v = route.query.textbook
  return Array.isArray(v) ? v[0] ?? '' : v ?? ''
})

/** 展示用短名（按下划线或空白分割，去下载后缀） */
const displayName = computed(() => textbookName.value.split(/[_\s]+/)[0].trim() || textbookName.value)

/** 消息流 */
interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
  error?: boolean
  /** 预渲染的 markdown HTML（assistant 消息；流式中由节流定时器更新） */
  html?: string
  /** 处理过程时间线：思考 + 工具结果，按到达顺序穿插（折叠展示在回答上方） */
  steps?: StepItem[]
}
const messages = ref<ChatMsg[]>([])
const isPending = ref(false)
const draft = ref('')
/** 深度检索档：本轮请求传给后端，整轮检索共用（后端 deep 默认 false） */
const deepSearch = ref(false)
/** 当前阶段提示（保留以兼容后续工具进度事件） */
const currentStage = ref('')
/** 写盘确认（agent 要写文件生成资料时弹出） */
interface ConfirmInfo {
  sessionId: string
  message: string
}
const confirmRequest = ref<ConfirmInfo | null>(null)
const isConfirming = ref(false)
/** 澄清反问（任务描述不明确时 agent 追问用户；作答后经 clarify_answers 续跑） */
interface ClarifyPanel {
  sessionId: string
  questions: ClarifyQuestion[]
  error?: string
}
const clarifyPanel = ref<ClarifyPanel | null>(null)
/** 澄清面板中各问题的作答（key = 问题 id） */
const clarifyAnswers = reactive<Record<string, string>>({})
const isClarifying = ref(false)
/** 当前进行中的一轮对话（供续跑阶段复用其 pending 消息与事件回调） */
interface ActiveTurn {
  pending: ChatMsg
  handlers: import('../api/chat').StreamHandlers
}
const activeTurn = ref<ActiveTurn | null>(null)
/** 流式渲染节流定时器：每 ~60ms 重渲染一次 markdown，避免逐 token 全量解析 */
let renderTimer: number | undefined

/** 节流渲染：单轮内多次调用被合并为每 ~60ms 一次 */
function scheduleRender(pending: ChatMsg): void {
  if (renderTimer === undefined) {
    renderTimer = window.setTimeout(() => {
      renderTimer = undefined
      pending.html = renderMarkdown(pending.content)
    }, 60)
  }
}

/** 立即终止并执行最后一次完整渲染（收尾用） */
function flushRender(pending: ChatMsg): void {
  if (renderTimer !== undefined) {
    window.clearTimeout(renderTimer)
    renderTimer = undefined
  }
  pending.html = renderMarkdown(pending.content)
}

/** 左侧会话列表 */
const sessions = ref<SessionInfo[]>([])
const isListLoading = ref(false)
const listError = ref('')

/** 最左产物文件列：当前对话生成的资料文件（GET /study/files） */
const products = ref<ProductFile[]>([])
const isProductsLoading = ref(false)
const productsError = ref('')
/** 正在预览的文件相对路径 + 内容 */
const previewPath = ref('')
const previewContent = ref('')
const previewLoading = ref(false)
/** 当前会话 ID（新会话在进入页面时即生成） */
const currentSessionId = ref('')
/** 加载历史中（切换会话时的短暂状态） */
const isHistoryLoading = ref(false)
/** 正在删除的会话 ID（防重复点击，同时仅允许删一个） */
const deletingId = ref('')
/** 产物列是否折叠 */
const productsCollapsed = ref(false)
/** 自动刷新定时器（产物在流式生成中逐步产出，需定时拉取最新列表） */
let productsPollingTimer: number | undefined

/** 模型选择（本地偏好，来自 Pinia models store）：切换/新增模型；activeModelId 为 null 表示服务端默认 */
const modelMenuOpen = ref(false)
/** 新增模型表单 */
const newLabel = ref('')
const newModelName = ref('')
const newBaseUrl = ref('')
const newApiKey = ref('')
const modelError = ref('')

/** 当前模型展示名（无自定义模型显示「默认模型」） */
const activeModelLabel = computed(() => modelsStore.activeLabel)

/** 当前选中的模型实体（null = 服务端默认，请求时不带模型字段） */
function currentModel(): import('../api/types').ModelConfig | undefined {
  return modelsStore.activeModel ?? undefined
}

/** 消息区 / 输入框引用 */
const messageBody = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLTextAreaElement | null>(null)

/** 欢迎语（空会话时展示） */
const WELCOME = '你好，我是这本教材的学习助手。你可以问我书里的任何内容、连续追问，也可以让我基于教材生成复习资料（如「帮我生成一份考点总结」）。'

/** 滚动到底部 */
async function scrollToBottom(): Promise<void> {
  await nextTick()
  if (messageBody.value) messageBody.value.scrollTop = messageBody.value.scrollHeight
}

/** 生成新会话 ID */
function newSessionId(): string {
  return crypto.randomUUID()
}

/** 刷新左侧会话列表（新会话发送后 / 进入页面时） */
async function loadSessions(): Promise<void> {
  if (!textbookName.value) return
  isListLoading.value = true
  listError.value = ''
  try {
    sessions.value = await listSessions(textbookName.value)
  } catch (err) {
    listError.value = (err as Error).message
  } finally {
    isListLoading.value = false
  }
}

/** 刷新产物文件列（当前会话对应 task；无会话或已折叠时不请求） */
async function loadProducts(): Promise<void> {
  if (!currentSessionId.value || !textbookName.value || productsCollapsed.value) return
  isProductsLoading.value = true
  productsError.value = ''
  try {
    products.value = await listFiles(currentSessionId.value, textbookName.value)
    if (previewPath.value && !products.value.some((p) => p.rel_path === previewPath.value)) {
      previewPath.value = ''
      previewContent.value = ''
    }
  } catch (err) {
    productsError.value = (err as Error).message
  } finally {
    isProductsLoading.value = false
  }
}

/** 开启产物轮询：流式对话中产物逐步生成/更新，每 5s 拉一次最新列表 */
function startProductsPolling(): void {
  if (productsPollingTimer !== undefined) return
  productsPollingTimer = window.setInterval(() => void loadProducts(), 5000)
}

/** 折叠/展开产物列：折叠时同时暂停轮询，展开时立即拉一次并恢复轮询 */
function toggleProducts(): void {
  productsCollapsed.value = !productsCollapsed.value
  if (productsCollapsed.value) {
    window.clearInterval(productsPollingTimer)
    productsPollingTimer = undefined
  } else {
    void loadProducts()
    startProductsPolling()
  }
}

/** 点击产物文件：读取内容预览（再次点击同文件收起） */
async function openPreview(p: ProductFile): Promise<void> {
  if (previewPath.value === p.rel_path) {
    previewPath.value = ''
    previewContent.value = ''
    return
  }
  previewPath.value = p.rel_path
  previewContent.value = ''
  previewLoading.value = true
  try {
    previewContent.value = await readProductFile(currentSessionId.value, textbookName.value, p.rel_path)
  } catch (err) {
    previewContent.value = `（读取失败：${(err as Error).message}）`
  } finally {
    previewLoading.value = false
  }
}

/** 开始一个全新会话（空历史） */
function startNewSession(): void {
  currentSessionId.value = newSessionId()
  messages.value = [{ role: 'assistant', content: WELCOME, html: renderMarkdown(WELCOME) }]
  products.value = []
  previewPath.value = ''
  previewContent.value = ''
  isPending.value = false
  void scrollToBottom()
  inputEl.value?.focus()
}

/** 删除单个历史会话（垃圾桶按钮）；先用确认弹窗二次确认，删的是当前会话则重置为新会话 */
function removeSession(sessionId: string): void {
  if (deletingId.value) return
  dialog.warning({
    title: '删除会话',
    content: '删除后无法恢复，确定删除该会话吗？',
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: () => {
      void doRemoveSession(sessionId)
    },
  })
}

/** 确认后真正执行删除 */
async function doRemoveSession(sessionId: string): Promise<void> {
  deletingId.value = sessionId
  listError.value = ''
  try {
    await deleteSession(sessionId)
    sessions.value = sessions.value.filter((s) => s.session_id !== sessionId)
    if (sessionId === currentSessionId.value) startNewSession()
    message.success('会话已删除')
  } catch (err) {
    listError.value = (err as Error).message
    message.error((err as Error).message)
  } finally {
    deletingId.value = ''
  }
}

/** 打开一个历史会话：加载最近消息并继续提问 */
async function openSession(sessionId: string): Promise<void> {
  if (sessionId === currentSessionId.value) return
  currentSessionId.value = sessionId
  isHistoryLoading.value = true
  try {
    const history = await fetchHistory(sessionId)
    // 后端已限制 50 轮，前端再切片兜底；历史无欢迎语；assistant 消息预渲染 markdown
    messages.value = history
      .slice(-100)
      .map(
        (m: HistoryMessage): ChatMsg => ({
          role: m.role,
          content: m.content,
          error: false,
          html: m.role === 'assistant' ? renderMarkdown(m.content) : undefined,
          steps: m.role === 'assistant' ? (m.steps ?? toLegacySteps(m.tools)) : [],
        }),
      )
    if (!messages.value.length) messages.value.push({ role: 'assistant', content: WELCOME, html: renderMarkdown(WELCOME) })
  } catch (err) {
    messages.value = [{ role: 'assistant', content: (err as Error).message, error: true }]
  } finally {
    isHistoryLoading.value = false
    await scrollToBottom()
    inputEl.value?.focus()
    void loadProducts()
  }
}

/** 兼容旧接口（只有 tools 无 steps）时把工具结果平铺成时间线 */
function toLegacySteps(tools?: Array<{ name: string; content: string }>): StepItem[] {
  return (tools ?? []).map((t) => ({ type: 'tool' as const, name: t.name, content: t.content }))
}

/** 图片加载失败时隐藏（不再显示裂图图标），并补一个占位提示。
 *  绑定在消息容器上（@error.capture）：error 事件不冒泡但可被捕获阶段捕获，
 *  因此能覆盖 v-html 渲染出的 <img>，且无需内联 onerror（防 XSS 面）。 */
function handleImgError(e: Event): void {
  const el = e.target as HTMLImageElement
  el.style.display = 'none'
  const tip = document.createElement('span')
  tip.className = 'msg-img-fallback'
  tip.textContent = '(图片加载失败)'
  el.after(tip)
}

/** 结束当前轮：收尾渲染、复位状态、允许聚焦输入 */
async function finishTurn(): Promise<void> {
  const turn = activeTurn.value
  if (turn) flushRender(turn.pending) // 取消节流定时器并做最后一次完整渲染
  isPending.value = false
  currentStage.value = ''
  confirmRequest.value = null
  isConfirming.value = false
  clarifyPanel.value = null
  isClarifying.value = false
  activeTurn.value = null
  await scrollToBottom()
  inputEl.value?.focus()
  void loadProducts() // 本轮可能刚生成/修改了资料文件，刷新产物列
}

/** 处理一轮 chat（含后续续跑）共用的推理逻辑 */
function buildHandlers(pending: ChatMsg) {
  return {
    onStage: (_stage: string, message: string) => {
      currentStage.value = message
    },
    onToken: (content: string) => {
      pending.content += content
      scheduleRender(pending)
      void scrollToBottom()
    },
    onDone: (sessionId: string) => {
      currentSessionId.value = sessionId
      currentStage.value = ''
      // 新会话首轮后入列 / 已有会话 preview 更新
      void loadSessions()
    },
    onError: (message: string) => {
      pending.content = message
      pending.error = true
      currentStage.value = ''
    },
    onNeedConfirm: (info: { sessionId: string; message: string }) => {
      // 停在写盘确认：保持 isPending，弹出确认条，等待用户允许/拒绝后续跑
      confirmRequest.value = { sessionId: info.sessionId, message: info.message }
      isConfirming.value = false
    },
    onClarify: (info: { sessionId: string; questions: ClarifyQuestion[]; maxQuestions: number; error?: string }) => {
      // 停在澄清反问：弹出作答面板，等待用户作答（或直接输入新问题取消本轮澄清）
      clarifyPanel.value = { sessionId: info.sessionId, questions: info.questions, error: info.error || undefined }
      // 校验失败重问时清空旧作答，按错误提示重填
      Object.keys(clarifyAnswers).forEach((k) => delete clarifyAnswers[k])
      isClarifying.value = false
      void scrollToBottom()
    },
    // 思考与工具结果按到达顺序推入同一时间线：流式思考分片合并进上一条 thought，
    // 工具结果单独成步，前端按顺序渲染即可实现"思考→检索→思考→回答"穿插
    onThought: (content: string) => {
      if (!content) return
      const steps = (pending.steps ??= [])
      const last = steps[steps.length - 1]
      if (last && last.type === 'thought') last.content += content
      else steps.push({ type: 'thought', content })
    },
    onTool: (info: { status: 'start' | 'done'; name: string; id?: string; content?: string }) => {
      // 只承接已完成的结果（后端只给 search_textbook 推 done），不再出现"运行中…"占位
      if (info.status === 'done' && info.content) {
        const steps = (pending.steps ??= [])
        steps.push({ type: 'tool', name: info.name, content: info.content })
      }
    },
  }
}

/** 发送一轮对话（SSE 流式：答案打字机 + 里程碑可确认写文件/回答澄清） */
async function send(): Promise<void> {
  const query = draft.value.trim()
  if (!query || !textbookName.value) return
  // 正常执行中禁止打断；停在澄清反问时允许直接输入新问题（后端自动取消本轮澄清再跑新问题）
  if (isPending.value && !clarifyPanel.value) return

  // 停在澄清反问却发新问题：清掉旧轮空占位，作为全新一轮发送
  if (clarifyPanel.value) {
    const turn = activeTurn.value
    if (turn && !turn.pending.content) {
      const idx = messages.value.indexOf(turn.pending)
      if (idx >= 0) messages.value.splice(idx, 1)
    }
    activeTurn.value = null
    clarifyPanel.value = null
    isClarifying.value = false
  }

  draft.value = ''
  messages.value.push({ role: 'user', content: query })
  // 占位消息：reactive 包装，token 逐字追加到 content（打字机效果）
  const pending = reactive<ChatMsg>({ role: 'assistant', content: '', html: '' })
  messages.value.push(pending)
  const handlers = buildHandlers(pending)
  activeTurn.value = { pending, handlers }
  isPending.value = true
  currentStage.value = ''
  confirmRequest.value = null
  isConfirming.value = false
  await scrollToBottom()

  try {
    await streamChat(
      textbookName.value,
      query,
      currentSessionId.value || undefined,
      handlers,
      undefined,
      currentModel(),
      undefined,
      deepSearch.value,
    )
    // 若因写盘确认/澄清反问被中断，后端发对应事件后即结束本流；此处不结束，等待用户回应
    if (!confirmRequest.value && !clarifyPanel.value) await finishTurn()
  } catch (err) {
    pending.content = (err as Error).message
    pending.error = true
    await finishTurn()
  }
}

/** 提交澄清作答：收集各问题答案（选项 method 或自定义文本）经 clarify_answers 续跑 */
async function submitClarify(): Promise<void> {
  const panel = clarifyPanel.value
  const turn = activeTurn.value
  if (!panel || !turn || isClarifying.value) return

  const answers = panel.questions.map((q) => ({ id: q.id, answer: clarifyAnswers[q.id]?.trim() ?? '' }))
  if (answers.some((a) => !a.answer)) {
    message.warning('请完成所有澄清问题后再提交')
    return
  }
  isClarifying.value = true
  clarifyPanel.value = null // 立即隐藏面板：一次提交即进入处理中，不再残留
  try {
    await streamChat(textbookName.value, '', panel.sessionId, turn.handlers, undefined, currentModel(), answers)
    // 续跑途中可能再弹新澄清/写盘确认；均无则正常收尾
    if (!confirmRequest.value && !clarifyPanel.value) await finishTurn()
  } catch (err) {
    turn.pending.content = (err as Error).message
    turn.pending.error = true
    await finishTurn()
  }
}

/** 写盘确认的续跑：以 decision 走 /study/chat 续跑同一线程（已无独立 resume 端点） */
async function resumeTurn(approved: boolean): Promise<void> {
  const turn = activeTurn.value
  const sessionId = confirmRequest.value?.sessionId
  if (!turn || !sessionId || isConfirming.value) return

  isConfirming.value = true
  confirmRequest.value = null // 立即隐藏确认条：一次点击即进入处理中，不再残留
  try {
    await streamChat(textbookName.value, '', sessionId, turn.handlers, approved ? 'approve' : 'reject', currentModel())
    // 后端已对全部 pending 中断统一生效；若续跑途中又发起新一轮写盘会产生新 ask_confirm，挂起等下次确认
    if (!confirmRequest.value) await finishTurn()
  } catch (err) {
    turn.pending.content = (err as Error).message
    turn.pending.error = true
    await finishTurn()
  }
}

/** Enter 发送，Shift+Enter 换行 */
function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    void send()
  }
}

/** 文件大小人类可读（B / KB / MB） */
function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

/** 会话更新时间（ISO → HH:MM，跨天显示 MM-DD） */
function fmtTime(iso: string | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const p = (n: number): string => String(n).padStart(2, '0')
  const today = new Date()
  const sameDay = d.toDateString() === today.toDateString()
  return sameDay ? `${p(d.getHours())}:${p(d.getMinutes())}` : `${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

/** 会话列表预览：最新用户问题，截断到 60 字符（展示整形归前端） */
function previewOf(s: SessionInfo): string {
  const q = (s.last_question || '').trim()
  if (!q) return ''
  return `我: ${q.length > 60 ? q.slice(0, 60) + '…' : q}`
}

/** 切换模型：写入本地偏好并更新状态；传 null 回到服务端默认 */
function selectModel(id: string | null): void {
  modelsStore.select(id)
  modelMenuOpen.value = false
}

/** 删除一个自定义模型（删当前选中则自动回到服务端默认） */
function removeModelLocal(id: string): void {
  modelsStore.remove(id)
}

/** 点击菜单外部关闭（按钮/菜单自身 @click.stop 阻断冒泡） */
function onDocClick(): void {
  modelMenuOpen.value = false
}

/** 新增模型：校验后落库并立即选中，关闭菜单 */
function submitNewModel(): void {
  const label = newLabel.value.trim()
  const model = newModelName.value.trim()
  if (!label || !model) {
    modelError.value = '请至少填写名称与模型名'
    return
  }
  modelsStore.add({
    label,
    model,
    base_url: newBaseUrl.value.trim(),
    api_key: newApiKey.value.trim(),
  })
  newLabel.value = ''
  newModelName.value = ''
  newBaseUrl.value = ''
  newApiKey.value = ''
  modelError.value = ''
  modelMenuOpen.value = false
}

onMounted(() => {
  // 锁定视口：整页不随对话增长，仅消息区滚动（离开页面时解锁，见 onBeforeUnmount）
  document.body.classList.add('chat-locked')
  modelsStore.init() // 加载模型偏好（若尚未在应用挂载时初始化）
  studyStore.setTextbook(textbookName.value) // 记录当前教材，供全局引用
  startNewSession() // 点击教材进入 = 自动新会话
  void loadSessions() // 左侧列表自动出现
  startProductsPolling() // 产物轮询，流式生成中自动刷新
  document.addEventListener('click', onDocClick) // 点模型菜单外部时收起
})

onBeforeUnmount(() => {
  document.body.classList.remove('chat-locked')
  if (renderTimer !== undefined) {
    window.clearTimeout(renderTimer)
    renderTimer = undefined
  }
  window.clearInterval(productsPollingTimer)
  productsPollingTimer = undefined
  document.removeEventListener('click', onDocClick)
})
</script>

<template>
  <main class="chat-page">
    <!-- 顶栏 -->
    <header class="topbar">
      <button class="back" @click="router.push('/')">← 返回书架</button>
      <h1 class="book-title" :title="textbookName">{{ displayName }}</h1>
      <button class="new-session" @click="startNewSession">＋ 新建会话</button>
    </header>

    <!-- 三栏主体：会话列表 → 对话区 → 最右产物文件列（当前对话生成资料） -->
    <div class="layout">
      <!-- 左侧：会话列表 -->
      <aside class="session-list" aria-label="会话列表">
        <p class="list-head">
          会话
          <span class="list-count" v-if="sessions.length">{{ sessions.length }}</span>
        </p>
        <p v-if="listError" class="list-error">{{ listError }}</p>
        <p v-else-if="isListLoading" class="list-empty">加载中…</p>
        <p v-else-if="!sessions.length" class="list-empty">还没有历史会话，发一条消息开始吧</p>
        <ul v-else class="session-items">
          <li
            v-for="s in sessions"
            :key="s.session_id"
            class="session-row"
            :class="{ active: s.session_id === currentSessionId }"
          >
            <button class="session-item" :title="previewOf(s)" @click="openSession(s.session_id)">
              <span class="session-preview">{{ previewOf(s) }}</span>
              <span class="session-meta">
                <span class="session-time">{{ fmtTime(s.updated_at) }}</span>
                <span class="session-count">{{ s.message_count }} 条</span>
              </span>
            </button>
            <button
              class="trash"
              :disabled="deletingId === s.session_id"
              :title="'删除会话'"
              aria-label="删除会话"
              @click.stop="removeSession(s.session_id)"
            >
              <svg
                viewBox="0 0 24 24"
                width="14"
                height="14"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
              >
                <path d="M3 6h18" />
                <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                <path d="M10 11v6" />
                <path d="M14 11v6" />
              </svg>
            </button>
          </li>
        </ul>
      </aside>

      <!-- 右侧：对话区 -->
      <section class="chat-main">
        <div ref="messageBody" class="message-body" aria-live="polite" @error.capture="handleImgError">
          <p v-if="isHistoryLoading" class="history-loading">正在加载历史会话…</p>
          <div v-for="(msg, i) in messages" :key="i" class="message" :class="msg.role">
            <span class="msg-label">{{ msg.role === 'user' ? '我' : '助手' }}</span>
            <div class="msg-bubble" :class="{ error: msg.error }">
              <template v-if="msg.role === 'assistant' && !msg.error">
                <!-- 处理过程时间线：思考与工具结果穿插，折叠展示在回答上方 -->
                <details v-if="msg.steps && msg.steps.length" class="tool-log">
                  <summary>处理过程（{{ msg.steps.length }}）</summary>
                  <div v-for="(s, si) in msg.steps" :key="si" class="step-item">
                    <code v-if="s.type === 'tool'" class="tool-name">{{ s.name }}</code>
                    <span v-else class="step-tag">思考</span>
                    <pre class="tool-result">{{ s.content }}</pre>
                  </div>
                </details>
                <div class="markdown-body" v-html="msg.html ?? renderMarkdown(msg.content)"></div>
                <span v-if="msg.content === ''" class="thinking">
                  <i class="tdot" /><i class="tdot" /><i class="tdot" />
                </span>
              </template>
              <span v-else class="msg-text">{{ msg.content }}</span>
            </div>
          </div>
        </div>

        <!-- 阶段提示（SSE stage 事件） -->
        <p v-if="currentStage" class="stage-indicator" aria-live="polite">{{ currentStage }}</p>
        <!-- 澄清反问面板（agent 认为任务信息不足、需人工补充时） -->
        <div v-if="clarifyPanel" class="clarify-panel" role="dialog" aria-live="polite">
          <p class="clarify-title"><span class="clarify-title-dot" />助手需要先确认一下</p>
          <p v-if="clarifyPanel.error" class="clarify-error">{{ clarifyPanel.error }}</p>
          <div v-for="q in clarifyPanel.questions" :key="q.id" class="clarify-question">
            <p class="clarify-q-text">{{ q.question }}</p>
            <div v-if="q.type === 'choice' && q.options && q.options.length" class="clarify-options">
              <label
                v-for="opt in q.options"
                :key="`${q.id}-${opt.label}`"
                class="clarify-option"
                :class="{ active: clarifyAnswers[q.id] === opt.method }"
              >
                <input v-model="clarifyAnswers[q.id]" type="radio" :name="'clarify-' + q.id" :value="opt.method" />
                <span class="clarify-option-text">{{ opt.label }}</span>
              </label>
            </div>
            <div v-else class="clarify-open">
              <textarea
                v-model="clarifyAnswers[q.id]"
                class="clarify-input"
                rows="2"
                :placeholder="q.hint || '输入你的回答…'"
              />
            </div>
          </div>
          <div class="clarify-actions">
            <span class="clarify-tip">也可以直接在下方输入框发送新问题，自动取消本轮澄清</span>
            <button class="confirm-btn approve" :disabled="isClarifying" @click="submitClarify">
              {{ isClarifying ? '处理中…' : '提交回答' }}
            </button>
          </div>
        </div>
        <!-- 写盘确认条（agent 生成资料要写文件时） -->
        <div v-if="confirmRequest" class="confirm-bar" role="dialog" aria-live="polite">
          <span class="confirm-msg">{{ confirmRequest.message }}</span>
          <div class="confirm-actions">
            <button class="confirm-btn reject" :disabled="isConfirming" @click="resumeTurn(false)">拒绝</button>
            <button class="confirm-btn approve" :disabled="isConfirming" @click="resumeTurn(true)">允许</button>
          </div>
        </div>
        <footer class="composer">
          <textarea
            ref="inputEl"
            v-model="draft"
            class="input"
            rows="2"
            :placeholder="clarifyPanel ? '也可直接输入新问题，自动取消本轮澄清' : `关于《${displayName}》，想问点什么？`"
            :disabled="(isPending && !clarifyPanel) || isHistoryLoading"
            @keydown="onKeydown"
          />
          <!-- 输入容器底部工具行：左侧模型选择 + 右侧发送 -->
          <div class="composer-toolbar">
            <div class="model-picker">
              <button
                class="model-btn"
                type="button"
                :class="{ 'is-open': modelMenuOpen }"
                :title="`当前模型：${activeModelLabel}`"
                @click.stop="modelMenuOpen = !modelMenuOpen"
              >
                <span class="model-dot" aria-hidden="true" />
                <span class="model-label">{{ activeModelLabel }}</span>
                <span class="model-caret" aria-hidden="true">▾</span>
              </button>

              <!-- 模型选择 / 新增弹层 -->
              <div v-if="modelMenuOpen" class="model-menu" @click.stop>
                <div class="model-menu-head">
                  <span class="model-menu-title">选择模型</span>
                  <button class="model-menu-close" type="button" aria-label="关闭" @click="modelMenuOpen = false">✕</button>
                </div>

                <button class="model-option" type="button" :class="{ active: modelsStore.activeModelId === null }" @click="selectModel(null)">
                  <span class="model-option-label">默认模型</span>
                  <span class="model-option-sub">使用服务端配置</span>
                </button>

                <button
                  v-for="m in modelsStore.models"
                  :key="m.id"
                  class="model-option"
                  type="button"
                  :class="{ active: modelsStore.activeModelId === m.id }"
                  @click="selectModel(m.id)"
                >
                  <span class="model-option-main">
                    <span class="model-option-label">{{ m.label }}</span>
                    <span class="model-option-sub">{{ m.model }}</span>
                  </span>
                  <span class="model-option-del" title="删除此模型" @click.stop="removeModelLocal(m.id)">✕</span>
                </button>

                <div class="model-menu-divider" />

                <div class="model-form">
                  <label class="model-field">
                    <span>名称</span>
                    <input v-model="newLabel" type="text" placeholder="如：DeepSeek 主力" />
                  </label>
                  <label class="model-field">
                    <span>模型名</span>
                    <input v-model="newModelName" type="text" placeholder="如：deepseek-v4-flash" />
                  </label>
                  <label class="model-field">
                    <span>兼容地址</span>
                    <input v-model="newBaseUrl" type="text" placeholder="https://…/v1（可选）" />
                  </label>
                  <label class="model-field">
                    <span>密钥</span>
                    <input v-model="newApiKey" type="password" placeholder="API Key（可选）" />
                  </label>
                  <p v-if="modelError" class="model-error">{{ modelError }}</p>
                  <button class="model-submit" type="button" @click="submitNewModel">新增并切换</button>
                </div>
              </div>
            </div>

            <!-- 深度检索档：本轮所有教材检索走 HyDE+精排，每次检索多 ~5s，默认关 -->
            <button
              class="deep-toggle"
              type="button"
              :class="{ 'is-on': deepSearch }"
              :disabled="isPending"
              :title="deepSearch ? '本轮检索走深度档（更准，每次检索多约 5 秒），点击切回快检索' : '本轮检索走快档，点击开启深度档（更准，每次检索多约 5 秒）'"
              @click="deepSearch = !deepSearch"
            >
              <span class="deep-dot" aria-hidden="true" />
              <span>深度检索</span>
            </button>

            <button class="send" :disabled="!draft.trim() || (isPending && !clarifyPanel) || isHistoryLoading" @click="send">
              {{ isPending && !clarifyPanel ? '思考中' : '发送' }}
              <span class="send-arrow" aria-hidden="true">→</span>
            </button>
          </div>
        </footer>
      </section>

      <!-- 最右：产物文件列（可折叠） -->
      <aside class="product-list" :class="{ collapsed: productsCollapsed }" aria-label="产物文件">
        <p class="list-head">
          <span class="head-title">产物文件</span>
          <span class="list-count" v-if="products.length">{{ products.length }}</span>
          <button class="refresh-link" :disabled="isProductsLoading" title="刷新" @click="loadProducts">↻</button>
          <button
            class="collapse-btn"
            :title="productsCollapsed ? '展开产物列' : '折叠产物列'"
            :aria-label="productsCollapsed ? '展开产物列' : '折叠产物列'"
            @click="toggleProducts"
          >
            {{ productsCollapsed ? '‹' : '›' }}
          </button>
        </p>
        <div v-show="!productsCollapsed" class="product-body">
          <p v-if="productsError" class="list-error">{{ productsError }}</p>
          <p v-else-if="isProductsLoading" class="list-empty">加载中…</p>
          <p v-else-if="!products.length" class="list-empty">当前对话还没生成资料文件。对助手说「帮我生成一份考点总结」，落盘后这里会列出。</p>
          <ul v-else class="session-items">
            <li v-for="p in products" :key="p.rel_path">
              <button
                class="session-item product-item"
                :class="{ active: previewPath === p.rel_path }"
                :title="p.rel_path"
                @click="openPreview(p)"
              >
                <span class="session-preview">{{ p.rel_path }}</span>
                <span class="session-meta">
                  <span>{{ fmtSize(p.size) }}</span>
                </span>
              </button>
            </li>
          </ul>
          <!-- 文件内容预览（点文件展开/收起） -->
          <div v-if="previewPath" class="product-preview">
            <p class="preview-head">
              <span class="preview-path">{{ previewPath }}</span>
              <button class="preview-close" title="收起" @click="previewPath = ''; previewContent = ''">✕</button>
            </p>
            <pre class="preview-body">{{ previewLoading ? '读取中…' : previewContent }}</pre>
          </div>
        </div>
      </aside>
    </div>
  </main>
</template>

<style scoped>
.chat-page {
  flex: 1;
  width: 98%; /* 左右各留 1% */
  max-width: none;
  margin: 0 auto;
  padding: clamp(20px, 4vw, 40px) 0 32px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  animation: rise 0.6s var(--ease-out) both;
}

/* ---------- 顶栏 ---------- */
.topbar {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  padding: 0 clamp(16px, 3vw, 40px) 18px;
  border-bottom: 2px solid var(--ink);
  margin-bottom: 20px;
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

.book-title {
  font-family: var(--font-display);
  font-weight: 900;
  font-size: clamp(1.4rem, 3.4vw, 1.9rem);
  letter-spacing: 0.04em;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.new-session {
  margin-left: auto;
  border: 1px solid var(--accent);
  background: none;
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  padding: 8px 14px;
  border-radius: var(--radius);
  transition: color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.new-session:hover {
  background: var(--accent-soft);
}

/* ---------- 两栏布局 ---------- */
.layout {
  flex: 1;
  min-height: 0;
  display: flex;
  gap: 24px;
}

/* ---------- 最右产物文件列 ---------- */
.product-list {
  width: 240px;
  flex-shrink: 0;
  border-left: 1px solid var(--line);
  padding-left: 14px;
  display: flex;
  flex-direction: column;
  transition: width 0.25s var(--ease-out);
}

.product-list.collapsed {
  width: 28px;
  border-left: none;
  padding-left: 0;
  overflow: hidden;
}

/* 折叠态只保留展开按钮（收窄竖条）；标题/计数/刷新/列表/预览全部隐藏 */
.product-list.collapsed .head-title,
.product-list.collapsed .list-count,
.product-list.collapsed .refresh-link,
.product-list.collapsed .product-body {
  display: none;
}

.product-list.collapsed .list-head {
  justify-content: center;
  padding: 8px 0;
  border-bottom: none;
  margin-bottom: 0;
}

.product-list.collapsed .collapse-btn {
  margin-left: 0;
}

.refresh-link {
  background: none;
  border: none;
  color: var(--ink-3);
  font-size: 14px;
  line-height: 1;
  cursor: pointer;
}

.refresh-link:hover:not(:disabled) {
  color: var(--accent);
}

.refresh-link:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.collapse-btn {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--ink-2);
  font-size: 14px;
  line-height: 1;
  cursor: pointer;
  padding: 0;
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.collapse-btn:hover {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

.product-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto; /* 列表自身滚动，标题固定在顶部 */
  display: flex;
  flex-direction: column;
}

.product-item.active {
  background: var(--accent-soft);
}

.product-preview {
  margin-top: 12px;
  border-top: 1px dashed var(--line);
  padding-top: 8px;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.preview-head {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 6px;
}

.preview-path {
  flex: 1;
  min-width: 0;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.preview-close {
  background: none;
  border: none;
  color: var(--ink-3);
  cursor: pointer;
  font-size: 12px;
  padding: 0 2px;
}

.preview-close:hover {
  color: var(--bad);
}

.preview-body {
  margin: 0;
  padding: 8px 10px;
  max-height: 320px;
  overflow: auto;
  background: var(--paper-deep);
  border: 1px solid var(--line);
  border-radius: 4px;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--ink-2);
}

/* ---------- 左侧会话列表 ---------- */
.session-list {
  width: 240px;
  flex-shrink: 0;
  border-right: 1px solid var(--line);
  padding-right: 14px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.list-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.24em;
  color: var(--ink-3);
  padding-bottom: 10px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 10px;
}

.list-count {
  font-size: 10px;
  color: var(--accent);
}

.list-empty {
  font-size: 12px;
  line-height: 1.8;
  color: var(--ink-3);
  padding: 8px 0;
}

.list-error {
  font-size: 12px;
  color: var(--bad);
  padding: 8px 0;
}

.session-items {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

/* 行：会话内容 + 垃圾桶按钮并排，active 高亮整行 */
.session-row {
  display: flex;
  align-items: stretch;
  border-left: 2px solid transparent;
  border-radius: 2px;
  transition: background 0.2s var(--ease-out), border-color 0.2s var(--ease-out);
}

.session-row:hover {
  background: var(--paper-deep);
}

.session-row.active {
  background: var(--accent-soft);
  border-left-color: var(--accent);
}

.session-item {
  flex: 1;
  min-width: 0;
  text-align: left;
  background: none;
  border: none;
  padding: 10px 4px 10px 10px;
}

.session-preview {
  display: block;
  font-size: 15px;
  line-height: 1.6;
  color: var(--ink);
  /* 单行截断：一行放不下直接省略，不占两行，给主对话留空间 */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-row.active .session-preview {
  color: var(--accent-deep);
}

/* 垃圾桶按钮（默认弱化，hover 变红） */
.trash {
  flex-shrink: 0;
  align-self: center;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  margin-right: 6px;
  background: none;
  border: none;
  border-radius: 2px;
  color: var(--ink-3);
  opacity: 0.35;
  transition: color 0.2s var(--ease-out), opacity 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.trash:hover:not(:disabled) {
  opacity: 1;
  color: var(--bad);
  background: var(--bad-soft);
}

.trash:disabled {
  opacity: 0.2;
  cursor: not-allowed;
}

.session-meta {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-top: 6px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--ink-3);
}

/* ---------- 右侧对话区 ---------- */
.chat-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.history-loading {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
  padding: 8px 4px;
}

.message-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 8px 4px 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.message {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  animation: rise 0.35s var(--ease-out) both;
}

.message.user {
  flex-direction: row-reverse;
}

.msg-label {
  flex-shrink: 0;
  margin-top: 8px;
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.2em;
  color: var(--ink-3);
  writing-mode: vertical-rl;
}

.message.user .msg-label {
  color: var(--accent);
}

.msg-bubble {
  /* min-width:0：flex 子项默认 min-width:auto，会被图片固有宽度撑开导致 max-width 失效溢出；
     设 0 允许气泡按容器收缩，图片 max-width:100% 才能正确生效 */
  min-width: 0;
  /* 兜底：任何内容（图片/长串）都不越出气泡边界 */
  overflow: hidden;
  max-width: min(80%, 760px);
  padding: 16px 20px;
  border-radius: 4px;
  line-height: 1.85;
  font-size: 17px;
  word-break: break-word;
}

.message.assistant .msg-bubble {
  background: var(--surface);
  border: 1px solid var(--line);
  box-shadow: 0 1px 0 rgba(33, 29, 22, 0.04), 0 12px 28px -20px rgba(33, 29, 22, 0.35);
}

.message.user .msg-bubble {
  background: var(--accent);
  color: #fff;
}

/* ---------- 工具调用折叠区 ---------- */
.tool-log {
  margin-top: 10px;
  border-top: 1px dashed var(--line);
  padding-top: 8px;
  font-family: var(--font-mono);
}

.tool-log summary {
  cursor: pointer;
  font-size: 12px;
  color: var(--ink-2);
  user-select: none;
}

.tool-log summary:hover {
  color: var(--accent);
}

.step-item {
  margin-top: 10px;
}

.step-tag {
  font-size: 12px;
  color: var(--ink-3);
  background: var(--paper-deep);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 1px 6px;
}

.tool-name {
  font-size: 12px;
  color: var(--accent);
  background: var(--accent-soft);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 1px 6px;
}

.tool-result {
  margin: 6px 0 0;
  padding: 8px 10px;
  max-height: 220px;
  overflow: auto;
  background: var(--paper-deep);
  border: 1px solid var(--line);
  border-radius: 4px;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--ink-2);
}

.msg-bubble.error {
  background: var(--bad-soft);
  border-color: rgba(163, 39, 31, 0.4);
  color: var(--bad);
}

.msg-text {
  white-space: pre-wrap;
}

.thinking {
  display: inline-flex;
  gap: 5px;
  align-items: center;
  margin-left: 2px;
}

.tdot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--ink-3);
  animation: blink 1.2s var(--ease-out) infinite;
}

.tdot:nth-child(2) {
  animation-delay: 0.18s;
}

.tdot:nth-child(3) {
  animation-delay: 0.36s;
}

/* ---------- 阶段提示 ---------- */
.stage-indicator {
  padding: 6px 4px 0;
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.08em;
  color: var(--accent);
  animation: rise 0.3s var(--ease-out) both;
}

/* ---------- 写盘确认条 ---------- */
.confirm-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin: 10px 0;
  padding: 12px 14px;
  background: var(--accent-soft);
  border: 1px solid var(--accent);
  border-radius: 10px;
  animation: rise 0.3s var(--ease-out) both;
}

.confirm-msg {
  font-size: 14px;
  line-height: 1.6;
  color: var(--ink);
}

.confirm-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

.confirm-btn {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.1em;
  padding: 7px 16px;
  border-radius: 999px;
  cursor: pointer;
  transition: background 0.25s var(--ease-out), color 0.25s var(--ease-out), border-color 0.25s var(--ease-out);
}

.confirm-btn.reject {
  background: none;
  border: 1px solid var(--line);
  color: var(--ink-2);
}

.confirm-btn.reject:hover:not(:disabled) {
  border-color: var(--bad);
  color: var(--bad);
  background: var(--bad-soft);
}

.confirm-btn.approve {
  background: var(--accent);
  border: 1px solid var(--accent);
  color: #fff;
}

.confirm-btn.approve:hover:not(:disabled) {
  background: var(--accent-deep);
}

.confirm-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* ---------- 澄清反问面板 ---------- */
.clarify-panel {
  margin: 10px 0;
  padding: 14px 16px;
  background: var(--surface);
  border: 1px solid var(--accent);
  border-radius: 10px;
  animation: rise 0.3s var(--ease-out) both;
}

.clarify-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 4px;
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.12em;
  color: var(--ink);
}

.clarify-title-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.clarify-error {
  margin: 0 0 8px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--bad);
  background: var(--bad-soft);
  border: 1px solid rgba(163, 39, 31, 0.4);
  border-radius: 4px;
  padding: 6px 10px;
}

.clarify-question {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--line);
}

.clarify-q-text {
  margin: 0 0 6px;
  font-size: 14px;
  line-height: 1.7;
  color: var(--ink);
}

.clarify-options {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.clarify-option {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  font-size: 13px;
  color: var(--ink-2);
  cursor: pointer;
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.clarify-option:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.clarify-option.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: var(--accent-deep);
}

/* 隐藏原生单选钮但保留可聚焦：label 整块承担选中态反馈 */
.clarify-option input {
  position: absolute;
  opacity: 0;
}

.clarify-open textarea {
  width: 100%;
  resize: vertical;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--paper);
  color: var(--ink);
  font-family: inherit;
  font-size: 13px;
  line-height: 1.6;
  outline: none;
  transition: border-color 0.2s var(--ease-out);
}

.clarify-open textarea:focus {
  border-color: var(--accent);
}

.clarify-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 12px;
}

.clarify-tip {
  font-size: 12px;
  color: var(--ink-3);
  line-height: 1.6;
}

/* ---------- 输入区（DeepSeek 风格圆角容器） ---------- */
.composer {
  display: flex;
  flex-direction: column;
  gap: 4px;
  border: 1px solid var(--line);
  background: var(--surface);
  border-radius: 14px;
  padding: 12px 14px 8px;
  transition: border-color 0.25s var(--ease-out);
}

.composer:focus-within {
  border-color: var(--accent);
}

.input {
  flex: 1;
  resize: none;
  border: none;
  background: none;
  color: var(--ink);
  font-family: inherit;
  font-size: 16px;
  line-height: 1.7;
  padding: 4px 2px;
  outline: none;
}

.input:focus {
  border: none;
}

.input:disabled {
  opacity: 0.6;
}

/* 输入容器底部工具行：左模型选择 + 深度检索开关，右侧发送靠边 */
.composer-toolbar {
  display: flex;
  align-items: flex-end;
  gap: 10px;
}

.composer-toolbar .send {
  margin-left: auto;
}

/* ---------- 模型选择器（左下角） ---------- */
.model-picker {
  position: relative;
}

.model-btn {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--ink-2);
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 7px 12px;
  border-radius: 999px;
  cursor: pointer;
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.model-btn:hover,
.model-btn.is-open {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

.model-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--ok);
  box-shadow: 0 0 0 3px var(--ok-soft);
}

.model-label {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-caret {
  font-size: 10px;
  color: var(--ink-3);
}

/* ---------- 深度检索开关（模型选择右侧） ---------- */
.deep-toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--ink-3);
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 7px 12px;
  border-radius: 999px;
  cursor: pointer;
  white-space: nowrap;
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.deep-toggle:hover:not(:disabled) {
  color: var(--accent);
  border-color: var(--accent);
}

.deep-toggle:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.deep-toggle.is-on {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

.deep-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--line);
}

.deep-toggle.is-on .deep-dot {
  background: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.model-menu {
  position: absolute;
  bottom: calc(100% + 10px);
  left: 0;
  width: 300px;
  max-height: 400px;
  overflow-y: auto;
  padding: 12px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.14);
  z-index: 50;
}

.model-menu-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.model-menu-title {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.1em;
  color: var(--ink-3);
}

.model-menu-close {
  border: none;
  background: transparent;
  color: var(--ink-3);
  font-size: 12px;
  cursor: pointer;
  padding: 2px 6px;
}

.model-menu-close:hover {
  color: var(--accent);
}

.model-option {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  text-align: left;
  border: 1px solid transparent;
  border-radius: 6px;
  background: none;
  padding: 8px 10px;
  cursor: pointer;
  transition: background 0.2s var(--ease-out), border-color 0.2s var(--ease-out);
}

.model-option:hover {
  background: var(--paper-deep);
}

.model-option.active {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.model-option-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.model-option-label {
  font-size: 13px;
  color: var(--ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-option-sub {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-option-del {
  flex-shrink: 0;
  color: var(--ink-3);
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 3px;
}

.model-option-del:hover {
  color: var(--bad);
  background: var(--bad-soft);
}

.model-menu-divider {
  height: 1px;
  margin: 8px 0;
  background: var(--line);
}

.model-form {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.model-field > span {
  font-size: 11px;
  color: var(--ink-3);
}

.model-field input {
  padding: 7px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 12px;
  outline: none;
  transition: border-color 0.2s var(--ease-out);
}

.model-field input:focus {
  border-color: var(--accent);
}

.model-error {
  margin: 0;
  font-size: 11px;
  color: var(--bad);
}

.model-submit {
  border: none;
  background: var(--accent);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  padding: 8px 0;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background 0.2s var(--ease-out);
}

.model-submit:hover {
  background: var(--accent-deep);
}

.send {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: none;
  background: var(--accent);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.1em;
  padding: 9px 18px;
  border-radius: 999px;
  white-space: nowrap;
  transition: background 0.25s var(--ease-out), transform 0.25s var(--ease-out);
}

.send:hover:not(:disabled) {
  background: var(--accent-deep);
  transform: translateY(-1px);
}

.send:disabled {
  background: var(--line);
  color: var(--ink-3);
  cursor: not-allowed;
}

.send-arrow {
  transition: transform 0.25s var(--ease-out);
}

.send:hover:not(:disabled) .send-arrow {
  transform: translateX(4px);
}

/* ---------- 动效 ---------- */
@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(12px);
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
    opacity: 0.4;
  }
}

/* ---------- 响应式 ---------- */
@media (max-width: 720px) {
  .layout {
    flex-direction: column;
  }

  .product-list {
    display: none; /* 窄屏不展示产物列，聚焦问答 */
  }

  .session-list {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--line);
    padding-right: 0;
    padding-bottom: 12px;
    max-height: 200px;
  }

  .msg-bubble {
    max-width: 88%;
  }

  .msg-label {
    display: none;
  }
}
</style>

<!--
  视口锁定（非 scoped：需作用于 body / .app 全局节点）
  进入聊天页时 body 挂 .chat-locked：整页固定为视口高度、禁止页面滚动，
  对话增长只在 .message-body 内部滚动；离开页面时类被移除，其他页面恢复正常整页滚动。
-->
<style>
body.chat-locked {
  height: 100vh;
  height: 100dvh;
  overflow: hidden;
}

body.chat-locked .app {
  height: 100vh;
  height: 100dvh;
  min-height: 0;
  overflow: hidden;
}

/* ---------- Markdown 正文样式（必须全局非 scoped） ----------
   v-html 注入的内容不带 data-v 属性，scoped 规则（.markdown-body img[data-v-x]）
   匹配不到，会导致图片等按原始尺寸渲染而溢出/被裁切。
   统一挂在 .chat-page 下避免影响其他页面。 */
.chat-page .markdown-body {
  line-height: 1.85;
  font-size: 17px;
  word-break: break-word;
  color: var(--ink);
}

.chat-page .markdown-body > :first-child {
  margin-top: 0;
}

.chat-page .markdown-body > :last-child {
  margin-bottom: 0;
}

.chat-page .markdown-body p {
  margin: 0.5em 0;
}

.chat-page .markdown-body h1,
.chat-page .markdown-body h2,
.chat-page .markdown-body h3,
.chat-page .markdown-body h4,
.chat-page .markdown-body h5,
.chat-page .markdown-body h6 {
  margin: 1.1em 0 0.5em;
  font-weight: 700;
  line-height: 1.4;
}

.chat-page .markdown-body h1 {
  font-size: 1.45em;
}

.chat-page .markdown-body h2 {
  font-size: 1.3em;
}

.chat-page .markdown-body h3 {
  font-size: 1.15em;
}

.chat-page .markdown-body h4,
.chat-page .markdown-body h5,
.chat-page .markdown-body h6 {
  font-size: 1em;
}

.chat-page .markdown-body ul,
.chat-page .markdown-body ol {
  margin: 0.5em 0;
  padding-left: 1.6em;
}

.chat-page .markdown-body li {
  margin: 0.2em 0;
}

.chat-page .markdown-body code {
  font-family: var(--font-mono);
  font-size: 0.9em;
  background: var(--paper-deep);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 0.1em 0.35em;
}

.chat-page .markdown-body pre {
  margin: 0.7em 0;
  padding: 12px 14px;
  background: var(--paper-deep);
  border: 1px solid var(--line);
  border-radius: 4px;
  overflow-x: auto;
}

.chat-page .markdown-body pre code {
  background: none;
  border: none;
  padding: 0;
  font-size: 0.85em;
}

.chat-page .markdown-body blockquote {
  margin: 0.7em 0;
  padding: 0.2em 1em;
  border-left: 3px solid var(--accent);
  color: var(--ink-2);
  background: var(--accent-soft);
}

.chat-page .markdown-body table {
  border-collapse: collapse;
  margin: 0.7em 0;
  max-width: 100%;
  display: block;
  overflow-x: auto;
}

.chat-page .markdown-body th,
.chat-page .markdown-body td {
  border: 1px solid var(--line);
  padding: 6px 12px;
  text-align: left;
}

.chat-page .markdown-body th {
  background: var(--paper-deep);
  font-weight: 600;
}

.chat-page .markdown-body a {
  color: var(--accent);
  text-decoration: underline;
}

.chat-page .markdown-body img {
  display: block;
  max-width: 100%;
  height: auto; /* 等比缩放，防止宽图拉伸变形 */
  margin: 8px 0;
  border: 1px solid var(--line);
  border-radius: 2px;
}

.chat-page .markdown-body hr {
  border: none;
  border-top: 1px solid var(--line);
  margin: 1em 0;
}

/* v-html / JS 创建的裂图占位（同样不带 data-v，需全局） */
.chat-page .msg-img-fallback {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
  padding: 4px 0;
}
</style>
