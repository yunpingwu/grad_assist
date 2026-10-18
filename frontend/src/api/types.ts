/* ============================================================
   后端接口类型定义（与 textbook_api_documentation.md 对齐）
   ============================================================ */

/** 任务状态：创建后 pending → 解析中 running → 终态 succeeded / failed */
export type TaskState = 'pending' | 'running' | 'succeeded' | 'failed'

/** POST /upload 响应 */
export interface UploadResult {
  textbook_path: string
  message: string
}

/** POST /resolve 响应 */
export interface ResolveResult {
  task_id: string
  message: string
}

/** 增量进度事件（SSE message 事件 data / /status events[] 元素） */
export interface TaskEvent {
  ts: number
  status: TaskState
  message: string
  progress: number
  data: Record<string, unknown>
}

/** SSE done 事件 data */
export interface DoneEventData {
  task_id?: string
}

/** SSE error 事件 data */
export interface ErrorEventData {
  message: string
}

/** 教材页码偏移条目：ask_offset 事件的候选值、确认回传的单元 */
export interface OffsetEntry {
  textbook_name: string
  offset: number
}

/** SSE info 事件：首个事件，携带 task_id 供续跑 */
export interface InfoEventData {
  type: 'info'
  task_id: string
  resumed: boolean
  message: string
  progress?: number
  /** 断线重进时后端随首帧带回的目录树快照（首解析尚无树时不带） */
  tocs?: TocEventData[]
}

/** SSE ask_offset 事件：章节页码偏移待人工确认 */
export interface AskOffsetEventData {
  type: 'ask_offset'
  offsets: OffsetEntry[]
  message?: string
}

/** 目录树小节（编号 / 标题 / 印刷页码） */
export interface TocSection {
  num: string
  title: string
  printed_page: number
}

/** 目录树章（含所属小节） */
export interface TocChapter {
  num: string
  title: string
  printed_page: number
  sections: TocSection[]
}

/** SSE toc 事件：某本教材的三层目录树（实时推送；断线重进改由 info.tocs 快照恢复） */
export interface TocEventData {
  type: 'toc'
  textbook: string
  chapters: TocChapter[]
}

/** GET /list 教材条目（教材注册表 → 书架卡片数据） */
export interface TextbookInfo {
  textbook_name: string
  collection_name: string
  chunk_count: number
  created_at: string
}

/** GET /list 分页响应（items + total + 当前分页信息） */
export interface TextbookPage {
  items: TextbookInfo[]
  total: number
  page: number
  page_size: number
}

/** 处理过程时间线里的一步：思考（thought）或工具结果（tool），按到达顺序穿插 */
export type StepItem =
  | { type: 'thought'; content: string }
  | { type: 'tool'; name: string; content?: string }

/** GET /study/chat 事件（data 内 JSON 的 type 字段） */
export type StreamEventType = 'token' | 'done' | 'error' | 'stage' | 'tool' | 'ask_confirm' | 'thought'

/** GET /sessions 会话摘要（左侧会话栏数据源） */
export interface SessionInfo {
  session_id: string
  updated_at: string
  message_count: number
  /** 最新用户问题原文；截断等展示整形由前端负责 */
  last_question: string
}

/** GET /history 会话消息 */
export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
  created_at?: string
  images?: Array<{ index: number; url: string; description: string }>
  /** 思考 + 检索工具结果按真实顺序穿插的时间线（历史回显折叠展示） */
  steps?: StepItem[]
  /** 兼容旧接口数据：只含工具结果 */
  tools?: Array<{ name: string; content: string }>
}

/** GET /status 任务快照 */
export interface TaskStatus {
  task_id: string
  status: TaskState
  progress: number
  message: string
  events: TaskEvent[]
  created_at: number
  updated_at: number
}

/** 对话模型配置（用户本地偏好）：切换/新增模型用，随 /study/chat 请求传给后端 */
export interface ModelConfig {
  /** 本地唯一 id（uuid） */
  id: string
  /** 展示名（用户起名，如「DeepSeek 主力」） */
  label: string
  /** 模型名（OpenAI 兼容，如 deepseek-v4-flash） */
  model: string
  /** OpenAI 兼容地址 */
  base_url: string
  /** 密钥 */
  api_key: string
}
