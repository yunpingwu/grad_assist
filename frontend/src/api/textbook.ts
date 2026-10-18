/* ============================================================
   API 层：教材摄入接口封装
   - /upload 上传；/resolve 以 SSE 流返回解析实时进度（合并创建任务与订阅为单请求）
   - 普通 JSON 请求走共享 axios 实例（http），SSE 流式走 fetch
   ============================================================ */

import type {
  AskOffsetEventData,
  DoneEventData,
  ErrorEventData,
  InfoEventData,
  OffsetEntry,
  TaskEvent,
  TextbookPage,
  TocEventData,
  UploadResult,
} from './types'
import { API_BASE_URL, http } from './http'
import { authHeaders } from '../utils/user'

/** 解析任务 SSE 事件回调 */
export interface ResolveHandlers {
  onInfo?: (data: InfoEventData) => void
  onMessage?: (data: TaskEvent) => void
  onAskOffset?: (data: AskOffsetEventData) => void
  onToc?: (data: TocEventData) => void
  onDone?: (data: DoneEventData) => void
  onError?: (data: ErrorEventData) => void
}

/** 解析请求选项：初始解析不传；偏移确认后续跑传 taskId + offsets */
export interface ResolveOptions {
  taskId?: string
  offsets?: OffsetEntry[]
}

/** 解析单条 SSE 帧（data: {json}）并按 json.type 字段分发 */
function handleFrame(raw: string, handlers: ResolveHandlers): void {
  const data = raw
    .split('\n')
    .filter((l) => l.startsWith('data:'))
    .map((l) => l.slice(5).trim())
    .join('')
  if (!data) return
  let obj: Record<string, unknown>
  try {
    obj = JSON.parse(data) as Record<string, unknown>
  } catch {
    /* 忽略无法解析的帧 */
    return
  }
  switch (obj.type) {
    case 'info':
      handlers.onInfo?.(obj as unknown as InfoEventData)
      break
    case 'ask_offset':
      handlers.onAskOffset?.(obj as unknown as AskOffsetEventData)
      break
    case 'toc':
      handlers.onToc?.(obj as unknown as TocEventData)
      break
    case 'done':
      handlers.onDone?.(obj as unknown as DoneEventData)
      break
    case 'error':
      handlers.onError?.(obj as unknown as ErrorEventData)
      break
    case 'message':
    default:
      handlers.onMessage?.(obj as unknown as TaskEvent)
  }
}

/**
 * 分页获取已入库教材列表（书架数据源；教材库全局共享）
 * 不传 page_size：每页条数由后端默认值决定，调用方从响应 page_size 回读，避免前后端两处默认值漂移。
 * @param page 页码，从 1 开始
 */
export async function listTextbooks(page = 1): Promise<TextbookPage> {
  const { data } = await http.get<TextbookPage>('/list', { params: { page } })
  return data
}

/**
 * 批量上传教材 PDF（multipart/form-data，字段名 files）
 */
export async function uploadTextbooks(files: File[]): Promise<UploadResult> {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  const { data } = await http.post<UploadResult>('/upload', form)
  return data
}

/**
 * 解析教材：POST /resolve 以 SSE 流返回实时进度。
 * - 初始解析：传 textbookPath，不带 options
 * - 偏移确认续跑：传空 textbookPath，options 带 taskId + offsets（JSON body）
 */
export async function resolveTextbooks(
  textbookPath: string,
  handlers: ResolveHandlers,
  options: ResolveOptions = {},
): Promise<void> {
  const params = new URLSearchParams()
  if (options.taskId) params.set('task_id', options.taskId)
  if (textbookPath) params.set('textbook_path', textbookPath)

  const headers: Record<string, string> = { ...authHeaders() }
  let body: string | undefined
  if (options.offsets) {
    body = JSON.stringify(options.offsets)
    headers['Content-Type'] = 'application/json'
  }

  const qs = params.toString()
  const resp = await fetch(`${API_BASE_URL}/resolve${qs ? `?${qs}` : ''}`, {
    method: 'POST',
    headers,
    body,
  })
  if (!resp.ok || !resp.body) throw new Error(`解析失败：HTTP ${resp.status}`)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      handleFrame(buf.slice(0, idx), handlers)
      buf = buf.slice(idx + 2)
    }
  }
  if (buf.trim()) handleFrame(buf, handlers) // 处理无尾随空行的残余帧
}