/* ============================================================
   API 层：教材助手对话接口封装
   /study/chat 为 SSE 流式对话：token 答案增量 / done 收尾(含 session_id) / error 异常
   agent 依据意图自主决定是否联网/检索/写文件；检索快慢档（深度检索）是唯一由用户定的开关。
   普通 JSON 请求走共享 axios 实例（http），SSE 流式走 fetch（axios 无法读流式 body）。
   ============================================================ */

import type { HistoryMessage, ModelConfig, SessionInfo } from './types'
import { API_BASE_URL, http } from './http'
import { authHeaders } from '../utils/user'

/** 澄清问题的建议选项：label 展示给用户，method 为选中后回灌模型的解决方法正文 */
export interface ClarifyOption {
  label: string
  method: string
}

/** 后端 ask_clarify 事件里的单个澄清问题 */
export interface ClarifyQuestion {
  id: string
  question: string
  type: 'choice' | 'open'
  options?: ClarifyOption[]
  hint?: string | null
}

/** 前端提交给后端的人工作答（POST /study/chat 的 clarify_answers） */
export interface ClarifyAnswer {
  id: string
  answer: string
}

/** 流式对话事件回调 */
export interface StreamHandlers {
  onStage?: (stage: string, message: string) => void
  onToken?: (content: string) => void
  onDone?: (sessionId: string) => void
  onError?: (message: string) => void
  /** 写盘确认请求（agent 要写文件生成资料时） */
  onNeedConfirm?: (info: { sessionId: string; message: string }) => void
  /** 澄清反问请求（任务描述不明确时 agent 追问用户；作答经 clarify_answers 续跑） */
  onClarify?: (info: { sessionId: string; questions: ClarifyQuestion[]; maxQuestions: number; error?: string }) => void
  /** 工具调用进度（检索结果等；从前端折叠展示，不混入正文） */
  onTool?: (info: { status: 'start' | 'done'; name: string; id?: string; content?: string }) => void
  /** 模型思考/推理内容（前端折叠展示，帮助看清思路） */
  onThought?: (content: string) => void
}

/** 解析单条 SSE 帧（data: {json}）并按 type 分发 */
function handleFrame(raw: string, handlers: StreamHandlers): void {
  const line = raw.trim()
  if (!line.startsWith('data:')) return
  try {
    const ev = JSON.parse(line.slice(5).trim()) as { type: string; [k: string]: unknown }
    switch (ev.type) {
      case 'stage':
        handlers.onStage?.(ev.stage as string, ev.message as string)
        break
      case 'token':
        handlers.onToken?.((ev.content as string) ?? '')
        break
      case 'done':
        handlers.onDone?.((ev.session_id as string) ?? '')
        break
      case 'ask_confirm':
        handlers.onNeedConfirm?.({
          sessionId: (ev.session_id as string) ?? '',
          message: (ev.message as string) ?? '助手准备写入文件，是否允许？',
        })
        break
      case 'ask_clarify':
        handlers.onClarify?.({
          sessionId: (ev.session_id as string) ?? '',
          questions: ((ev.questions as ClarifyQuestion[]) ?? []).map((q) => ({
            id: (q.id as string) ?? '',
            question: (q.question as string) ?? '',
            type: q.type === 'open' ? 'open' : 'choice',
            options: q.options ?? [],
            hint: q.hint ?? null,
          })),
          maxQuestions: (ev.max_questions as number) ?? 3,
          error: (ev.error as string) ?? '',
        })
        break
      case 'tool':
        handlers.onTool?.({
          status: ev.status === 'done' ? 'done' : 'start',
          name: (ev.name as string) ?? 'tool',
          id: (ev.id as string) ?? undefined,
          content: (ev.content as string) ?? '',
        })
        break
      case 'thought':
        handlers.onThought?.((ev.content as string) ?? '')
        break
      case 'error':
        handlers.onError?.((ev.message as string) ?? '生成答案失败')
        break
    }
  } catch {
    /* 忽略无法解析的帧 */
  }
}

/** 逐块读取 SSE 响应体并分帧回调 */
async function consumeSSE(resp: Response, handlers: StreamHandlers): Promise<void> {
  const reader = resp.body!.getReader()
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

/**
 * 发起一轮统一助手对话（POST /study/chat，SSE）。
 * @param textbookName 教材完整名（与后端过滤/检索精确匹配）
 * @param query 用户问题或资料生成要求；回应写盘确认/澄清反问时传空串
 * @param sessionId 已有会话 ID；新会话可传前端生成的 uuid，省略则由后端生成
 * @param handlers 事件回调：token 增量 / done(含 session_id) / error
 * @param decision 可选：'approve'|'reject'，回应上一轮停在写盘确认的会话（一次性放行/拒绝）
 * @param model 可选：自定义模型配置；缺省走服务端默认模型
 * @param clarifyAnswers 可选：回应上一轮停在澄清反问的会话，每个问题一条作答
 * @param deep 可选：本轮走深度检索档（HyDE + 精排，每次检索多约 5s）；缺省 false 走快检索，整轮所有检索共用
 */
export async function streamChat(
  textbookName: string,
  query: string,
  sessionId: string | undefined,
  handlers: StreamHandlers,
  decision?: 'approve' | 'reject',
  model?: ModelConfig,
  clarifyAnswers?: ClarifyAnswer[],
  deep?: boolean,
): Promise<void> {
  const resp = await fetch(`${API_BASE_URL}/study/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      textbook_name: textbookName,
      query,
      session_id: sessionId,
      ...(decision ? { decision } : {}),
      ...(clarifyAnswers ? { clarify_answers: clarifyAnswers } : {}),
      ...(deep ? { deep: true } : {}),
      ...(model ? { model: model.model, base_url: model.base_url, api_key: model.api_key } : {}),
    }),
  })
  if (!resp.ok || !resp.body) throw new Error(`问答失败：HTTP ${resp.status}`)
  await consumeSSE(resp, handlers)
}

/**
 * 获取某教材的会话列表（最近更新在前）。
 * @param textbookName 教材完整名（后端按此精确过滤）
 * @returns 会话摘要列表
 */
export async function listSessions(textbookName: string): Promise<SessionInfo[]> {
  const { data } = await http.get<SessionInfo[]>('/sessions', {
    params: { textbook_name: textbookName },
  })
  return data
}

/**
 * 获取某会话的历史消息（后端已限制最近 50 轮）。
 * @param sessionId 会话 ID
 * @returns 消息列表
 */
export async function fetchHistory(sessionId: string): Promise<HistoryMessage[]> {
  const { data } = await http.get<HistoryMessage[]>('/history', { params: { session_id: sessionId } })
  return data
}

/**
 * 删除单个历史会话（DELETE /sessions/{session_id}）。
 * @param sessionId 会话 ID
 * 204 = 删除成功；404 = 会话本就不存在，视为已删除（幂等），均不抛错。
 */
export async function deleteSession(sessionId: string): Promise<void> {
  try {
    await http.delete(`/sessions/${encodeURIComponent(sessionId)}`)
  } catch (err) {
    const status = (err as { response?: { status?: number } })?.response?.status
    if (status === 404) return
    throw err
  }
}

/** GET /study/files 产物文件条目 */
export interface ProductFile {
  filename: string
  rel_path: string
  size: number
}

/**
 * 列出当前会话已生成的产物文件（GET /study/files）。
 * @param taskId 会话/任务 ID（即 session_id）
 * @param textbookName 教材完整名
 */
export async function listFiles(taskId: string, textbookName: string): Promise<ProductFile[]> {
  const { data } = await http.get<ProductFile[]>('/study/files', {
    params: { task_id: taskId, textbook_name: textbookName },
  })
  return data
}

/**
 * 读取产物文件内容（GET /study/file，路径经沙箱校验）。
 * @param taskId 会话/任务 ID（即 session_id）
 * @param textbookName 教材完整名
 * @param path 相对路径
 */
export async function readProductFile(taskId: string, textbookName: string, path: string): Promise<string> {
  const { data } = await http.get<{ content: string }>('/study/file', {
    params: { task_id: taskId, textbook_name: textbookName, path },
  })
  return data.content
}