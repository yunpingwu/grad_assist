/* ============================================================
   进行中解析任务的本地持久化
   - /resolve 的进度流绑定单个请求，离开进度页即断流；task_id 此前只活在
     ProgressView 组件内存里，返回上传页后就找不回任务
   - 后端支持凭 task_id 断点续跑（checkpoint），故前端只需把 {task_id, textbook_path}
     落 localStorage：进度页 info 事件写入、done 清除，上传页据此展示「继续查看解析」入口
   ============================================================ */

const KEY = 'gradassist_…_task'

/** 进行中的解析任务（凭 task_id 可续跑，textbook_path 仅展示用） */
export interface IngestTask {
  taskId: string
  textbookPath: string
}

export function saveIngestTask(task: IngestTask): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(task))
  } catch {
    /* 隐私模式等写入失败不影响主流程 */
  }
}

export function getIngestTask(): IngestTask | null {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return null
    const obj = JSON.parse(raw) as Partial<IngestTask>
    return obj?.taskId ? { taskId: obj.taskId, textbookPath: obj.textbookPath ?? '' } : null
  } catch {
    return null
  }
}

export function clearIngestTask(): void {
  try {
    localStorage.removeItem(KEY)
  } catch {
    /* 忽略 */
  }
}
