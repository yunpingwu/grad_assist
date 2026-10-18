/* ============================================================
   匿名设备身份：多用户软隔离
   - 首次访问生成 uuid 存 localStorage，之后复用
   - 所有请求带 X-User-Id 请求头，后端据此区分用户（软隔离，非安全边界）
   ============================================================ */

/** localStorage 密钥：user_id 同时充当「恢复码」 */
const USER_ID_KEY = 'gradassist_user_id'
/** opt-in 标记：用户是否已显式创建/粘贴过恢复码，用于和「请求自动生成的匿名 id」解耦 */
const OPT_KEY = 'gradassist_recovery_optin'

/** 用户是否已显式拥有恢复码（点击过创建或粘贴找回） */
export function hasRecoveryOptIn(): boolean {
  return localStorage.getItem(OPT_KEY) === '1'
}

/** 标记用户已显式创建/粘贴恢复码 */
export function setRecoveryOptIn(): void {
  localStorage.setItem(OPT_KEY, '1')
}

/** 是否已存在恢复码（不触发创建） */
export function hasUserId(): boolean {
  return Boolean(localStorage.getItem(USER_ID_KEY))
}

/** 读取已存在的恢复码；不存在返回空串（不触发创建） */
export function getStoredUserId(): string {
  return localStorage.getItem(USER_ID_KEY) ?? ''
}

/** 生成并持久化一个新的恢复码 */
export function createUserId(): string {
  const id = crypto.randomUUID()
  localStorage.setItem(USER_ID_KEY, id)
  return id
}

/** 用指定值覆盖当前恢复码（粘贴找回历史用）；非法空串时忽略 */
export function setUserId(id: string): void {
  const v = (id ?? '').trim()
  if (v) localStorage.setItem(USER_ID_KEY, v)
}

/** 获取（或生成）当前设备的匿名用户 ID */
export function getUserId(): string {
  let id = localStorage.getItem(USER_ID_KEY)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(USER_ID_KEY, id)
  }
  return id
}

/** 附加 X-User-Id 的请求头 */
export function authHeaders(): Record<string, string> {
  return { 'X-User-Id': getUserId() }
}
