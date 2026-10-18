/* ============================================================
   Markdown 渲染工具：marked 渲染 + DOMPurify 消毒 + KaTeX 公式
   - 模型输出按 markdown 渲染（### 标题 / **粗体** / `行内码` / 代码块 / 表格 / 图片）
   - LaTeX 公式：`$...$` 行内、`$$...$$` 独立行，经 KaTeX 渲染为 HTML
   - 渲染结果必须经 DOMPurify 消毒后再进 v-html，防 XSS
   ============================================================ */

import DOMPurify from 'dompurify'
import katex from 'katex'
import { marked } from 'marked'

// gfm：表格/删除线等；breaks：单个换行 → <br>，保持与旧纯文本观感一致
marked.use({ gfm: true, breaks: true })

// KaTeX 渲染公共参数：出错不抛异常（缺某个宏包时保留原文提示），output="html" 避免 MathML
const KATEX_OPTS = { throwOnError: false, output: 'html' } as const

// ── LaTeX 公式扩展（marked 原生 tokenizer/renderer）─────────────────

// 块级公式：$$ ... $$（仅在行首命中）
const blockMathExtension = {
  name: 'blockMath',
  level: 'block' as const,
  start(src: string): number | undefined {
    return src.startsWith('$$') ? 0 : undefined
  },
  tokenizer(src: string) {
    const match = src.match(/^\$\$([\s\S]+?)\$\$/)
    if (match?.[1]?.trim()) return { type: 'blockMath', raw: match[0], text: match[1].trim() }
    return undefined
  },
  renderer(token: { text: string }) {
    return katex.renderToString(token.text, { ...KATEX_OPTS, displayMode: true })
  },
}

// 行内公式：$ ... $（内容非空且不含换行）
const inlineMathExtension = {
  name: 'inlineMath',
  level: 'inline' as const,
  start(src: string): number | undefined {
    const i = src.indexOf('$')
    return i === -1 ? undefined : i
  },
  tokenizer(src: string) {
    const match = src.match(/^\$([^$\n]+?)\$/)
    if (match?.[1]?.trim()) return { type: 'inlineMath', raw: match[0], text: match[1].trim() }
    return undefined
  },
  renderer(token: { text: string }) {
    return katex.renderToString(token.text, KATEX_OPTS)
  },
}

marked.use({ extensions: [blockMathExtension, inlineMathExtension] })

// 链接统一新窗口打开（在 DOMPurify 属性清理后追加，避免被 strip）
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank')
    node.setAttribute('rel', 'noopener noreferrer')
  }
})

/** 将 markdown 文本渲染为已消毒的 HTML；空文本返回空串。 */
export function renderMarkdown(text: string): string {
  if (!text) return ''
  const html = marked.parse(text, { async: false }) as string
  // KaTeX 输出依赖 class 与少量内联 style，需显式放行，其余仍按默认白名单清理
  return DOMPurify.sanitize(html, { ADD_ATTR: ['style'] })
}