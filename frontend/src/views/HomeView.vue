<script setup lang="ts">
/**
 * 首页 · 教材上传
 * 流程：选择/拖拽 PDF → 批量上传（/upload）→ 创建任务（/resolve 拿 task_id）→ 跳转进度页
 * 文件选择/拖拽交给 Naive UI n-upload（default-upload=false + n-upload-dragger，只作选择器），
 * 文件列表与删除仍由本组件自管理，上传动作沿用 api/textbook 的 uploadTextbooks。
 */
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { NUpload, NUploadDragger, useMessage, type UploadFileInfo } from 'naive-ui'
import { uploadTextbooks } from '../api/textbook'
import { clearIngestTask, getIngestTask } from '../utils/ingestTask'
import type { IngestTask } from '../utils/ingestTask'

const router = useRouter()
const message = useMessage()

/** 上次离开时仍在解析的任务（进度页 info 落盘、done 清除）；存在则展示恢复入口 */
const pendingTask = ref<IngestTask | null>(null)

onMounted(() => {
  pendingTask.value = getIngestTask()
})

/** 从教材目录路径取末级目录名做展示（如 pdf-20260927-153000-ab12cd34） */
function taskLabel(path: string): string {
  const segs = path.replace(/[\\/]+$/, '').split(/[\\/]/)
  return segs[segs.length - 1] || path
}

/** 跳转进度页续跑（仅带 task_id，由后端 checkpoint 断点恢复） */
function resumeTask(task: IngestTask): void {
  void router.push({ name: 'progress', query: { task_id: task.taskId } })
}

/** 忽略失效任务记录（如后端已清理该任务时手动移除入口） */
function dismissTask(): void {
  clearIngestTask()
  pendingTask.value = null
}

/** 已选文件列表（自管理；n-upload 内部列表不参与展示，选中即取出 File） */
const files = ref<File[]>([])
/** 上传 + 建任务中 */
const isBusy = ref(false)
/** 错误提示 */
const errorMessage = ref('')

/** 选中文件总大小 */
const totalSize = computed(() => files.value.reduce((sum, f) => sum + f.size, 0))

/** 格式化文件大小 */
function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

/**
 * n-upload @change：逐文件收集 File，非 PDF 提醒跳过，同名同尺寸去重。
 * show-file-list=false 且 default-upload=false，n-upload 只充当「选择器」而非状态源。
 */
function onFileChange(data: { file: UploadFileInfo; fileList: UploadFileInfo[] }): void {
  const raw = data.file.file
  if (!raw) return
  if (!raw.name.toLowerCase().endsWith('.pdf')) {
    message.warning(`「${raw.name}」不是 PDF，已跳过`)
    return
  }
  if (files.value.some((e) => e.name === raw.name && e.size === raw.size)) {
    return
  }
  files.value = files.value.concat(raw)
}

/** 移除单个文件 */
function removeFile(index: number): void {
  files.value = files.value.filter((_, i) => i !== index)
}

/** 清空全部 */
function clearAll(): void {
  files.value = []
}

/**
 * 开始解析：先上传，再跳转进度页（进度页发起 /resolve SSE 流）
 */
async function start(): Promise<void> {
  if (!files.value.length || isBusy.value) return
  isBusy.value = true
  errorMessage.value = ''
  try {
    const uploaded = await uploadTextbooks(files.value)
    await router.push({ name: 'progress', query: { path: uploaded.textbook_path } })
  } catch (err) {
    errorMessage.value = (err as Error).message
  } finally {
    isBusy.value = false
  }
}
</script>

<template>
  <main class="home">
    <!-- 标题区 -->
    <header class="hero">
      <router-link class="back-link" to="/">← 返回书架</router-link>
      <p class="eyebrow">GRAD ASSIST · TEXTBOOK INGESTION</p>
      <h1 class="title">教材摄入</h1>
      <p class="subtitle">
        上传教材 PDF，流水线自动完成文件校验、目录解析、章节切割与向量入库。<br class="hide-mobile" />
        全程实时进度，断线可自动恢复。
      </p>
    </header>

    <!-- 流程说明 -->
    <aside class="steps" aria-label="操作步骤">
      <ol>
        <li class="step">
          <span class="step-num">01</span>
          <div class="step-body">
            <p class="step-title">选择教材</p>
            <p class="step-desc">支持多选 / 拖拽 PDF 文件</p>
          </div>
        </li>
        <li class="step">
          <span class="step-num">02</span>
          <div class="step-body">
            <p class="step-title">上传建任务</p>
            <p class="step-desc">批量上传并创建解析任务</p>
          </div>
        </li>
        <li class="step">
          <span class="step-num">03</span>
          <div class="step-body">
            <p class="step-title">追踪进度</p>
            <p class="step-desc">阶段清单 + 实时日志实时跟进</p>
          </div>
        </li>
      </ol>
    </aside>

    <!-- 上传面板 -->
    <section class="upload-card" aria-label="上传教材">
      <!-- 进行中解析任务：恢复入口 -->
      <div v-if="pendingTask" class="resume-bar">
        <span class="resume-pulse" aria-hidden="true" />
        <p class="resume-text">
          有教材正在解析中
          <span v-if="pendingTask.textbookPath" class="resume-path">{{ taskLabel(pendingTask.textbookPath) }}</span>
        </p>
        <button class="resume-btn" type="button" @click="resumeTask(pendingTask)">继续查看解析 →</button>
        <button class="resume-dismiss" type="button" aria-label="忽略该任务记录" @click="dismissTask">✕</button>
      </div>

      <n-upload
        class="dropzone-upload"
        multiple
        accept="application/pdf,.pdf"
        :default-upload="false"
        :show-file-list="false"
        :disabled="isBusy"
        @change="onFileChange"
      >
        <n-upload-dragger class="dropzone">
          <span class="dz-badge">PDF</span>
          <p class="dz-title">拖拽教材到这里</p>
          <p class="dz-hint">或点击选择文件 · 仅支持 PDF · 可多选</p>
        </n-upload-dragger>
      </n-upload>

      <!-- 文件列表 -->
      <ul v-if="files.length" class="file-list">
        <li v-for="(f, i) in files" :key="`${f.name}-${f.size}-${i}`" class="file-item">
          <span class="file-doc" aria-hidden="true">▤</span>
          <span class="file-name" :title="f.name">{{ f.name }}</span>
          <span class="file-size">{{ formatSize(f.size) }}</span>
          <button class="file-remove" :aria-label="`移除 ${f.name}`" @click="removeFile(i)">✕</button>
        </li>
      </ul>

      <!-- 操作区 -->
      <div class="action-row">
        <p class="error" v-if="errorMessage">{{ errorMessage }}</p>
        <p class="action-hint" v-else-if="files.length">
          {{ files.length }} 个文件 · 共 {{ formatSize(totalSize) }}
          <button class="text-btn" @click="clearAll">清空</button>
        </p>
        <p class="action-hint" v-else>尚未选择教材文件</p>
        <button class="cta" :disabled="!files.length || isBusy" @click="start">
          {{ isBusy ? '解析启动中' : '开始解析' }}
          <span class="cta-arrow" aria-hidden="true">→</span>
        </button>
      </div>
    </section>
  </main>
</template>

<style scoped>
.home {
  flex: 1;
  width: min(1180px, 100%);
  margin: 0 auto;
  padding: clamp(32px, 6vw, 72px) clamp(20px, 5vw, 64px) 80px;
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(0, 0.65fr);
  grid-template-areas:
    'hero  steps'
    'card  card';
  gap: clamp(24px, 3.5vw, 44px);
}

/* ---------- 标题区 ---------- */
.hero {
  grid-area: hero;
  align-self: end;
  animation: rise 0.7s var(--ease-out) both;
}

.back-link {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.1em;
  color: var(--ink-2);
  background: var(--surface);
  margin-bottom: 26px;
  padding: 8px 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  transition: color 0.2s var(--ease-out), border-color 0.2s var(--ease-out);
}

.back-link:hover {
  color: var(--accent);
  border-color: var(--accent);
}

.eyebrow {
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.34em;
  color: var(--accent);
  margin-bottom: 18px;
}

.title {
  font-family: var(--font-display);
  font-weight: 900;
  font-size: clamp(3rem, 8vw, 6rem);
  line-height: 1.05;
  letter-spacing: 0.04em;
  position: relative;
  display: inline-block;
}

/* 标题下朱红短划 */
.title::after {
  content: '';
  position: absolute;
  left: 0;
  bottom: -14px;
  width: 64px;
  height: 6px;
  background: var(--accent);
}

.subtitle {
  margin-top: 40px;
  max-width: 56ch;
  font-size: 15px;
  line-height: 1.9;
  color: var(--ink-2);
}

/* ---------- 流程说明 ---------- */
.steps {
  grid-area: steps;
  border-left: 1px solid var(--line);
  padding-left: clamp(18px, 2.5vw, 30px);
  animation: rise 0.7s var(--ease-out) 0.12s both;
}

.step {
  display: flex;
  gap: 16px;
  padding: 16px 0;
  border-bottom: 1px dashed var(--line);
  list-style: none;
}

.step-num {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 22px;
  color: var(--ink-3);
  line-height: 1.2;
}

.step-title {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 16px;
}

.step-desc {
  margin-top: 3px;
  font-size: 12px;
  color: var(--ink-3);
}

/* ---------- 上传面板 ---------- */
.upload-card {
  grid-area: card;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: clamp(18px, 2.5vw, 28px);
  box-shadow: 0 1px 0 rgba(33, 29, 22, 0.04), 0 18px 40px -28px rgba(33, 29, 22, 0.35);
  animation: rise 0.7s var(--ease-out) 0.2s both;
}

/* ---------- 进行中任务恢复条 ---------- */
.resume-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 18px;
  padding: 12px 16px;
  border: 1px solid rgba(194, 59, 34, 0.35);
  border-radius: var(--radius);
  background: var(--accent-soft);
}

.resume-pulse {
  width: 9px;
  height: 9px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 0 rgba(194, 59, 34, 0.5);
  animation: pulse 1.6s var(--ease-out) infinite;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(194, 59, 34, 0.45);
  }
  70% {
    box-shadow: 0 0 0 8px rgba(194, 59, 34, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(194, 59, 34, 0);
  }
}

.resume-text {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  color: var(--ink);
}

.resume-path {
  margin-left: 8px;
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--ink-2);
  word-break: break-all;
}

.resume-btn {
  border: 1px solid var(--accent);
  background: var(--accent);
  color: #fff;
  font-size: 12.5px;
  padding: 8px 14px;
  border-radius: var(--radius);
  white-space: nowrap;
  transition: background 0.2s var(--ease-out);
}

.resume-btn:hover {
  background: var(--accent-deep);
}

.resume-dismiss {
  border: none;
  background: none;
  color: var(--ink-3);
  font-size: 12px;
  padding: 6px;
  transition: color 0.2s var(--ease-out);
}

.resume-dismiss:hover {
  color: var(--accent);
}

/* ---------- 拖拽区（n-upload-dragger）：布局/留白自定义，边框配色走主题 Upload 覆写 ---------- */
.dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  padding: clamp(28px, 4vw, 44px) 20px;
  cursor: pointer;
  transition: transform 0.25s var(--ease-out), border-color 0.25s var(--ease-out), background-color 0.25s var(--ease-out);
}

/* 悬停 / 拖拽悬停时轻微上浮，与原版自定义 dropzone 的手感一致 */
.dropzone-upload :deep(.n-upload-trigger:hover .n-upload-dragger),
.dropzone-upload :deep(.n-upload--drag-over .n-upload-dragger) {
  transform: translateY(-2px);
}

.dz-badge {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.2em;
  color: var(--surface);
  background: var(--ink);
  padding: 3px 10px;
  border-radius: 2px;
}

.dz-title {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 19px;
  margin-top: 6px;
}

.dz-hint {
  font-size: 12px;
  color: var(--ink-3);
}

/* ---------- 文件列表 ---------- */
.file-list {
  list-style: none;
  margin-top: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 9px 12px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  animation: rise 0.35s var(--ease-out) both;
}

.file-doc {
  color: var(--accent);
  font-size: 16px;
}

.file-name {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
}

.file-remove {
  border: none;
  background: none;
  color: var(--ink-3);
  font-size: 13px;
  padding: 2px 6px;
  border-radius: 2px;
  transition: color 0.2s var(--ease-out), background 0.2s var(--ease-out);
}

.file-remove:hover {
  color: var(--bad);
  background: var(--bad-soft);
}

/* ---------- 操作区 ---------- */
.action-row {
  margin-top: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.error {
  color: var(--bad);
  font-size: 13px;
}

.action-hint {
  font-size: 12px;
  color: var(--ink-3);
}

.text-btn {
  border: none;
  background: none;
  color: var(--ink-2);
  text-decoration: underline;
  font-size: 12px;
  margin-left: 10px;
}

.text-btn:hover {
  color: var(--accent);
}

.cta {
  display: inline-flex;
  align-items: center;
  gap: 14px;
  border: none;
  background: var(--accent);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.12em;
  padding: 14px 26px;
  border-radius: var(--radius);
  transition: background 0.25s var(--ease-out), transform 0.25s var(--ease-out), box-shadow 0.25s var(--ease-out);
  box-shadow: 0 10px 24px -12px rgba(194, 59, 34, 0.7);
}

.cta:hover:not(:disabled) {
  background: var(--accent-deep);
  transform: translateY(-1px);
}

.cta:disabled {
  background: var(--line);
  color: var(--ink-3);
  box-shadow: none;
  cursor: not-allowed;
}

.cta-arrow {
  transition: transform 0.25s var(--ease-out);
}

.cta:hover:not(:disabled) .cta-arrow {
  transform: translateX(5px);
}

/* ---------- 动效 ---------- */
@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(18px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

/* ---------- 响应式 ---------- */
@media (max-width: 900px) {
  .home {
    grid-template-columns: 1fr;
    grid-template-areas:
      'hero'
      'steps'
      'card';
  }

  .steps {
    border-left: none;
    border-top: 1px solid var(--line);
    padding-left: 0;
    padding-top: 8px;
  }

  .hide-mobile {
    display: none;
  }
}
</style>
