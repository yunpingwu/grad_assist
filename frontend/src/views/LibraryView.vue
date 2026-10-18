<script setup lang="ts">
/**
 * 书架页 · 已入库教材列表
 * 卡片 = 一本书：上方封面占位（暂无封面图，用书皮色 + 书名首字），下方教材名。
 * 点击任意教材进入多轮对话；右上角「添加教材」进入摄入流程（/ingest）。
 */
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { NPagination } from 'naive-ui'
import { listTextbooks } from '../api/textbook'
import type { TextbookInfo } from '../api/types'
import { useStudyStore } from '../stores/study'

const router = useRouter()
const studyStore = useStudyStore()

/** 教材列表（当前页） */
const textbooks = ref<TextbookInfo[]>([])
/** 分页信息：total 与每页条数均由后端 /list 下发，前端不自带默认值 */
const total = ref(0)
const page = ref(1)
const pageSize = ref(0)
/** 总页数（未取到每页条数或 total=0 时为 0） */
const totalPages = () => (pageSize.value < 1 ? 0 : Math.max(Math.ceil(total.value / pageSize.value), 0))
/** 加载中 / 错误 */
const isLoading = ref(true)
const errorMessage = ref('')

/**
 * 封面底色：按「排序后的均匀色环铺开」为每本书分配柔和色调。
 * - 均匀铺开（第 i 本 hue = i/N × 360°）→ 不同书的色相差恒为 360°/N，撞色最少、差异最大化
 * - 排序由书名决定 → 列表重排时同一本书颜色保持不变
 * - 饱和度/明度锁定在柔和区间（S≈38%, L≈40%）→ 低饱和低亮、不刺眼，
 *   同时保留「古籍书封」暗调质感，保证封面上纸色文字可读
 */
let coverPalette = new Map<string, string>()

function buildCoverPalette(items: TextbookInfo[]): void {
  // 排序 → 去重，名次决定色调（分类无关，仅用于稳定分布）
  const names = [...new Set(items.map((b) => b.textbook_name))].sort()
  const n = Math.max(names.length, 1)
  const SAT = 38
  const LIGHT = 40
  coverPalette = new Map(
    names.map((name, i) => {
      const hue = ((i / n) * 360 + 20) % 360
      return [name, `hsl(${hue.toFixed(1)}, ${SAT}%, ${LIGHT}%)`]
    }),
  )
}

/** 展示用短名：按下划线或空白分割取首段（去 z-library 等下载后缀），仅用于显示；点击仍传完整名 */
function shortName(name: string): string {
  return name.split(/[_\s]+/)[0].trim() || name
}

/** 加载教材列表 */
async function load(): Promise<void> {
  isLoading.value = true
  errorMessage.value = ''
  try {
    const result = await listTextbooks(page.value)
    textbooks.value = result.items
    total.value = result.total
    pageSize.value = result.page_size
    // 删除操作或异常导致当前页越界时，回退到最后一页
    if (result.items.length === 0 && result.total > 0) {
      page.value = totalPages()
      if (page.value >= 1) {
        await load()
        return
      }
    }
    buildCoverPalette(textbooks.value)
  } catch (err) {
    errorMessage.value = (err as Error).message
  } finally {
    isLoading.value = false
  }
}

/** 跳转到指定页 */
async function goToPage(p: number): Promise<void> {
  if (p < 1 || p > totalPages() || p === page.value) return
  page.value = p
  await load()
}

/** 进入对话页：记录当前教材到 study store（供全局引用）并跳转 */
function openChat(name: string): void {
  studyStore.setTextbook(name)
  void router.push({ name: 'chat', query: { textbook: name } })
}

onMounted(load)
</script>

<template>
  <main class="library">
    <!-- 标题区 -->
    <header class="hero">
      <p class="eyebrow">GRAD ASSIST · LIBRARY</p>
      <h1 class="title">教材书架</h1>
      <p class="subtitle">
        选一本书，开始提问。已入库教材会出现在这里，点击任意一本进入多轮对话。
      </p>
      <p class="count" v-if="!isLoading && !errorMessage">
        共 {{ total }} 本已入库
      </p>
    </header>

    <!-- 加载中 -->
    <section v-if="isLoading" class="shelf" aria-label="加载中">
      <div v-for="i in 6" :key="i" class="card skeleton" aria-hidden="true">
        <div class="cover"></div>
        <div class="card-body">
          <p class="card-name"></p>
        </div>
      </div>
    </section>

    <!-- 出错 -->
    <section v-else-if="errorMessage" class="empty">
      <p class="empty-code">LOAD FAILED</p>
      <h2 class="empty-title">书架加载失败</h2>
      <p class="empty-desc">{{ errorMessage }}</p>
      <button class="cta" @click="load">重试</button>
    </section>

    <!-- 空书架 -->
    <section v-else-if="!textbooks.length" class="empty">
      <p class="empty-code">EMPTY SHELF</p>
      <h2 class="empty-title">书架上还没有教材</h2>
      <p class="empty-desc">上传一本教材 PDF，解析入库后它就会出现在这里，供你提问。</p>
      <button class="cta" @click="router.push('/ingest')">添加教材 →</button>
    </section>

    <!-- 书架网格 -->
    <section v-else class="shelf" aria-label="教材列表">
      <button
        v-for="book in textbooks"
        :key="book.collection_name"
        class="card"
        @click="openChat(book.textbook_name)"
      >
        <!-- 封面占位：书皮色 + 短书名（split[0]）+ 集合编号 -->
        <span class="cover" :style="{ backgroundColor: coverPalette.get(book.textbook_name) }">
          <span class="cover-tag">{{ book.collection_name.toUpperCase() }}</span>
          <span class="cover-title" :title="book.textbook_name">{{ shortName(book.textbook_name) }}</span>
          <span class="cover-foot">GRAD ASSIST</span>
        </span>
        <!-- 名称在下（短名） -->
        <span class="card-body">
          <span class="card-name" :title="book.textbook_name">{{ shortName(book.textbook_name) }}</span>
          <span class="card-meta">{{ book.chunk_count }} 段已入库</span>
        </span>
      </button>
    </section>

    <!-- 分页 -->
    <div v-if="!isLoading && !errorMessage && totalPages() > 1" class="pager">
      <n-pagination
        :page="page"
        :page-size="pageSize"
        :item-count="total"
        :page-slot="7"
        @update:page="goToPage"
      />
    </div>
  </main>
</template>

<style scoped>
.library {
  flex: 1;
  width: min(1180px, 100%);
  margin: 0 auto;
  padding: clamp(32px, 6vw, 72px) clamp(20px, 5vw, 64px) 80px;
}

/* ---------- 标题区 ---------- */
.hero {
  margin-bottom: clamp(28px, 4vw, 44px);
  animation: rise 0.7s var(--ease-out) both;
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
  font-size: clamp(2.6rem, 7vw, 4.6rem);
  line-height: 1.1;
  letter-spacing: 0.04em;
  position: relative;
  display: inline-block;
}

/* 标题下朱红短划 */
.title::after {
  content: '';
  position: absolute;
  left: 0;
  bottom: -12px;
  width: 56px;
  height: 5px;
  background: var(--accent);
}

.subtitle {
  margin-top: 34px;
  max-width: 52ch;
  font-size: 15px;
  line-height: 1.9;
  color: var(--ink-2);
}

.count {
  margin-top: 16px;
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.12em;
  color: var(--ink-3);
}

/* ---------- 书架网格 ---------- */
.shelf {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: clamp(18px, 3vw, 30px);
  animation: rise 0.7s var(--ease-out) 0.1s both;
}

.card {
  display: flex;
  flex-direction: column;
  text-align: left;
  background: none;
  border: none;
  padding: 0;
  outline: none;
}

/* ---------- 封面占位 ---------- */
.cover {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  aspect-ratio: 3 / 4;
  border-radius: 2px 6px 6px 2px;
  /* 书脊高光：左侧一条浅色竖线模拟书脊 */
  box-shadow:
    inset 5px 0 0 rgba(251, 247, 238, 0.22),
    inset -2px 0 0 rgba(33, 29, 22, 0.14),
    0 14px 28px -14px rgba(33, 29, 22, 0.45);
  color: var(--paper);
  padding: 16px 12px;
  transition: transform 0.28s var(--ease-out), box-shadow 0.28s var(--ease-out);
}

.card:hover .cover,
.card:focus-visible .cover {
  transform: translateY(-4px);
  box-shadow:
    inset 5px 0 0 rgba(251, 247, 238, 0.22),
    inset -2px 0 0 rgba(33, 29, 22, 0.14),
    0 22px 38px -16px rgba(33, 29, 22, 0.5);
}

.cover-tag {
  align-self: flex-end;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.2em;
  opacity: 0.85;
}

.cover-title {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: clamp(1rem, 1.8vw, 1.25rem);
  line-height: 1.6;
  text-align: center;
  /* 最多 4 行，超长省略（悬浮可见完整名） */
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
  word-break: break-all;
}

.cover-foot {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.3em;
  opacity: 0.6;
}

/* ---------- 名称在下 ---------- */
.card-body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 4px 0;
}

.card-name {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 16px;
  line-height: 1.45;
  color: var(--ink);
  /* 两行截断 */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-meta {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--ink-3);
}

/* ---------- 分页 ---------- */
.pager {
  display: flex;
  justify-content: center;
  margin-top: clamp(28px, 4vw, 44px);
}

/* ---------- 骨架屏 ---------- */
.skeleton .cover {
  background: var(--line);
  animation: pulse 1.3s var(--ease-out) infinite;
}

.skeleton .card-name {
  width: 70%;
  height: 14px;
  background: var(--line);
  border-radius: 2px;
  animation: pulse 1.3s var(--ease-out) infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.45;
  }
}

/* ---------- 空态 / 出错 ---------- */
.empty {
  text-align: center;
  padding: clamp(40px, 8vw, 90px) 20px;
  border: 1px dashed var(--line);
  border-radius: 4px;
  animation: rise 0.6s var(--ease-out) both;
}

.empty-code {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.3em;
  color: var(--accent);
}

.empty-title {
  margin-top: 14px;
  font-family: var(--font-display);
  font-weight: 900;
  font-size: clamp(1.5rem, 3.4vw, 2.2rem);
}

.empty-desc {
  margin: 12px auto 26px;
  max-width: 44ch;
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

/* ---------- 动效 ---------- */
@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(16px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

/* ---------- 响应式 ---------- */
@media (max-width: 560px) {
  .shelf {
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 16px;
  }
}
</style>
