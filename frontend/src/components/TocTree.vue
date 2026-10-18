<script setup lang="ts">
/**
 * 目录树面板：展示各教材解析出的三层目录（教材 → 章 → 小节）
 * 古籍目录页样式：点线引导 + 右对齐等宽页码；toc 事件逐本到达时按教材 upsert
 */
import { computed } from 'vue'
import type { TocEventData } from '../api/types'

const props = defineProps<{
  /** 已到达的目录树（按教材聚合，toc 事件 upsert 后的顺序） */
  books: TocEventData[]
}>()

/** 单本教材统计：章数 / 小节数 */
function bookStats(book: TocEventData): string {
  const secs = book.chapters.reduce((n, c) => n + c.sections.length, 0)
  return `${book.chapters.length} 章 · ${secs} 小节`
}

/** 有任意小节时才渲染小节层 */
const hasBooks = computed(() => props.books.length > 0)
</script>

<template>
  <div class="toc-tree">
    <template v-if="hasBooks">
      <section v-for="book in books" :key="book.textbook" class="toc-book">
        <header class="toc-book-head">
          <h4 class="toc-book-name">{{ book.textbook }}</h4>
          <span class="toc-book-meta">{{ bookStats(book) }}</span>
        </header>
        <ol class="toc-list">
          <li v-for="ch in book.chapters" :key="ch.num" class="toc-chapter">
            <div class="toc-line chapter">
              <span class="toc-num">{{ ch.num }}</span>
              <span class="toc-title">{{ ch.title }}</span>
              <span class="toc-dots" aria-hidden="true" />
              <span class="toc-page">{{ ch.printed_page }}</span>
            </div>
            <ol v-if="ch.sections.length" class="toc-sections">
              <li v-for="sec in ch.sections" :key="sec.num" class="toc-line section">
                <span class="toc-num">{{ sec.num }}</span>
                <span class="toc-title">{{ sec.title }}</span>
                <span class="toc-dots" aria-hidden="true" />
                <span class="toc-page">{{ sec.printed_page }}</span>
              </li>
            </ol>
          </li>
        </ol>
      </section>
    </template>
    <p v-else class="toc-empty">目录解析完成后将在此展示目录树…</p>
  </div>
</template>

<style scoped>
.toc-tree {
  flex: 1;
  min-height: 300px;
  max-height: 430px;
  overflow-y: auto;
  padding: 12px 4px 6px;
  animation: rise 0.4s var(--ease-out) both;
}

.toc-book {
  margin-bottom: 22px;
}

.toc-book:last-child {
  margin-bottom: 4px;
}

.toc-book-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 8px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--line);
}

.toc-book-name {
  font-family: var(--font-display);
  font-weight: 900;
  font-size: 14px;
  letter-spacing: 0.04em;
  color: var(--ink);
  word-break: break-all;
}

.toc-book-meta {
  flex-shrink: 0;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--ink-3);
}

.toc-list {
  list-style: none;
  display: flex;
  flex-direction: column;
}

.toc-sections {
  list-style: none;
  margin: 2px 0 4px;
}

/* 一行目录条目：编号 标题 …… 页码 */
.toc-line {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 4px 0;
}

.toc-line.chapter {
  margin-top: 4px;
}

.toc-line.chapter .toc-num {
  font-family: var(--font-display);
  font-weight: 700;
  color: var(--accent);
  font-size: 12.5px;
}

.toc-line.chapter .toc-title {
  font-family: var(--font-display);
  font-weight: 700;
  color: var(--ink);
  font-size: 13.5px;
}

.toc-line.section {
  padding-left: 26px;
}

.toc-line.section .toc-num {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
}

.toc-line.section .toc-title {
  font-size: 12.5px;
  color: var(--ink-2);
}

.toc-num {
  flex-shrink: 0;
  min-width: 42px;
}

.toc-title {
  flex-shrink: 0;
  max-width: calc(100% - 100px);
  word-break: break-all;
}

/* 点线引导：弹性占位，虚线随剩余宽度铺开 */
.toc-dots {
  flex: 1;
  min-width: 20px;
  border-bottom: 1px dotted var(--ink-3);
  opacity: 0.5;
  transform: translateY(-4px);
}

.toc-page {
  flex-shrink: 0;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--ink-2);
  min-width: 26px;
  text-align: right;
}

.toc-empty {
  padding: 40px 0;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--ink-3);
}

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
</style>
