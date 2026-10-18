<script setup lang="ts">
/**
 * 解析阶段清单（进度时间轴）
 * 依据任务状态 / 进度，将 6 个流水线节点标记为 已完成 / 进行中 / 待开始
 */
import { computed } from 'vue'
import type { TaskState } from '../api/types'

const props = defineProps<{
  /** 进度 0~1 */
  progress: number
  /** 任务状态 */
  state: TaskState
  /** 终态结果（成功 / 失败 / 未结束） */
  final: 'succeeded' | 'failed' | null
}>()

/** 流水线节点（threshold：该节点完成的进度阈值，对齐接口文档进度区间） */
const STAGES: Array<{ key: string; label: string; code: string; threshold: number }> = [
  { key: 'init', label: '任务创建', code: 'resolve', threshold: 0.1 },
  { key: 'load', label: '教材校验', code: 'load_textbook', threshold: 0.15 },
  { key: 'toc', label: '目录解析', code: 'split_contents', threshold: 0.4 },
  { key: 'split', label: '章节切割', code: 'split', threshold: 0.55 },
  { key: 'parse', label: '章节解析', code: 'parse_to_md', threshold: 0.8 },
  { key: 'store', label: '入库索引', code: 'split_text_and_store', threshold: 1 },
]

type StageState = 'done' | 'active' | 'pending'

/** 计算单个节点的状态 */
function stateOf(idx: number): StageState {
  if (props.final === 'succeeded') return 'done'
  const threshold = STAGES[idx].threshold
  if (props.progress >= threshold) return 'done'
  if (props.state === 'failed') return 'pending'
  if (idx === 0) return props.progress > 0 || props.state === 'running' ? 'active' : 'pending'
  return props.progress >= STAGES[idx - 1].threshold ? 'active' : 'pending'
}

/** 全部节点状态（按序） */
const stageStates = computed<StageState[]>(() => STAGES.map((_, i) => stateOf(i)))
</script>

<template>
  <ol class="stages" aria-label="解析阶段">
    <li
      v-for="(stage, i) in STAGES"
      :key="stage.key"
      class="stage"
      :class="stageStates[i]"
    >
      <span class="index">{{ String(i + 1).padStart(2, '0') }}</span>
      <div class="body">
        <p class="label">{{ stage.label }}</p>
        <p class="code">{{ stage.code }}</p>
      </div>
      <span class="mark">
        <span v-if="stageStates[i] === 'done'" class="check" aria-hidden="true">✓</span>
        <span v-else-if="stageStates[i] === 'active'" class="pulse" aria-hidden="true" />
      </span>
    </li>
  </ol>
</template>

<style scoped>
.stages {
  list-style: none;
  display: flex;
  flex-direction: column;
}

.stage {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 15px 4px;
  border-bottom: 1px dashed var(--line);
  transition: opacity 0.3s var(--ease-out);
}

.index {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 20px;
  width: 34px;
  color: var(--ink-3);
  transition: color 0.3s var(--ease-out);
}

.body {
  flex: 1;
  min-width: 0;
}

.label {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 15px;
  color: var(--ink-3);
  transition: color 0.3s var(--ease-out);
}

.code {
  margin-top: 2px;
  font-family: var(--font-mono);
  font-size: 10.5px;
  letter-spacing: 0.06em;
  color: var(--ink-3);
  opacity: 0.7;
}

/* 已完成 */
.stage.done .index {
  color: var(--ok);
}

.stage.done .label {
  color: var(--ink);
}

.check {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--ok);
  color: var(--surface);
  font-size: 12px;
}

/* 进行中 */
.stage.active .index {
  color: var(--accent);
}

.stage.active .label {
  color: var(--accent);
}

.pulse {
  width: 10px;
  height: 10px;
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
    box-shadow: 0 0 0 9px rgba(194, 59, 34, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(194, 59, 34, 0);
  }
}

/* 待开始 */
.stage.pending {
  opacity: 0.55;
}
</style>
