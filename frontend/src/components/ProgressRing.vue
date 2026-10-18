<script setup lang="ts">
/**
 * 环形进度指示器
 * 依据任务状态着色：pending 灰 / running 朱红 / succeeded 绿 / failed 红
 */
import { computed } from 'vue'
import type { TaskState } from '../api/types'

const props = defineProps<{
  /** 进度 0~1 */
  progress: number
  /** 任务状态 */
  state: TaskState
}>()

const R = 64
const CIRCUMFERENCE = 2 * Math.PI * R

/** 归一化进度 */
const clamped = computed(() => Math.min(Math.max(props.progress, 0), 1))
/** 环上缺口偏移 */
const dashOffset = computed(() => CIRCUMFERENCE * (1 - clamped.value))
/** 百分比整数 */
const percent = computed(() => Math.round(clamped.value * 100))
</script>

<template>
  <div class="ring" :class="state">
    <svg viewBox="0 0 160 160" width="160" height="160" aria-hidden="true">
      <circle class="track" cx="80" cy="80" :r="R" fill="none" stroke-width="10" />
      <circle
        class="value"
        cx="80"
        cy="80"
        :r="R"
        fill="none"
        stroke-width="10"
        stroke-linecap="round"
        :stroke-dasharray="CIRCUMFERENCE"
        :stroke-dashoffset="dashOffset"
      />
    </svg>
    <div class="center">
      <span class="pct">{{ percent }}</span>
      <span class="unit">%</span>
    </div>
  </div>
</template>

<style scoped>
.ring {
  position: relative;
  width: 160px;
  height: 160px;
  flex-shrink: 0;
}

.track {
  stroke: var(--line);
}

.value {
  stroke: var(--ink-3);
  transform: rotate(-90deg);
  transform-origin: center;
  transition: stroke-dashoffset 0.5s var(--ease-out), stroke 0.4s var(--ease-out);
}

/* 运行中：朱红 + 呼吸光晕 */
.ring.running .value {
  stroke: var(--accent);
  filter: drop-shadow(0 0 6px rgba(194, 59, 34, 0.45));
}

.ring.pending .value {
  stroke: var(--ink-3);
}

.ring.succeeded .value {
  stroke: var(--ok);
}

.ring.failed .value {
  stroke: var(--bad);
}

.center {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: baseline;
  justify-content: center;
  gap: 2px;
  padding-top: 62px;
}

.pct {
  font-family: var(--font-display);
  font-weight: 900;
  font-size: 40px;
  letter-spacing: 0.01em;
}

.unit {
  font-family: var(--font-mono);
  font-size: 14px;
  color: var(--ink-3);
}

@media (max-width: 560px) {
  .ring {
    width: 128px;
    height: 128px;
  }
  .ring svg {
    width: 128px;
    height: 128px;
  }
  .center {
    padding-top: 50px;
  }
  .pct {
    font-size: 30px;
  }
}
</style>
