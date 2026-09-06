<template>
  <div
    class="float-card-wrapper"
    :class="[`delay-${delay}`, { 'hover-scale': hoverScale, 'clickable': clickable }]"
    @click="handleClick"
  >
    <div class="float-card-inner" :style="cardStyle">
      <div v-if="$slots.header" class="card-header">
        <slot name="header" />
      </div>
      <div class="card-body">
        <slot />
      </div>
      <div v-if="$slots.footer" class="card-footer">
        <slot name="footer" />
      </div>
      <!-- 装饰发光边框 -->
      <div class="card-glow" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  // 入场动画延迟 (1-4)
  delay: {
    type: Number,
    default: 1,
    validator: (v) => v >= 1 && v <= 4,
  },
  // 悬浮时是否缩放
  hoverScale: {
    type: Boolean,
    default: true,
  },
  // 是否可点击
  clickable: {
    type: Boolean,
    default: false,
  },
  // 自定义宽度
  width: {
    type: String,
    default: '100%',
  },
  // 自定义高度
  height: {
    type: String,
    default: '100%',
  },
  // 背景色
  bgColor: {
    type: String,
    default: '',
  },
  // 边框颜色
  borderColor: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['click'])

const cardStyle = computed(() => ({
  width: props.width,
  height: props.height,
  ...(props.bgColor ? { '--card-bg': props.bgColor } : {}),
  ...(props.borderColor ? { '--card-border': props.borderColor } : {}),
}))

const handleClick = () => {
  if (props.clickable) {
    emit('click')
  }
}
</script>

<style scoped lang="scss">
.float-card-wrapper {
  height: 100%;
  animation: card-in 0.7s cubic-bezier(0.22, 0.61, 0.36, 1) both;
  transition: transform 0.25s ease, box-shadow 0.25s ease;

  &.delay-1 {
    animation-delay: 0.02s;
  }
  &.delay-2 {
    animation-delay: 0.1s;
  }
  &.delay-3 {
    animation-delay: 0.18s;
  }
  &.delay-4 {
    animation-delay: 0.26s;
  }

  &.hover-scale:hover {
    transform: translateY(-3px);
  }

  &.clickable {
    cursor: pointer;
  }
}

.float-card-inner {
  position: relative;
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 14px 16px;
  border-radius: $border-radius;
  /* 渐变描边：外层 conic 流光边 + 内层玻璃底 */
  background:
    linear-gradient(165deg, var(--card-bg, rgba(15, 34, 64, 0.82)), rgba(8, 16, 32, 0.9)) padding-box,
    linear-gradient(
        135deg,
        var(--card-border, rgba(0, 212, 255, 0.5)),
        var(--card-border, rgba(124, 92, 255, 0.25)),
        var(--card-border, rgba(0, 212, 255, 0.35))
      )
      border-box;
  border: 1px solid transparent;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow:
    0 8px 28px rgba(0, 0, 0, 0.38),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
  overflow: hidden;
  transition: box-shadow 0.3s ease, filter 0.3s ease;
  min-height: 0;

  &:hover {
    box-shadow:
      0 12px 36px rgba(0, 0, 0, 0.46),
      0 0 28px rgba(0, 212, 255, 0.14);
  }
}

.card-glow {
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 30% 20%, rgba(0, 212, 255, 0.07), transparent 70%);
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.5s ease;
}

.float-card-inner:hover .card-glow {
  opacity: 1;
}

.card-header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 13px;
  letter-spacing: 1px;
  color: $text-sub;
}

.card-body {
  flex: 1;
  min-height: 0;
}

.card-footer {
  flex-shrink: 0;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(0, 212, 255, 0.1);
}
</style>
