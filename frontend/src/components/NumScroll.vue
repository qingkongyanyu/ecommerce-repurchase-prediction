<template>
  <span class="num-rolling" ref="numRef">{{ displayValue }}</span>
</template>

<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import { gsap } from 'gsap'

const props = defineProps({
  // 目标数值
  target: {
    type: Number,
    default: 0,
  },
  // 是否带千位分隔符
  withComma: {
    type: Boolean,
    default: true,
  },
  // 小数位数
  decimals: {
    type: Number,
    default: 0,
  },
  // 前缀
  prefix: {
    type: String,
    default: '',
  },
  // 后缀
  suffix: {
    type: String,
    default: '',
  },
  // 动画持续时间（秒）
  duration: {
    type: Number,
    default: 1.5,
  },
  // 是否自动开始
  autoStart: {
    type: Boolean,
    default: true,
  },
})

const numRef = ref(null)
const currentValue = ref(0)
const displayValue = computed(() => {
  const formatted = currentValue.value.toFixed(props.decimals)
  const withComma = props.withComma
    ? Number(formatted).toLocaleString('zh-CN', {
        minimumFractionDigits: props.decimals,
        maximumFractionDigits: props.decimals,
      })
    : formatted
  return `${props.prefix}${withComma}${props.suffix}`
})

let anim = null

const animate = () => {
  if (anim) {
    anim.kill()
  }

  const targetVal = props.target

  // 如果目标为0，直接显示
  if (targetVal === 0) {
    currentValue.value = 0
    return
  }

  // 从0开始滚动到目标值
  currentValue.value = 0
  anim = gsap.to(currentValue, {
    value: targetVal,
    duration: props.duration,
    ease: 'power2.out',
    overwrite: 'auto',
  })
}

onMounted(() => {
  if (props.autoStart) {
    animate()
  }
})

watch(
  () => props.target,
  (newVal, oldVal) => {
    if (newVal !== oldVal) {
      animate()
    }
  },
  { deep: true }
)
</script>

<style scoped lang="scss">
.num-rolling {
  font-variant-numeric: tabular-nums;
  display: inline-block;
  font-weight: 600;
  letter-spacing: 0.5px;
  background: linear-gradient(135deg, #ffffff 30%, #40a9ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
</style>