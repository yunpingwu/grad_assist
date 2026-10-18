import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import Components from 'unplugin-vue-components/vite'
import { NaiveUiResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    vue(),
    // Naive UI 按需引入：模板内 <n-xxx> 由 Components 自动注册（样式为 CSS-in-JS，随组件注入）。
    // useMessage/useDialog 等组合式 API 需在使用处从 'naive-ui' 显式导入。
    Components({
      resolvers: [NaiveUiResolver()],
      dts: 'src/components.d.ts',
    }),
  ],
  server: {
    port: 5173,
    open: false,
  },
})
