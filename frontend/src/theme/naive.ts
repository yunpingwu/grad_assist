import type { GlobalThemeOverrides } from 'naive-ui'

/**
 * Naive UI 纸墨主题：把组件库配色/圆角/字体对齐 styles/main.css 的「纸墨 · 学术档案」色板。
 * 颜色与 CSS 变量保持同源（改主色时需两处同步），组件行为样式全部走 themeOverrides，
 * 不再用 :deep 覆盖内部类。
 */
export const paperInkTheme: GlobalThemeOverrides = {
  common: {
    // 字体：正文与全局一致
    fontFamily:
      "'Noto Sans SC', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Microsoft YaHei UI', sans-serif",

    // 点缀色：朱砂主色 + 松绿 ok + 赭红 bad + 枯黄 warning
    primaryColor: '#c23b22',
    primaryColorHover: '#d3543a',
    primaryColorPressed: '#9e2f1a',
    primaryColorSuppl: '#c23b22',
    errorColor: '#a3271f',
    errorColorHover: '#b5453a',
    errorColorPressed: '#84201a',
    errorColorSuppl: '#a3271f',
    successColor: '#27623c',
    successColorHover: '#3a7a52',
    successColorPressed: '#1e4c2f',
    successColorSuppl: '#27623c',
    warningColor: '#a8761f',
    warningColorHover: '#bb8b35',
    warningColorPressed: '#886017',
    warningColorSuppl: '#a8761f',
    infoColor: '#5a5244',
    infoColorHover: '#6d6455',
    infoColorPressed: '#463f33',
    infoColorSuppl: '#5a5244',

    // 纸面 / 墨字 / 界行
    borderRadius: '3px',
    bodyColor: '#f5efe3',
    cardColor: '#fbf7ee',
    modalColor: '#fbf7ee',
    popoverColor: '#fbf7ee',
    tableColor: '#fbf7ee',
    inputColor: '#fbf7ee',
    actionColor: '#ece2cd',
    borderColor: '#dfd5c2',
    dividerColor: '#dfd5c2',
    textColorBase: '#211d16',
    textColor1: '#211d16',
    textColor2: '#5a5244',
    textColor3: '#8a8071',
    placeholderColor: '#8a8071',
    closeIconColor: '#8a8071',
  },
  // 上传拖拽区：虚线界行 → 悬停转朱砂，底色透明融入纸面卡片
  Upload: {
    draggerBorder: '1.5px dashed #8a8071',
    draggerBorderHover: '1.5px dashed #c23b22',
    draggerColor: 'rgba(194, 59, 34, 0.06)',
  },
  // 弹窗/消息贴近纸质，阴影收敛
  Dialog: {
    boxShadow: '0 10px 30px rgba(33, 29, 22, 0.12)',
  },
  Message: {
    boxShadow: '0 6px 18px rgba(33, 29, 22, 0.1)',
  },
}
