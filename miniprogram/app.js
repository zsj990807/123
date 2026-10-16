const cloud = require('./utils/cloud')

App({
  globalData: {
    cloudReady: false
  },

  onLaunch() {
    if (!wx.cloud) {
      console.warn('[cloud] 当前基础库不支持云开发，已降级为本地内置数据')
      return
    }
    if (!cloud.init()) {
      console.warn('[cloud] 未配置云开发环境 ID（见 utils/config.js），当前使用本地内置数据')
      return
    }
    this.globalData.cloudReady = true
    console.info('[cloud] 云开发初始化完成')
  }
})
