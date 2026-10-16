const { syncTabBar } = require('../../utils/ui')

Page({
  data: {},

  onShow() {
    syncTabBar(this, 2)
  }
})
