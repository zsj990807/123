Component({
  data: {
    selected: 0,
    list: [
      { pagePath: '/pages/menu/index', text: '专属菜单' },
      { pagePath: '/pages/order/index', text: '我的点单' },
      { pagePath: '/pages/dishEdit/index', text: '添加菜品' }
    ]
  },

  methods: {
    onTap(e) {
      const { path, index } = e.currentTarget.dataset
      if (index === this.data.selected) return
      // 立即反馈选中态，目标页 onShow 会再做一次同步
      this.setData({ selected: index })
      wx.switchTab({ url: path })
    }
  }
})
