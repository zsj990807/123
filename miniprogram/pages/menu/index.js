const { CATS, listByCat } = require('../../utils/dishes')
const { syncTabBar } = require('../../utils/ui')

Page({
  data: {
    cats: CATS,
    current: 0,
    list: []
  },

  onLoad() {
    this.loadCat(0)
  },

  onShow() {
    // 自定义 tabBar 不会自动同步高亮，切回本页时手动同步
    syncTabBar(this, 0)
  },

  loadCat(index) {
    const list = listByCat(CATS[index].key).map((dish, i) => {
      const item = Object.assign({}, dish)
      item.tilt = i % 2 ? 'tilt-r' : 'tilt-l'
      return item
    })
    this.setData({ current: index, list })
  },

  selectCat(e) {
    const index = Number(e.currentTarget.dataset.index)
    if (index === this.data.current) return
    this.loadCat(index)
  },

  openDetail(e) {
    const index = Number(e.currentTarget.dataset.index)
    const dish = this.data.list[index]
    if (!dish) return
    wx.navigateTo({ url: '/pages/detail/index?id=' + dish.id })
  }
})
