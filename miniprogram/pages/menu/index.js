const repo = require('../../utils/repo')
const { CATS } = require('../../utils/dishes')
const { syncTabBar } = require('../../utils/ui')

Page({
  data: {
    cats: CATS,
    current: 0,
    list: [],
    loading: true
  },

  async onLoad() {
    // 首次进入把内置菜品写入云端；云环境不可用时会直接跳过
    try {
      await repo.ensureSeeded()
    } catch (e) {
      console.warn('[menu] 初始化内置菜品失败', e)
    }
    await this.loadCat(0)
  },

  onShow() {
    // 自定义 tabBar 不会自动同步高亮，切回本页时手动同步
    syncTabBar(this, 0)
  },

  async loadCat(index) {
    const key = CATS[index].key
    this.setData({ current: index, loading: true })

    let list = []
    try {
      list = await repo.fetchDishes(key)
    } catch (e) {
      console.warn('[menu] 云端读取失败，降级为内置数据', e)
      list = repo.localDishes(key)
      wx.showToast({ title: '云端读取失败，已用内置菜单', icon: 'none' })
    }

    list = list.map((dish, i) => {
      const item = Object.assign({}, dish)
      item.tilt = i % 2 ? 'tilt-r' : 'tilt-l'
      return item
    })
    this.setData({ list, loading: false })
  },

  selectCat(e) {
    const index = Number(e.currentTarget.dataset.index)
    if (index === this.data.current) return
    this.loadCat(index)
  },

  openDetail(e) {
    const index = Number(e.currentTarget.dataset.index)
    const dish = this.data.list[index]
    if (!dish || !dish.id) return
    wx.navigateTo({ url: '/pages/detail/index?id=' + dish.id })
  }
})
