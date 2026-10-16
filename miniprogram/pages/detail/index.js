const repo = require('../../utils/repo')
const store = require('../../utils/store')

Page({
  data: {
    id: '',
    dish: null,
    loading: true,
    qty: 1
  },

  async onLoad(query) {
    const id = query.id || ''
    this.setData({ id })

    const dish = await repo.fetchDish(id)
    this.setData({ dish, loading: false })
    if (dish) {
      wx.setNavigationBarTitle({ title: dish.name })
    }
  },

  incQty() {
    this.setData({ qty: Math.min(this.data.qty + 1, 99) })
  },

  decQty() {
    this.setData({ qty: Math.max(this.data.qty - 1, 1) })
  },

  addToOrder() {
    const dish = this.data.dish
    if (!dish) return
    const qty = this.data.qty
    store.addItem(dish, qty)
    wx.showToast({ title: `已加入点单 ×${qty}`, icon: 'none' })
    this.setData({ qty: 1 })
  },

  goEdit() {
    const dish = this.data.dish
    if (!dish || !dish.id) return
    // dishEdit 是 tabBar 页，只能 switchTab；用 storage 传递要编辑的 id
    wx.setStorageSync('ning.editDishId', dish.id)
    wx.switchTab({ url: '/pages/dishEdit/index' })
  }
})
