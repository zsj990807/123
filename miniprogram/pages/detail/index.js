const repo = require('../../utils/repo')

Page({
  data: {
    id: '',
    dish: null,
    loading: true
  },

  async onLoad(query) {
    const id = query.id || ''
    this.setData({ id })

    const dish = await repo.fetchDish(id)
    this.setData({ dish, loading: false })
    if (dish) {
      wx.setNavigationBarTitle({ title: dish.name })
    }
  }
})
