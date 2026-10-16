const { findById } = require('../../utils/dishes')

Page({
  data: {
    id: '',
    dish: null
  },

  onLoad(query) {
    const id = query.id || ''
    const dish = findById(id)
    this.setData({ id, dish })
    if (dish) {
      wx.setNavigationBarTitle({ title: dish.name })
    }
  }
})
