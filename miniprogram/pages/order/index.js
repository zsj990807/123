const { syncTabBar } = require('../../utils/ui')
const store = require('../../utils/store')
const cloud = require('../../utils/cloud')

Page({
  data: {
    list: [],
    note: '',
    summary: { dishCount: 0, totalQty: 0, estMinutes: 0 },
    submitting: false
  },

  onShow() {
    syncTabBar(this, 1)
    this.refresh()
  },

  refresh() {
    const list = store.getOrder()
    this.setData({ list, summary: store.summarize(list) })
  },

  incQty(e) {
    const id = e.currentTarget.dataset.id
    const item = this.data.list.find(x => x.id === id)
    if (!item) return
    store.updateQty(id, item.qty + 1)
    this.refresh()
  },

  decQty(e) {
    const id = e.currentTarget.dataset.id
    const item = this.data.list.find(x => x.id === id)
    if (!item) return
    store.updateQty(id, item.qty - 1)
    this.refresh()
  },

  remove(e) {
    const id = e.currentTarget.dataset.id
    store.removeItem(id)
    wx.showToast({ title: '已移除', icon: 'none' })
    this.refresh()
  },

  onNoteInput(e) {
    this.setData({ note: e.detail.value })
  },

  goMenu() {
    wx.switchTab({ url: '/pages/menu/index' })
  },

  async submitOrder() {
    if (this.data.submitting) return
    if (!this.data.list.length) {
      wx.showToast({ title: '还没有点菜', icon: 'none' })
      return
    }

    this.setData({ submitting: true })
    const payload = {
      items: this.data.list,
      note: this.data.note,
      summary: this.data.summary,
      createdAt: Date.now()
    }

    try {
      const db = cloud.db()
      if (db) {
        await db.collection(cloud.COLLECTIONS.ORDERS).add({ data: payload })
      }
      store.clearOrder()
      wx.showToast({ title: '已提交给主厨', icon: 'success' })
      this.setData({ note: '' })
    } catch (e) {
      console.warn('[order] 提交失败', e)
      wx.showToast({ title: '云端暂不可用，已保留本地点单', icon: 'none' })
    } finally {
      this.setData({ submitting: false })
      this.refresh()
    }
  }
})
