/** 自定义 tabBar 选中态同步：在 tab 页的 onShow 中调用 */
function syncTabBar(page, selected) {
  if (typeof page.getTabBar !== 'function') return
  const bar = page.getTabBar()
  if (bar) bar.setData({ selected })
}

function toast(title, icon) {
  wx.showToast({ title, icon: icon || 'none' })
}

module.exports = { syncTabBar, toast }
