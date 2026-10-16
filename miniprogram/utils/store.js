const ORDER_KEY = 'ning.order.v1'
const USER_DISH_KEY = 'ning.userDishes.v1'

/** 点单（本地暂存，P3 起改为提交到云数据库 orders） */
function getOrder() {
  return wx.getStorageSync(ORDER_KEY) || []
}

function setOrder(order) {
  wx.setStorageSync(ORDER_KEY, order || [])
}

/** 加菜到点单：已存在则累加份数 */
function addItem(dish, qty) {
  if (!dish || !dish.id) return []
  const order = getOrder()
  const idx = order.findIndex(x => x.id === dish.id)
  if (idx >= 0) {
    order[idx].qty = Math.min((order[idx].qty || 0) + qty, 99)
  } else {
    order.push({
      id: dish.id,
      name: dish.name,
      img: dish.img,
      minutes: dish.minutes || 0,
      qty
    })
  }
  setOrder(order)
  return order
}

/** 更新某道菜的份数 */
function updateQty(id, qty) {
  const order = getOrder()
  const idx = order.findIndex(x => x.id === id)
  if (idx >= 0) {
    order[idx].qty = Math.max(1, Math.min(qty, 99))
    setOrder(order)
  }
  return order
}

/** 移除某道菜 */
function removeItem(id) {
  const order = getOrder().filter(x => x.id !== id)
  setOrder(order)
  return order
}

/** 清空点单 */
function clearOrder() {
  setOrder([])
  return []
}

/** 汇总统计 */
function summarize(order) {
  const list = order || getOrder()
  const dishCount = list.length
  const totalQty = list.reduce((s, x) => s + (x.qty || 0), 0)
  const estMinutes = list.reduce((m, x) => Math.max(m, x.minutes || 0), 0)
  return { dishCount, totalQty, estMinutes }
}

/** 用户自建菜品（P4 起写入云数据库 dishes） */
function getUserDishes() {
  return wx.getStorageSync(USER_DISH_KEY) || []
}

function setUserDishes(list) {
  wx.setStorageSync(USER_DISH_KEY, list || [])
}

module.exports = {
  getOrder, setOrder, addItem, updateQty, removeItem, clearOrder, summarize,
  getUserDishes, setUserDishes
}
