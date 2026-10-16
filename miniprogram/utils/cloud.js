const config = require('./config')

const COLLECTIONS = {
  DISHES: 'dishes',
  ORDERS: 'orders'
}

/** 是否具备可用的云开发环境（基础库支持 + 已配置环境 ID） */
function isReady() {
  return !!(wx.cloud && config.CLOUD_ENV)
}

/** 初始化云开发，返回是否初始化成功 */
function init() {
  if (!isReady()) return false
  wx.cloud.init({ env: config.CLOUD_ENV, traceUser: true })
  return true
}

/** 取数据库实例；未配置云环境时返回 null，调用方需降级处理 */
function db() {
  return isReady() ? wx.cloud.database() : null
}

module.exports = { COLLECTIONS, isReady, init, db }
