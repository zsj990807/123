/**
 * 数据访问层：统一收敛「云数据库 / 本地内置数据」两种来源。
 *
 * 云环境不可用（未配置、集合不存在、无权限）时自动降级为本地内置数据，
 * 保证界面在任何情况下都能渲染出来。
 */

const cloud = require('./cloud')
const { DISHES, listByCat, findById } = require('./dishes')

const SEED_FLAG = 'ning.seeded.v1'

/** 本地内置数据（降级用） */
function localDishes(catKey) {
  return listByCat(catKey)
}

function localDish(id) {
  return findById(id)
}

/** 统一 id 字段：云数据库返回的是 _id */
function normalize(doc) {
  if (!doc) return null
  const d = Object.assign({}, doc)
  if (d._id && !d.id) d.id = d._id
  return d
}

/** 把内置菜品整理为云端文档结构 */
function seedDocs() {
  const docs = []
  Object.keys(DISHES).forEach(cat => {
    DISHES[cat].forEach((dish, i) => {
      const d = Object.assign({}, dish)
      d.cat = cat
      d.order = i
      d.builtin = true
      docs.push(d)
    })
  })
  return docs
}

/**
 * 首次运行把内置菜品写入云数据库。
 * 使用固定 _id（即菜品 id），重复执行不会产生重复数据。
 */
async function ensureSeeded() {
  const db = cloud.db()
  if (!db) return false
  if (wx.getStorageSync(SEED_FLAG)) return true

  const col = db.collection(cloud.COLLECTIONS.DISHES)
  const docs = seedDocs()
  let ok = 0

  for (let i = 0; i < docs.length; i++) {
    const d = docs[i]
    try {
      await col.doc(d.id).set({ data: d })
      ok++
    } catch (e) {
      // 该 _id 已被其他用户创建时写入会失败，跳过即可
      console.warn('[repo] 初始化跳过', d.id, e && e.errMsg)
    }
  }

  console.info('[repo] 内置菜品初始化完成', ok + '/' + docs.length)
  if (ok > 0) wx.setStorageSync(SEED_FLAG, true)
  return ok > 0
}

/** 按分类读取菜品；云端不可用时返回内置数据 */
async function fetchDishes(catKey) {
  const db = cloud.db()
  if (!db) return localDishes(catKey)

  const res = await db.collection(cloud.COLLECTIONS.DISHES)
    .where({ cat: catKey })
    .get()

  return (res.data || [])
    .map(normalize)
    .sort((a, b) => (a.order || 0) - (b.order || 0))
}

/** 按 id 读取单个菜品；云端不可用时查内置数据 */
async function fetchDish(id) {
  const db = cloud.db()
  if (!db) return localDish(id)

  try {
    const res = await db.collection(cloud.COLLECTIONS.DISHES).doc(id).get()
    return normalize(res.data) || localDish(id)
  } catch (e) {
    console.warn('[repo] 云端单条读取失败，降级为内置数据', id, e && e.errMsg)
    return localDish(id)
  }
}

module.exports = {
  localDishes,
  localDish,
  ensureSeeded,
  fetchDishes,
  fetchDish
}
