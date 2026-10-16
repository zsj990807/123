const { syncTabBar } = require('../../utils/ui')
const repo = require('../../utils/repo')
const cloud = require('../../utils/cloud')

const CAT_KEYS = ['main', 'veg', 'soup']
const CAT_NAMES = ['主菜', '蔬菜', '炖汤']

/** 把 wx API 包装成 Promise，便于 await */
function pickMedia() {
  return new Promise((resolve, reject) => {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: res => resolve(res.tempFiles[0].tempFilePath),
      fail: reject
    })
  })
}

function compress(src) {
  return new Promise((resolve, reject) => {
    wx.compressImage({
      src, quality: 72,
      success: res => resolve(res.tempFilePath),
      fail: () => resolve(src) // 压缩失败就用原图
    })
  })
}

function uploadToCloud(cloudPath, filePath) {
  return new Promise((resolve, reject) => {
    wx.cloud.uploadFile({ cloudPath, filePath, success: resolve, fail: reject })
  })
}

Page({
  data: {
    mode: 'add',
    id: '',
    name: '',
    desc: '',
    time: '',
    minutes: '',
    tag: '',
    flavor: '',
    serving: '',
    catIndex: 0,
    catNames: CAT_NAMES,
    imgUrl: '',
    cloudFileID: '',
    chef: '',
    ingredients: [{ name: '', amount: '' }],
    steps: [''],
    saving: false
  },

  async onLoad() {
    // tabBar 页首次创建；编辑模式由 onShow 中的 storage 触发
    wx.setNavigationBarTitle({ title: '添加菜品' })
  },

  onShow() {
    syncTabBar(this, 2)
    const editId = wx.getStorageSync('ning.editDishId')
    if (editId) {
      wx.removeStorageSync('ning.editDishId')
      this.loadDish(editId)
    }
  },

  async loadDish(id) {
    const dish = await repo.fetchDish(id)
    if (!dish) return
    const catIndex = Math.max(0, CAT_KEYS.indexOf(dish.cat))
    this.setData({
      mode: 'edit',
      id: dish.id,
      name: dish.name || '',
      desc: dish.desc || '',
      time: dish.time || '',
      minutes: dish.minutes ? String(dish.minutes) : '',
      tag: dish.tag || '',
      flavor: dish.flavor || '',
      serving: dish.serving || '',
      catIndex,
      imgUrl: dish.img || '',
      cloudFileID: dish.img && dish.img.indexOf('cloud://') === 0 ? dish.img : '',
      chef: dish.chef || '',
      ingredients: (dish.ingredients && dish.ingredients.length)
        ? dish.ingredients.map(x => ({ name: x.name || '', amount: x.amount || '' }))
        : [{ name: '', amount: '' }],
      steps: (dish.steps && dish.steps.length) ? dish.steps.slice() : ['']
    })
    wx.setNavigationBarTitle({ title: '编辑菜品' })
  },

  /** 通用输入：data-field 指定字段 */
  onFieldInput(e) {
    const field = e.currentTarget.dataset.field
    this.setData({ [field]: e.detail.value })
  },

  onCatChange(e) {
    this.setData({ catIndex: Number(e.detail.value) })
  },

  onIngNameInput(e) {
    const i = Number(e.currentTarget.dataset.index)
    this.setData({ [`ingredients[${i}].name`]: e.detail.value })
  },

  onIngAmountInput(e) {
    const i = Number(e.currentTarget.dataset.index)
    this.setData({ [`ingredients[${i}].amount`]: e.detail.value })
  },

  addIngredient() {
    const list = this.data.ingredients.concat({ name: '', amount: '' })
    this.setData({ ingredients: list })
  },

  removeIngredient(e) {
    const i = Number(e.currentTarget.dataset.index)
    const list = this.data.ingredients.filter((_, idx) => idx !== i)
    this.setData({ ingredients: list.length ? list : [{ name: '', amount: '' }] })
  },

  onStepInput(e) {
    const i = Number(e.currentTarget.dataset.index)
    this.setData({ [`steps[${i}]`]: e.detail.value })
  },

  addStep() {
    this.setData({ steps: this.data.steps.concat('') })
  },

  removeStep(e) {
    const i = Number(e.currentTarget.dataset.index)
    const list = this.data.steps.filter((_, idx) => idx !== i)
    this.setData({ steps: list.length ? list : [''] })
  },

  async chooseImage() {
    try {
      const temp = await pickMedia()
      const filePath = await compress(temp)
      this.setData({ imgUrl: filePath, cloudFileID: '' })

      if (cloud.isReady()) {
        try {
          const res = await uploadToCloud('dishes/' + Date.now() + '.jpg', filePath)
          this.setData({ cloudFileID: res.fileID, imgUrl: res.fileID })
        } catch (e) {
          console.warn('[dishEdit] 图片上传云端失败，保留本地临时图', e)
          wx.showToast({ title: '云端上传失败，图片为本地临时图', icon: 'none' })
        }
      }
    } catch (e) {
      // 用户取消选图
    }
  },

  async save() {
    if (this.data.saving) return
    const d = this.data
    if (!d.name.trim()) {
      wx.showToast({ title: '请填写菜名', icon: 'none' })
      return
    }

    this.setData({ saving: true })

    const ingredients = d.ingredients
      .map(x => ({ name: x.name.trim(), amount: x.amount.trim() }))
      .filter(x => x.name)
    const steps = d.steps.map(x => x.trim()).filter(x => x)
    const catKey = CAT_KEYS[d.catIndex]

    const payload = {
      name: d.name.trim(),
      desc: d.desc.trim(),
      time: d.time.trim(),
      minutes: parseInt(d.minutes, 10) || 0,
      tag: d.tag.trim(),
      flavor: d.flavor.trim(),
      serving: d.serving.trim(),
      cat: catKey,
      img: d.cloudFileID || d.imgUrl,
      chef: d.chef.trim(),
      ingredients,
      steps,
      builtin: false
    }

    try {
      const db = cloud.db()
      if (!db) throw new Error('cloud-off')

      if (d.mode === 'edit') {
        await db.collection(cloud.COLLECTIONS.DISHES).doc(d.id).update({ data: payload })
        wx.showToast({ title: '已保存修改', icon: 'success' })
      } else {
        const id = 'dish-' + Date.now()
        await db.collection(cloud.COLLECTIONS.DISHES).doc(id).set({
          data: Object.assign({ order: 99 }, payload)
        })
        wx.showToast({ title: '已添加菜品', icon: 'success' })
      }

      setTimeout(() => wx.switchTab({ url: '/pages/menu/index' }), 700)
    } catch (e) {
      console.warn('[dishEdit] 保存失败', e)
      wx.showToast({ title: '保存失败，请检查云开发权限', icon: 'none' })
    } finally {
      this.setData({ saving: false })
    }
  }
})
