/**
 * 内置菜品数据（源自 HTML 原型）
 * img 指向列表缩略图；详情页大图在 P1 接入云存储后替换
 */

const CATS = [
  { key: 'main', name: '主菜' },
  { key: 'veg', name: '蔬菜' },
  { key: 'soup', name: '炖汤' }
]

const DISHES = {
  main: [
    {
      id: 'shrimp-egg',
      name: '虾仁滑蛋',
      desc: '鸡蛋柔嫩，虾仁清甜。',
      time: '约 15 分钟',
      minutes: 15,
      tag: '鲜香嫩滑',
      flavor: '清淡鲜甜',
      serving: '1-2 人',
      img: '/assets/thumb-shrimp-egg.jpg',
      chef: '「蛋要嫩，铲子就得懒一点——少翻，慢推。」',
      ingredients: [
        { name: '鸡蛋', amount: '3 枚' },
        { name: '鲜虾仁', amount: '120 g' },
        { name: '小葱', amount: '1 根' },
        { name: '盐', amount: '少许' },
        { name: '花生油', amount: '1 勺' }
      ],
      steps: [
        '虾仁用盐抓洗后吸干水分，热锅滑油十秒即捞出。',
        '蛋液加盐打散，倒入留有余温的锅中。',
        '转小火，铲子从边缘缓慢推向中间，蛋液半凝固时放回虾仁。',
        '撒葱花，翻两下立刻出锅，全程不超过一分钟。'
      ]
    },
    {
      id: 'chicken',
      name: '马哥超绝炒鸡公',
      desc: '超水准的招牌菜不解释',
      time: '约 30 分钟',
      minutes: 30,
      tag: '香辣 回味无穷',
      flavor: '香辣',
      serving: '2-3 人',
      img: '/assets/thumb-chicken.jpg',
      chef: '「这道菜不解释，吃过的都懂。」',
      ingredients: [
        { name: '走地公鸡', amount: '800 g' },
        { name: '干辣椒', amount: '20 g' },
        { name: '花椒', amount: '8 g' },
        { name: '大蒜', amount: '1 头' },
        { name: '姜', amount: '1 块' },
        { name: '生抽', amount: '2 勺' }
      ],
      steps: [
        '鸡斩小块，用料酒和生抽腌十分钟。',
        '热油炝香干辣椒、花椒、蒜与姜。',
        '下鸡块生炒至变色，加少量水加盖焖十分钟。',
        '开大火收汁，收到能挂住勺子为止。'
      ]
    },
    {
      id: 'clams',
      name: '薄荷炒花甲',
      desc: '吮指美味的炒花甲',
      time: '约 30 分钟',
      minutes: 30,
      tag: '鲜',
      flavor: '鲜辣',
      serving: '2 人',
      img: '/assets/thumb-clams.jpg',
      chef: '「薄荷是广东人的私心，去腥，还提鲜。」',
      ingredients: [
        { name: '花甲', amount: '500 g' },
        { name: '新鲜薄荷', amount: '1 把' },
        { name: '小米辣', amount: '2 个' },
        { name: '蒜末', amount: '2 勺' },
        { name: '生抽', amount: '1 勺' }
      ],
      steps: [
        '花甲用盐水养两小时，让它吐净沙。',
        '热锅爆香蒜末与小米辣。',
        '下花甲，加盖焖到全部开口。',
        '淋生抽，撒薄荷，翻两下出锅。'
      ]
    },
    {
      id: 'tofu',
      name: '打抛肉末煎豆腐',
      desc: '一秒回到泰兰德',
      time: '约 30 分钟',
      minutes: 30,
      tag: '下饭',
      flavor: '咸鲜微辣',
      serving: '2 人',
      img: '/assets/thumb-tofu.jpg',
      chef: '「一勺浇在饭上，一秒回到曼谷街头。」',
      ingredients: [
        { name: '猪前腿肉末', amount: '200 g' },
        { name: '老豆腐', amount: '1 块' },
        { name: '九层塔', amount: '1 把' },
        { name: '小米辣', amount: '2 个' },
        { name: '鱼露', amount: '1 勺' }
      ],
      steps: [
        '豆腐切厚片，两面煎至金黄盛起。',
        '肉末炒散，加鱼露与蚝油炒出香。',
        '放小米辣和九层塔快速翻炒。',
        '趁热浇在煎好的豆腐上。'
      ]
    }
  ],

  veg: [
    {
      id: 'broccoli',
      name: '蒜蓉西兰花',
      desc: '脆嫩清甜，蒜香扑鼻。',
      time: '约 10 分钟',
      minutes: 10,
      tag: '清爽',
      flavor: '清爽',
      serving: '2 人',
      img: '/assets/thumb-broccoli.jpg',
      chef: '「脆，才是蔬菜的礼貌。」',
      ingredients: [
        { name: '西兰花', amount: '1 颗' },
        { name: '大蒜', amount: '5 瓣' },
        { name: '盐', amount: '少许' },
        { name: '蚝油', amount: '半勺' }
      ],
      steps: [
        '西兰花切小朵，用盐水泡洗。',
        '沸水焯三十秒，立刻过冷水锁色。',
        '蒜片小火爆香至微黄。',
        '下西兰花，加盐和蚝油翻匀即出。'
      ]
    },
    {
      id: 'cabbage',
      name: '手撕包菜',
      desc: '锅气十足的干锅风味',
      time: '约 12 分钟',
      minutes: 12,
      tag: '微辣 下饭',
      flavor: '微辣',
      serving: '2 人',
      img: '/assets/thumb-cabbage.jpg',
      chef: '「锅要够热，包菜下去得要有响声。」',
      ingredients: [
        { name: '包菜', amount: '半颗' },
        { name: '腊肉', amount: '50 g' },
        { name: '干辣椒', amount: '4 个' },
        { name: '蒜末', amount: '1 勺' },
        { name: '生抽', amount: '1 勺' }
      ],
      steps: [
        '包菜手撕成块，洗净沥干水。',
        '腊肉切薄片，煸出油。',
        '下蒜末与干辣椒爆香。',
        '大火下包菜快炒三十秒，淋生抽。'
      ]
    },
    {
      id: 'babycabbage',
      name: '上汤娃娃菜',
      desc: '汤鲜味美，一口回甘',
      time: '约 15 分钟',
      minutes: 15,
      tag: '清甜',
      flavor: '清甜',
      serving: '2 人',
      img: '/assets/thumb-babycabbage.jpg',
      chef: '「汤要好，菜才敢做得这么清淡。」',
      ingredients: [
        { name: '娃娃菜', amount: '2 颗' },
        { name: '皮蛋', amount: '1 个' },
        { name: '咸蛋', amount: '半个' },
        { name: '高汤', amount: '400 ml' },
        { name: '枸杞', amount: '少许' }
      ],
      steps: [
        '皮蛋与咸蛋切丁，先煮进高汤里出味。',
        '娃娃菜纵向剖开，下汤煮三分钟。',
        '加枸杞再煮一分钟。',
        '连汤一起装碗，汤别收干。'
      ]
    }
  ],

  soup: [
    {
      id: 'tomato-beef',
      name: '番茄牛腩汤',
      desc: '浓郁微酸，暖到胃里。',
      time: '约 90 分钟',
      minutes: 90,
      tag: '暖胃',
      flavor: '浓郁微酸',
      serving: '3 人',
      img: '/assets/thumb-tomato-beef.jpg',
      chef: '「酸要酸得有厚度，靠的是时间。」',
      ingredients: [
        { name: '牛腩', amount: '500 g' },
        { name: '番茄', amount: '4 个' },
        { name: '洋葱', amount: '半个' },
        { name: '姜', amount: '3 片' },
        { name: '盐', amount: '适量' }
      ],
      steps: [
        '牛腩焯水，与两个番茄同炖九十分钟。',
        '剩下两个番茄另起锅炒出沙。',
        '把牛腩汤倒进番茄沙里，合煮二十分钟。',
        '加盐调味，起锅前撒葱花。'
      ]
    },
    {
      id: 'corn-rib',
      name: '玉米排骨汤',
      desc: '清甜不腻，慢火细炖',
      time: '约 60 分钟',
      minutes: 60,
      tag: '清甜',
      flavor: '清甜',
      serving: '3 人',
      img: '/assets/thumb-corn-rib.jpg',
      chef: '「不勾芡、不重口，就是要喝那口清甜。」',
      ingredients: [
        { name: '排骨', amount: '500 g' },
        { name: '甜玉米', amount: '2 根' },
        { name: '姜', amount: '3 片' },
        { name: '枸杞', amount: '少许' },
        { name: '盐', amount: '适量' }
      ],
      steps: [
        '排骨冷水下锅焯去浮沫。',
        '玉米切段，与排骨、姜片一同下锅。',
        '转小火慢炖六十分钟，中途不揭盖。',
        '出锅前放枸杞，只加一点盐。'
      ]
    },
    {
      id: 'cantonese-soup',
      name: '老火例汤',
      desc: '每日限定，慢炖四小时',
      time: '约 240 分钟',
      minutes: 240,
      tag: '慢炖',
      flavor: '醇厚',
      serving: '3-4 人',
      img: '/assets/thumb-cantonese-soup.jpg',
      chef: '「例汤没有菜单，只有当天。」',
      ingredients: [
        { name: '猪骨', amount: '500 g' },
        { name: '老鸡', amount: '半只' },
        { name: '时令药材', amount: '1 份' },
        { name: '姜', amount: '3 片' },
        { name: '盐', amount: '适量' }
      ],
      steps: [
        '猪骨与老鸡冷水下锅焯透。',
        '与药材、姜片一同入砂锅，水一次加足。',
        '小火煲足四小时，全程不揭盖。',
        '关火后只加一点盐，喝的是原味。'
      ]
    }
  ]
}

function catName(key) {
  const hit = CATS.find(c => c.key === key)
  return hit ? hit.name : ''
}

function listByCat(key) {
  return DISHES[key] || []
}

function findById(id) {
  const keys = Object.keys(DISHES)
  for (let i = 0; i < keys.length; i++) {
    const hit = DISHES[keys[i]].find(d => d.id === id)
    if (hit) return hit
  }
  return null
}

module.exports = { CATS, DISHES, catName, listByCat, findById }
