import modelData from 'src/assets/modelDataMap.json'

const routes = [
  {
    path: '/',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      {
        path: '',
        name: 'Index',
        component: () => import('pages/IndexPage.vue')
      },{
        path: 'graph',
        name: 'Graph',
        component: () => import('pages/ComputationalGraph.vue')
      },
      ...['simple','complex'].map(type => {
        return {
          path:`/${type}`,
          children:Object.keys(modelData[type]).map(name => {
            return {
              path:name,
              name,
              component: () => import('pages/WebNNSamplesSimple.vue')
            }
          })
        }
      })
    ]
  },

  // Always leave this as last one,
  // but you can also remove it
  {
    path: '/:catchAll(.*)*',
    component: () => import('pages/ErrorNotFound.vue')
  }
]

export default routes
