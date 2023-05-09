<script setup>
import {data} from "autoprefixer";
import MenuItem from "../components/MenuItem.vue";
import {
  QBtn,
  QDrawer,
  QExpansionItem,
  QHeader,
  QItemLabel,
  QLayout,
  QList,
  QPage,
  QPageContainer,
  QSeparator,
  QToolbar,
  QToolbarTitle
} from "quasar";
import modelData from 'src/assets/modelDataMap.json'
import {computed, ref} from 'vue'

const leftDrawerOpen = ref(false)
const links = computed(() => {
  return [{
    title:'Startseite',
    disabled:false,
    to:{name:'Index'}
  },{
    title:'Computational Graph',
    disabled:false,
    to:{name:'Graph'}
  },
  ...['simple','complex'].map(type => {
    const children = Object.keys(modelData[type]).filter(key => key !== 'title').reduce((acc,name) => {
      const data = modelData[type][name]
      const disabled = `disabled` in data && data.disabled
      return [...acc,{title:data.title,disabled,to:{name}}]
    },[])
    return {
      title: modelData[type].title,
      type,
      children
    }
  },{})]
})
</script>

<template>
  <QLayout view="lHh Lpr lFf">
    <QHeader elevated>
      <QToolbar>
        <QBtn flat
              dense
              round
              icon="menu"
              aria-label="Menu"
              @click="leftDrawerOpen = !leftDrawerOpen" />

        <QToolbarTitle>
           Web Neural Network API
        </QToolbarTitle>

        <div>Webtech Workshop 2023</div>
      </QToolbar>
    </QHeader>

    <QDrawer v-model="leftDrawerOpen"
             show-if-above
             bordered>
      <QList separator>
        <QItemLabel header>Navigation</QItemLabel>
        <template v-for="link in links" :key="link.title">
          <QExpansionItem v-if="'children' in link"
                          expand-separator
                          :model-value="$route.path.includes(`/${link.type}/`)"
                          :links="link.children"
                          :label="link.title">
            <QList v-for="child in link.children"
                   :key="child.name"
                   separator>
              <MenuItem :link="child" />
            </QList>
          </QExpansionItem>
          <MenuItem v-else :link="link" />
        </template>
      </QList>
      <QSeparator />
    </QDrawer>

    <QPageContainer>
      <QPage padding :class="$route.name">
        <router-view :key="$route.name" />
      </QPage>
    </QPageContainer>
  </QLayout>
</template>
