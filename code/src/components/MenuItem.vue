<script setup>
import {QItem,QItemLabel} from "quasar";
import {computed} from "vue";

const props = defineProps({
  link:{type:Object,required:true}
})
</script>

<template>
  <QItem :clickable="$route.name !== link.to.name"
         :to="link.to"
         :disable="link.disabled"
         exact>
    <QItemLabel>{{link.title}}</QItemLabel>
  </QItem>
</template>
