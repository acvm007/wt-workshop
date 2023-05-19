<script setup>
import ModelDataDefinitions from '/src/assets/modelDataMap.json'
import WebNNBase from "components/WebNNBase.vue";
import {computed} from "vue";
import {useRoute} from "vue-router";

const route = useRoute()
const modelData = computed(() => {
  return ModelDataDefinitions.simple[route.name]
})
</script>

<template>
  <WebNNBase :model-data="modelData" />
</template>
