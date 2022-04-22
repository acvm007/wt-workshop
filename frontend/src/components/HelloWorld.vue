<script setup>
import { ref } from 'vue'
import {multiplyMatrices} from "../scripts/context.js";

defineProps({
  msg: String
})

const count = ref(0)
const result= ref(null)

function multiply(){
  result.value=multiplyMatrices([3,4],[4,3],[3,3],count.value)
}
</script>

<template>
  <h1>{{ msg }}</h1>
  <label for="test">Hier den initialen Wert der Matrix eingeben</label>
  <input v-model.number="count" id="test">
  <button type="button" @click="multiply">Starten</button>
  <template v-if="result">
    <h3>Resultierende Matrix</h3>
    <div>{{result}}</div>
  </template>
</template>

<style scoped>
a {
  color: #42b983;
}
</style>
