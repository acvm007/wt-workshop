<script setup>
import '@webmachinelearning/webnn-polyfill'
import ArrayFields from "src/components/ArrayFields.vue";
import {QBtn, QForm, QInput, QSeparator,QSpinner} from "quasar";
import {buildGraph} from "src/scripts/webnnFunctions.js";
import {computed, ref} from "vue";

const isComputing = ref(false)
const showFooter = ref(false)
const size = ref(4)
const shape = ref([1,2,2,2])
const values = ref([])
const shapeIsValid = computed(() => {
  const allValid = shape.value.every(val => !!val)
  const correctSize = shape.value.length === size.value
  return size.value > 0 && correctSize && allValid
})
const valuesAreValid = computed(() => {
  return values.value.every(val => val !== null || val !== undefined || val !== '') && values.value.length === size.value
})
const graphResult = ref(null)

function reset(){
  values.value = []
  showFooter.value = false
}

async function test(){
  isComputing.value = true
  showFooter.value = true
  graphResult.value = await buildGraph(shape.value,values.value)
  isComputing.value = false
}
</script>

<template>
  <h3 class="q-ma-none">Computational Graph</h3>
  <QSeparator class="q-my-sm" />
  <QForm @submit="test">
    <section class="row items-center justify-between">
      <QInput v-model.number="size"
              v-if="false"
              type="number"
              class="q-mt-lg"
              outlined
              label="Anzahl der Tensor Dimensionen"
              :rules="[
              val => !!val || 'Die Größe des Tensors muss definiert sein',
              val => val > 0 || 'Die Größe des Tensors muss größer 0 sein'
            ]" />
      <ArrayFields v-if="size > 0 && false"
                   :inputs="shape"
                   :input-number="size"
                   label="Dimensionen des Tensors"
                   required-error="Alle Dimensionen sind erforderlich!" />
    </section>
    <ArrayFields v-if="shapeIsValid"
                 :inputs="values"
                 label="Eingabewerte des Graphens"
                 required-error="Alle Eingabewerte erforderlich!"
                 :input-number="shape.length" />
    <QBtn type="submit"
          :disable="!shapeIsValid || !valuesAreValid"
          label="Graph berechnen" />
  </QForm>
  <template v-if="showFooter">
    <QSeparator class="q-my-md" />
    <QSpinner v-if="isComputing" />
    <span v-else>Output of the graph:</span>
    <ol>
      <li v-for="(output,i) in graphResult" :key="output+i">{{output}}</li>
    </ol>
    <QBtn label="Reset"
          @click="reset" />
  </template>
</template>
