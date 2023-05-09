<script setup>
import '@webmachinelearning/webnn-polyfill'
import ArrayFields from "src/components/ArrayFields.vue";
import {QBtn, QForm, QInput, QSeparator,QSpinner} from "quasar";
import {compileMLGraph} from "src/scripts/webnnFunctions.js";
import {reactive, ref} from "vue";

const isComputing = ref(false)
const showFooter = ref(false)
const tensor = reactive({
  dimensions:[1,2,2,2],
  size: 8
})
const values = ref([])
const graphResult = ref(null)

async function test(){
  isComputing.value = true
  showFooter.value = true
  graphResult.value = await compileMLGraph(tensor,values.value)
  isComputing.value = false
}
</script>

<template>
  <h3 class="q-ma-none">Computational Graph</h3>
  <QSeparator class="q-my-sm" />
  <QForm @submit="test">
    <section class="row items-center justify-between">
      <QInput v-model.number="tensor.size"
              type="number"
              class="q-mt-lg"
              outlined
              label="Größe des Tensors"
              :disable="true"
              :rules="[
              val => !!val || 'Die Größe des Tensors muss definiert sein',
              val => val > 0 || 'Die Größe des Tensors muss größer 0 sein'
            ]" />
      <ArrayFields v-if="tensor.size % 2 === 0"
                   :inputs="tensor.dimensions"
                   :input-number="tensor.size / 2"
                   :disabled="true"
                   label="Dimensionen des Tensors"
                   required-error="Alle Dimensionen sind erforderlich!" />
    </section>
    <ArrayFields v-if="tensor.dimensions.length === tensor.size / 2"
                 :inputs="values"
                 label="Eingabewerte des Graphens"
                 required-error="Alle Eingabewerte erforderlich!"
                 :input-number="tensor.dimensions.length" />
    <QBtn type="submit"
          label="Graph berechnen" />
  </QForm>
  <template v-if="showFooter">
    <QSeparator class="q-my-md" />
    <QSpinner v-if="isComputing" />
    <span v-else>Output of the graph: {{graphResult}}</span>
  </template>
</template>
