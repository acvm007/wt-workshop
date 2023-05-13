<script setup>
import '@webmachinelearning/webnn-polyfill'
import {QMarkupTable, QSelect, QSeparator, QSpinner, QTab, QTabPanel, QTabPanels, QTabs,QImg} from "quasar";
import {GET} from "src/scripts/requestUtils.js";
import {computeGraphResults, getInputTensor, getTime, makePredictions} from "src/scripts/webnnFunctions.js";
import {computed, onBeforeMount, reactive, ref, watch} from "vue";
import {useRoute} from "vue-router";

const route = useRoute()
const props = defineProps({
  modelData:{type:Object,required:true}
})
const modelOptions = computed(() => {
  return props.modelData.options
})
const backendOptions = ref([
  {label:'CPU (Wasm)',val:'cpu'},
  {label:'GPU (WebGL)',val:'gpu'}])
const layoutOptions = ref(
  ['NCHW','NHWC'].map(label => {
    const disable = route.name === 'styleTransfer' && label === 'NHWC'
    return {label, val:label.toLowerCase(),disable}
  })
)
const model = ref(null)
const hardware = ref(null)
const layout = ref(null)
const dataEl = ref(null)
const canvasEl = ref(null)
const segCanvas = ref(null)
const labels = ref([])
const predictions = ref([])
const selected = ref('image')
const selectedStyle = ref(null)
const showCanvas = ref(false)
const image = reactive({
  src:`/workshop/images/${props.modelData.image}`,
  width:0,
  height:0
})
const inputOptions = ref(null)
const timers = reactive({
  input:0,
  build:0,
  compile:0,
  compute:0,
  predict:0
})

watch([hardware,model,layout],async () => {
  showCanvas.value = false
  if(hardware.value && model.value && layout.value && route.name !== 'styleTransfer') {
    if(model.value.width) image.width = model.value.width
    if(model.value.height) image.height = model.value.height
    inputOptions.value = props.modelData[layout.value.val]
    labels.value = (await GET(`${props.modelData.labelsUrl}/${inputOptions.value.labelsSlug}.txt`)).split('\n').filter(l => !!l)
    predictions.value = []
    await predict()
  }
})

onBeforeMount(() => {
  if(route.name === 'styleTransfer') {
    layout.value = layoutOptions.value[0]
    model.value = modelOptions.value[0]
    inputOptions.value = props.modelData[layout.value.val]
  }
})

async function selectStyle(src){
  selectedStyle.value = src
  await predict()
}

function getSizes(){
  const img = new Image();
  img.src = image.src
  img.onload = () => {
    image.width = props.modelData.width ?? img.width
    image.height = props.modelData.height ?? img.height
  };
}
async function predict(){
  const modelName = [model.value.val,layout.value.val].join('_')
  const options = {...inputOptions.value,...image,layout:layout.value.val}
  const type = route.name
  const {dimensions,tensor,inputTime} = getInputTensor(dataEl.value,canvasEl.value, options)
  const {result,buildTime,compileTime,executionTime} = await computeGraphResults(type,tensor,dimensions,modelName,inputOptions.value.outputShape,selectedStyle.value?.split('.')[0],hardware.value.val)

  const start = performance.now()
  predictions.value = await makePredictions(type,modelName,result,labels.value,inputOptions.value.outputShape,{...props.modelData,dimensions},{
    input:dataEl.value,
    output:canvasEl.value,
    segCanvas:segCanvas.value
  })
  const end = performance.now()
  const predictTime = getTime(start,end)
  showCanvas.value = true
  timers.input = inputTime
  timers.build = buildTime
  timers.compile = compileTime
  timers.compute = executionTime
  timers.predict = predictTime
  timers.total = inputTime +buildTime + compileTime + executionTime + predictTime

}
</script>

<template>
  <h3 class="q-ma-none">{{modelData.title}}</h3>
  <QSeparator class="q-my-sm" />
  <section class="row items-center q-gutter-lg">
    <QSelect v-model="hardware"
             label="Hardware auswählen"
             class="col"
             option-value="val"
             outlined
             :options="backendOptions" />
    <QSelect v-if="$route.name !== 'styleTransfer'"
             v-model="model"
             label="Vor trainiertes Model auswählen"
             class="col"
             option-value="val"
             outlined
             :disable="!hardware?.val"
             :options="modelOptions" />
    <QSelect v-if="$route.name !== 'styleTransfer'"
             v-model="layout"
             label="Layout des Eingabetensors auswählen"
             class="col"
             option-value="val"
             outlined
             :disable="!model?.val"
             :options="layoutOptions" />
  </section>
  <QSeparator class="q-my-sm" />
  <template v-if="$route.name === 'styleTransfer'">
    <section class="styleImages">
      <QImg v-for="(name,src) in modelData.styles"
            :key="src"
            :src="`/workshop/images/styles/${src}`"
            :alt="`Style ${name}`"
            :class="[{selected:selectedStyle === src}]"
            fit="contain"
            @click="selectStyle(src)">
        <div class="absolute-bottom text-subtitle1 text-center">
          {{name}}
        </div>
      </QImg>
    </section>
    <QSeparator class="q-my-sm" />
  </template>
  <section class="display">
    <div>
      <QTabs v-model="selected">
        <QTab name="image" label="Bildanalyse" />
      </QTabs>
      <QTabPanels v-model="selected" animated>
        <QTabPanel name="image">
          <img ref="dataEl"
               v-show="!showCanvas"
               :src="image.src"
               alt="Test image"
               crossorigin="anonymous" @load="getSizes" />
          <canvas ref="segCanvas"
                  v-show="showCanvas && $route.name === 'semanticSegmentation'" />
          <canvas ref="canvasEl"
                  v-show="showCanvas && $route.name !== 'semanticSegmentation'" />
        </QTabPanel>
      </QTabPanels>
    </div>
    <div class="predictions">
      <template v-if="$route.name === 'styleTransfer'">
        <p v-if="selectedStyle">SELECTED: {{selectedStyle}}</p>
        <p v-else>Kein Style selektiert</p>
      </template>
      <template v-else-if="model && layout">
        <p v-if="predictions === false">Keine Vorhersagen möglich</p>
        <QSpinner v-if="predictions.length === 0" />
        <template v-else-if="Array.isArray(predictions)">
          <section class="row items-center justify-between">
            <div class="col">
              <span>
                Tensor fertig in:
                <span class="text-bold">{{timers.input}} ms</span>
              </span>
              <QSeparator />
              <span>
                Gebaut in:
                <span class="text-bold">{{timers.build}} ms</span>
              </span>
            </div>
            <QSeparator vertical class="q-mr-lg" />
            <div class="col">
              <span>
                Kompiliert in:
                <span class="text-bold">{{timers.compile}} ms</span>
              </span>
              <QSeparator />
              <span>
                Ausgeführt in:
                <span class="text-bold">{{timers.compute}} ms</span>
              </span>
            </div>
            <QSeparator vertical class="q-mr-lg" />
            <div class="col">
              <span>
                Vorhersage in:
                <span class="text-bold">{{timers.predict}} ms</span>
              </span>
              <QSeparator />
              <span>
                Gesamtlaufzeit:
                <span class="text-bold">{{timers.total}} ms</span>
              </span>
            </div>
          </section>
          <QSeparator class="q-my-md" />
          <QMarkupTable separator="cell">
            <thead>
              <tr>
                <th>Label</th>
                <th>
                  <template v-if="$route.name === 'semanticSegmentation'">
                    Farbe
                  </template>
                  <template v-else>
                    Wahrscheinlichkeit
                  </template>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="prediction in predictions" :key="prediction.value">
                <td>{{prediction.label}}</td>
                <td>
                  <template v-if="$route.name === 'semanticSegmentation'">
                    <div class="round" :style="{
                    height:'1.5rem',
                    width:'1.5rem',
                    borderRadius:'50%',
                    backgroundColor:`rgba(${prediction.value})`
                  }"></div>
                  </template>
                  <template v-else>
                    {{prediction.value}}%
                  </template>
                </td>
              </tr>
            </tbody>
          </QMarkupTable>
        </template>
      </template>
      <p v-else>
        Kein Model/Layout ausgewählt
      </p>
    </div>
  </section>
</template>
