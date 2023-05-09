<script setup>
import '@webmachinelearning/webnn-polyfill'
import {op} from "@tensorflow/tfjs";
import {QMarkupTable, QSelect, QSeparator, QSpinner, QTab, QTabPanel, QTabPanels, QTabs,QImg} from "quasar";
import {GET} from "src/scripts/requestUtils.js";
import {computeGraphResults, getInputTensor, makePredictions} from "src/scripts/webnnFunctions.js";
import {computed, onBeforeMount, reactive, ref, watch} from "vue";
import {useRoute} from "vue-router";

const route = useRoute()
const props = defineProps({
  modelData:{type:Object,required:true}
})
const modelOptions = computed(() => {
  return props.modelData.options
})
const layoutOptions = ref(
  ['NCHW','NHWC'].map(label => {
    const disable = route.name === 'styleTransfer' && label === 'NHWC'
    return {label, val:label.toLowerCase(),disable}
  })
)
const model = ref(null)
const layout = ref(null)
const dataEl = ref(null)
const canvasEl = ref(null)
const segCanvas = ref(null)
const labels = ref([])
const predictions = ref([])
const facingMode = ref('environment')
const videoIsPlaying = ref(false)
const selected = ref('image')
const selectedStyle = ref(null)
const showCanvas = ref(false)
const image = reactive({
  src:`/images/${props.modelData.image}`,
  width:0,
  height:0
})
const inputOptions = ref(null)

watch([model,layout],async () => {
  showCanvas.value = false
  if(model.value && layout.value && route.name !== 'styleTransfer') {
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

async function beforeSwitch(value){
  if(value === 'video'){
    try {
      dataEl.value.srcObject = await navigator.mediaDevices.getUserMedia(getConstraints())
      videoIsPlaying.value = true
    } catch (error) {
      alert(`${error.name}`)
    }
  }
}

function getConstraints(){
  return{
    audio: false,
    video: {
      width: 1920,
      height: 1080,
      facingMode:facingMode.value
    }
  }
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
  const {dimensions,tensor} = getInputTensor(dataEl.value,canvasEl.value, options)
  const result = await computeGraphResults(type,tensor,dimensions,modelName,inputOptions.value.outputShape,selectedStyle.value?.split('.')[0])
  showCanvas.value = true
  predictions.value = await makePredictions(type,modelName,result,labels.value,inputOptions.value.outputShape,{...props.modelData,dimensions},{
    input:dataEl.value,
    output:canvasEl.value,
    segCanvas:segCanvas.value
  })
}
</script>

<template>
  <h3 class="q-ma-none">{{modelData.title}}</h3>
  <QSeparator class="q-my-sm" />
  <section class="row items-center q-gutter-lg">
    <QSelect v-model="model"
             label="Vortrainiertes Model auswählen"
             class="col"
             option-value="val"
             outlined
             :disable="$route.name === 'styleTransfer'"
             :options="modelOptions" />
    <QSelect v-model="layout"
             label="Layout des Eingabetensors auswählen"
             class="col"
             option-value="val"
             outlined
             :disable="!model?.val || $route.name === 'styleTransfer'"
             :options="layoutOptions" />
  </section>
  <QSeparator class="q-my-sm" />
  <template v-if="$route.name === 'styleTransfer'">
    <section class="styleImages">
      <QImg v-for="(name,src) in modelData.styles"
            :key="src"
            :src="`/images/styles/${src}`"
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
        <QTab name="video" label="Video Analyse" disable />
      </QTabs>
      <QTabPanels v-model="selected" animated @before-transition="beforeSwitch">
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
        <QTabPanel name="video">
          <video ref="dataEl" v-show="videoIsPlaying" autoplay></video>
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
        <QMarkupTable v-else-if="Array.isArray(predictions)"
                      separator="cell">
          <thead>
            <tr>
              <th>Label</th>
              <th>Wahrscheinlichkeit</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="prediction in predictions" :key="prediction.value">
              <td>{{prediction.label}}</td>
              <td>{{prediction.value}}%</td>
            </tr>
          </tbody>
        </QMarkupTable>
      </template>
      <p v-else>
        Kein Model/Layout ausgewählt
      </p>
    </div>
  </section>
</template>
