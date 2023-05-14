<script setup>
import {QInput} from "quasar";
import {computed} from "vue";

const props = defineProps({
  inputs:{type:Array,required:true},
  inputNumber:{type:Number,required:true},
  label:{type:String,required:true},
  requiredError:{type:String,required:true},
  disabled:{type:Boolean,default:false}
})
defineEmits(['update:inputs'])

const model = computed(() => props.inputs)
</script>

<template>
  <section class="col">
    <label>{{label}}</label>
    <section class="row q-gutter-x-sm">
      <QInput v-for="i in inputNumber"
              :key="`dim-${i}`"
              v-model.number="model[i - 1]"
              outlined
              type="number"
              :disable="disabled"
              :rules="[val => val !== '' || requiredError]" />
    </section>
  </section>
</template>
