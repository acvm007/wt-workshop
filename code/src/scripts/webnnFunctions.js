import * as Yolo2Decoder from "./libs/yolo2Decoder.js";
import {Graph} from "./Graph.js";
import {numpy} from "./libs/numpy.js";
import {GET} from "./requestUtils.js";
import * as tf from "@tensorflow/tfjs"
import {Renderer} from './libs/renderer.js'

async function getMLContext(options = {}) {
  return await navigator.ml.createContext({powerPreference: 'low-power', ...options});
}

export async function compileMLGraph(tensor, values) {
  const context = await getMLContext()
  const builder = new MLGraphBuilder(context);
  const desc = {type: 'float32', dimensions: tensor.dimensions};
  const intermediates = []
  const names = []
  tensor.dimensions.reduce((acc, curr, index, array) => {
    if (index % 2 === 0) {
      acc.push(array.slice(index, index + 2));
    }
    return acc;
  }, []).forEach((pair, pIndex) => {
    const inputs = tensor.dimensions.map((val, i) => {
      if (pIndex > 0) pIndex += tensor.dimensions.length
      return builder.input(`input_${i + pIndex}`, desc)
    })
    names.push(inputs[0].name, inputs[1].name)
    intermediates.push(builder.add(inputs[0], inputs[1]))
  })
  const output = builder.add(intermediates[0], intermediates[1])
  const graph = await builder.build({'output': output});
  const outputBuffer = new Float32Array(tensor.size);
  const graphInputs = names.reduce((acc, curr, i) => {
    acc = {...acc, [curr]: new Float32Array(tensor.size).fill(values[i])}
    return acc
  }, {})
  const out = await context.compute(graph, graphInputs, {'output': outputBuffer})
  return out.outputs.output
}

export function getInputTensor(inputElement,outputElement,options){
  const dimensions = [1, 3, options.height, options.width]
  const tensor = new Float32Array(
    dimensions.slice(1).reduce((a, b) => a * b));

  inputElement.width = inputElement.videoWidth ||
    inputElement.naturalWidth;
  inputElement.height = inputElement.videoHeight ||
    inputElement.naturalHeight;

  let [channels, height, width] = dimensions.slice(1);
  const mean = options.mean || [0, 0, 0, 0];
  const std = options.std || [1, 1, 1, 1];
  const normlizationFlag = options.normalize || false;
  const channelScheme = options.channelScheme || 'RGB';
  const scaledFlag = options.scaled || false;
  const inputLayout = options.layout;
  const imageChannels = 4; // RGBA
  if (inputLayout === 'nhwc') {
    [height, width, channels] = dimensions.slice(1);
  }
  const canvasElement = outputElement
  canvasElement.width = width;
  canvasElement.height = height;
  const canvasContext = canvasElement.getContext('2d');

  if (scaledFlag) {
    const resizeRatio = Math.max(Math.max(
      inputElement.width / width, inputElement.height / height), 1);
    const scaledWidth = Math.floor(inputElement.width / resizeRatio);
    const scaledHeight = Math.floor(inputElement.height / resizeRatio);
    canvasContext.drawImage(inputElement, 0, 0, scaledWidth, scaledHeight);
  } else {
    canvasContext.drawImage(inputElement, 0, 0, width, height);
  }

  let pixels = canvasContext.getImageData(0, 0, width, height).data;

  if (normlizationFlag) {
    pixels = new Float32Array(pixels).map((p) => p / 255);
  }

  for (let c = 0; c < channels; ++c) {
    for (let h = 0; h < height; ++h) {
      for (let w = 0; w < width; ++w) {
        let value;
        if (channelScheme === 'BGR') {
          value = pixels[h * width * imageChannels + w * imageChannels +
          (channels - c - 1)];
        } else {
          value = pixels[h * width * imageChannels + w * imageChannels + c];
        }
        if (inputLayout === 'nchw') {
          tensor[c * width * height + h * width + w] =
            (value - mean[c]) / std[c];
        } else {
          tensor[h * width * channels + w * channels + c] =
            (value - mean[c]) / std[c];
        }
      }
    }
  }
  return {tensor,dimensions};
}

export async function computeGraphResults(type,inputBuffer, dimensions, modelName, outputShape,style) {
  const context = await getMLContext()
  const builder = new MLGraphBuilder(context);
  const data = builder.input('input', {type: 'float32', dimensions});
  let outputBuffer
  if(type === 'objectDetection' && modelName.startsWith('ssdMobilenetV1')){
    outputBuffer =  {
      'boxes': new Float32Array(
        sizeOfShape([1, 1917, 1, 4])
      ),
      'scores': new Float32Array(
        sizeOfShape([1, 1917, 91])
      )
    }
  }
  else outputBuffer = new Float32Array(
    sizeOfShape(outputShape)
  )
  const computationalGraph = new Graph(context, builder, modelName)
  const outputOperand = await computationalGraph.load(data,style)
  const graph = await computationalGraph.compile(outputOperand)
  return (await computationalGraph.execute(graph, inputBuffer, outputBuffer)).outputs.output
}

export async function makePredictions(type,modelName, buffer, labels,outputShape,options,elements,returnNum = 3) {
  if (type === 'imageClassification') {
    const probs = Array.from(buffer);
    const indexes = probs.map((prob, index) => [prob, index]);
    const sorted = indexes.sort((a, b) => {
      return a[0] - b[0]
    });
    return sorted.reverse().slice(0, returnNum).map(([prob, index]) => {
      return {
        label: labels[index],
        value: (prob * 100).toFixed(2),
      }
    })
  }
  else if(type === 'objectDetection'){
    const [model,layout] = modelName.split('_')
    if(model === 'tinyYoloV2'){
      if (layout === 'nchw') {
        //transpose to NHWC
        buffer = tf.tidy(() => {
          const tensorOne = tf.tensor(buffer, outputShape, 'float32');
          return tf.transpose(tensorOne, [0, 2, 3, 1]).dataSync();
        });
      }
      const decodeOut = Yolo2Decoder.decodeYOLOv2({numClasses: 20}, buffer, options.anchors);
      const boxes = Yolo2Decoder.getBoxes(decodeOut, options.margin);
      if(boxes.length === 0) return false
      Yolo2Decoder.drawBoxes(elements.input,elements.output,boxes,labels)
      return boxes.map(box => {
        return {
          label: labels[box[0]],
          value: Math.round((box.pop() * 100)).toFixed(2)
        }
      })
    }
    else {
      console.log(type, model, layout, buffer);
    }
  }
  else if(type === 'semanticSegmentation'){
    const [argMaxBuffer, outShape] = tf.tidy(() => {
      const tensorA = tf.tensor(buffer, outputShape, 'float32');
      const axis = options.layout === 'nchw' ? 1 : 3;
      const tensorB = tf.argMax(tensorA, axis);
      return [tensorB.dataSync(), tensorB.shape];
    });

    const srcElement = elements.input
    const width = options.dimensions[2];
    const imWidth = srcElement.naturalWidth | srcElement.width;
    const imHeight = srcElement.naturalHeight | srcElement.height;
    const resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    console.log(resizeRatio);
    const scaledWidth = Math.floor(imWidth / resizeRatio);
    const scaledHeight = Math.floor(imHeight / resizeRatio);

    const segMap = {
      data: argMaxBuffer,
      outputShape: outShape,
      labels,
    };
    const renderer = new Renderer(elements.segCanvas);
    renderer.setup();
    await renderer.uploadNewTexture(srcElement,[scaledWidth,scaledHeight])
    const res = await renderer.drawOutputs(segMap)
    //console.log(res);
  }
  else {
    console.log(type,buffer)
  }
  return [{
    label: 'TEST',
    value: 0.00,
  }]
}

export async function buildConstantByNpy(builder, url) {
  const dataTypeMap = new Map([
    ['f2', {type: 'float16', array: Uint16Array}],
    ['f4', {type: 'float32', array: Float32Array}],
    ['f8', {type: 'float64', array: Float64Array}],
    ['i1', {type: 'int8', array: Int8Array}],
    ['i2', {type: 'int16', array: Int16Array}],
    ['i4', {type: 'int32', array: Int32Array}],
    ['i8', {type: 'int64', array: BigInt64Array}],
    ['u1', {type: 'uint8', array: Uint8Array}],
    ['u2', {type: 'uint16', array: Uint16Array}],
    ['u4', {type: 'uint32', array: Uint32Array}],
    ['u8', {type: 'uint64', array: BigUint64Array}],
  ]);
  const buffer = await GET(url, {responseType: 'arraybuffer'});
  const npArray = new numpy.Array(new Uint8Array(buffer));
  if (!dataTypeMap.has(npArray.dataType)) {
    throw new Error(`Data type ${npArray.dataType} is not supported.`);
  }
  const dimensions = npArray.shape;
  const type = dataTypeMap.get(npArray.dataType).type;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  const typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
  const dataView = new DataView(npArray.data.buffer);
  const littleEndian = npArray.byteOrder === '<';
  for (let i = 0; i < sizeOfShape(dimensions); ++i) {
    typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
      i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
  }
  return builder.constant({type, dimensions}, typedArray);
}

function sizeOfShape(shape) {
  return shape.reduce((a, b) => a * b)
}
