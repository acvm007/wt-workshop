import {getTime} from "src/scripts/webnnFunctions.js";
import Mobilenetv1_nchw from "src/modelGraphs/objectDetection/Mobilenetv1_nchw.js";
import Mobilenetv1_nhwc from "src/modelGraphs/objectDetection/Mobilenetv1_nhwc.js";
import Mobilenetv2_nchw from "src/modelGraphs/imageClassification/Mobilenetv2_nchw.js";
import Mobilenetv2_nhwc from "src/modelGraphs/imageClassification/Mobilenetv2_nhwc.js";
import Resnet50v2_nchw from "src/modelGraphs/imageClassification/Resnet50v2_nchw.js";
import Resnet50v2_nhcw from "src/modelGraphs/imageClassification/Resnet50v2_nhcw.js";
import Squeezenet_nchw from "src/modelGraphs/imageClassification/Squeezenet_nchw.js";
import Squeezenet_nhcw from "src/modelGraphs/imageClassification/Squeezenet_nhcw.js";
import Tiny_yolov2_nchw from "src/modelGraphs/objectDetection/Tiny_yolov2_nchw.js";
import Tiny_yolov2_nhwc from "src/modelGraphs/objectDetection/Tiny_yolov2_nhwc.js";
import DeeplabV3_nchw from "src/modelGraphs/semanticSegmentation/DeeplabV3_nchw.js";
import DeeplabV3_nhwc from "src/modelGraphs/semanticSegmentation/DeeplabV3_nhwc.js";
import StyleTransferNet from "src/modelGraphs/styleTransfer/styleTransferNet.js";

export class Graph {
  constructor(context,builder,modelName) {
    this.context = context;
    this.builder = builder;
    this.modelName = modelName
  }

  //Compile the graph
  async compile(outputOperand) {
    const start = performance.now()
    let input
    if(this.modelName.startsWith('ssdMobilenetV1')) input = outputOperand
    else input = {'output': outputOperand}
    const compiled = await this.builder.build(input)
    const end = performance.now()
    return {compiled,time:getTime(start,end)}
  }

  //Execute the graph
  async execute(graph,inputBuffer, outputBuffer) {
    const start = performance.now()
    const inputs = {'input': inputBuffer};
    const outputs = {'output': outputBuffer};
    const executed = await this.context.compute(graph, inputs, outputs);
    const end = performance.now()
    return {executed,time:getTime(start,end)}
  }

  //Abstraction function to build different graphs
  async load(data,style){
    const start = performance.now()
    const url = `https://web102.in-p.de/webnn/models/${this.modelName}/weights/`
    let modelGraph
    switch (this.modelName){
      //Simple use cases
      case 'mobilenetv2_nchw': {
        //Image Classification
        modelGraph = new Mobilenetv2_nchw(url,this.builder)
        break
      }
      case 'mobilenetv2_nhwc': {
        //Image Classification
        modelGraph = new Mobilenetv2_nhwc(url,this.builder)
        break
      }
      case 'squeezenet_nchw': {
        //Image Classification
        modelGraph = new Squeezenet_nchw(url,this.builder)
        break
      }
      case 'squeezenet_nhwc': {
        //Image Classification
        modelGraph = new Squeezenet_nhcw(url,this.builder)
        break
      }
      case 'resnet50v2_nchw': {
        //Image Classification
        modelGraph = new Resnet50v2_nchw(url,this.builder)
        break
      }
      case 'resnet50v2_nhwc': {
        //Image Classification
        modelGraph = new Resnet50v2_nhcw(url,this.builder)
        break
      }
      case 'ssdMobilenetV1_nchw':{
        //Object Detection
        modelGraph = new Mobilenetv1_nchw(url,this.builder)
        break
      }
      case 'ssdMobilenetV1_nhwc':{
        //Object Detection
        modelGraph = new Mobilenetv1_nhwc(url,this.builder)
        break
      }
      case 'tinyYoloV2_nchw':{
        //Object Detection
        modelGraph = new Tiny_yolov2_nchw(url,this.builder)
        break
      }
      case 'tinyYoloV2_nhwc':{
        //Object Detection
        modelGraph = new Tiny_yolov2_nhwc(url,this.builder)
        break
      }
      case 'fastStyleTransfer_nchw':{
        //Style Transfer
        modelGraph = new StyleTransferNet(url,this.builder,style)
        break
      }
      case 'deeplabV3_nchw':{
        //Semantic Segmentation
        modelGraph = new DeeplabV3_nchw(url,this.builder)
        break
      }
      case 'deeplabV3_nhwc':{
        //Semantic Segmentation
        modelGraph = new DeeplabV3_nhwc(url,this.builder)
        break
      }
      default: throw Error(`Model "${this.modelName}" not valid!`)
    }
    const graph = await modelGraph.load(data)
    const end = performance.now()
    return {graph,time:getTime(start,end)}
  }
}
