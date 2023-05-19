import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(
    input, namePrefix, dwBiasSuffix = '', relu6 = true, options = {}) {
    const prefix = this.weightsUrl + namePrefix;
    let weightsName = prefix + '.npy';
    let biasName = prefix + '_bn_offset.npy';
    if (namePrefix.includes('depthwise')) {
      weightsName = prefix + '_depthwise.npy';
      biasName = `${prefix}_${dwBiasSuffix}.npy`;
    } else if (namePrefix === 'logits_semantic') {
      weightsName = prefix + '_Conv2D.npy';
      biasName = prefix + '_biases.npy';
    }
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const bias = await buildConstantByNpy(this.builder, biasName);
    options.inputLayout = 'nhwc';
    options.autoPad = 'same-upper';
    if (namePrefix.includes('depthwise')) {
      options.filterLayout = 'ihwo';
    } else {
      options.filterLayout = 'ohwi';
    }
    options.bias = bias;
    if (relu6) {
      // `relu6` in TFLite equals to `clamp` in WebNN API
      options.activation = this.builder.clamp({minValue: 0, maxValue: 6});
    } else {
      options.activation = undefined;
    }
    return this.builder.conv2d(input, weights, options);
  }

  async buildLinearBottleneck(
    input, nameIndice, dwiseOptions, shortcut = true) {
    const namePrefix = 'MobilenetV2_expanded_conv_' + nameIndice;
    let dwBiasSuffix = 'depthwise_bn_offset';
    if (Number.parseInt(nameIndice) > 6) {
      dwBiasSuffix = 'BatchNorm_FusedBatchNorm';
    }
    const conv1x1Relu6 = await this.buildConv(
      input, `${namePrefix}_expand_Conv2D`);
    const dwise3x3Relu6 = await this.buildConv(
      conv1x1Relu6, `${namePrefix}_depthwise`,
      dwBiasSuffix, true, dwiseOptions);
    const conv1x1Linear = await this.buildConv(
      dwise3x3Relu6, `${namePrefix}_project_Conv2D`, '', false);
    if (shortcut) {
      return this.builder.add(input, conv1x1Linear);
    }
    return conv1x1Linear;
  }

  async load(data) {
    const strides = [2, 2];
    const conv0 = await this.buildConv(
      data, 'MobilenetV2_Conv_Conv2D', '', true, {strides});
    const conv1 = await this.buildConv(
      conv0, 'MobilenetV2_expanded_conv_depthwise',
      'depthwise_bn_offset', true, {groups: 32});
    const conv2 = await this.buildConv(
      conv1, 'MobilenetV2_expanded_conv_project_Conv2D', '', false);
    const bottleneck0 = await this.buildLinearBottleneck(
      conv2, '1', {strides, groups: 96}, false);
    const bottleneck1 = await this.buildLinearBottleneck(
      bottleneck0, '2', {groups: 144});
    const bottleneck2 = await this.buildLinearBottleneck(
      bottleneck1, '3', {strides, groups: 144}, false);
    const bottleneck3 = await this.buildLinearBottleneck(
      bottleneck2, '4', {groups: 192});
    const bottleneck4 = await this.buildLinearBottleneck(
      bottleneck3, '5', {groups: 192});
    const bottleneck5 = await this.buildLinearBottleneck(
      bottleneck4, '6', {groups: 192}, false);
    const bottleneck6 = await this.buildLinearBottleneck(
      bottleneck5, '7', {dilations: [2, 2], groups: 384});
    const bottleneck7 = await this.buildLinearBottleneck(
      bottleneck6, '8', {dilations: [2, 2], groups: 384});
    const bottleneck8 = await this.buildLinearBottleneck(
      bottleneck7, '9', {dilations: [2, 2], groups: 384});
    const bottleneck9 = await this.buildLinearBottleneck(
      bottleneck8, '10', {dilations: [2, 2], groups: 384}, false);
    const bottleneck10 = await this.buildLinearBottleneck(
      bottleneck9, '11', {dilations: [2, 2], groups: 576});
    const bottleneck11 = await this.buildLinearBottleneck(
      bottleneck10, '12', {dilations: [2, 2], groups: 576});
    const bottleneck12 = await this.buildLinearBottleneck(
      bottleneck11, '13', {dilations: [2, 2], groups: 576}, false);
    const bottleneck13 = await this.buildLinearBottleneck(
      bottleneck12, '14', {dilations: [4, 4], groups: 960});
    const bottleneck14 = await this.buildLinearBottleneck(
      bottleneck13, '15', {dilations: [4, 4], groups: 960});
    const bottleneck15 = await this.buildLinearBottleneck(
      bottleneck14, '16', {dilations: [4, 4], groups: 960}, false);

    const conv3 = await this.buildConv(bottleneck15, 'aspp0_Conv2D');
    const averagePool2d = this.builder.averagePool2d(bottleneck15,
      {windowDimensions: [65, 65], strides: [65, 65], layout: 'nhwc'});
    const conv4 = await this.buildConv(averagePool2d, 'image_pooling_Conv2D');
    const resample0 = this.builder.resample2d(
      conv4, {sizes: [65, 65], mode: 'linear', axes: [1, 2]});
    const concat = this.builder.concat([resample0, conv3], 3);

    const conv5 = await this.buildConv(concat, 'concat_projection_Conv2D');
    const conv6 = await this.buildConv(conv5, 'logits_semantic', '', false);
    const resample1 = this.builder.resample2d(
      conv6, {sizes: [65, 65], mode: 'linear', axes: [1, 2]});
    return this.builder.resample2d(
      resample1, {sizes: [513, 513], mode: 'linear', axes: [1, 2]});
  }
}
