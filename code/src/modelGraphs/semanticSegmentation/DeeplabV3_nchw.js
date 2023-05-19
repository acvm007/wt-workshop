import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, nameArray, activation = 'relu6', options = {}) {
    // nameArray: 0: bias name prefix, 1: depthWise Conv2D's bias name suffix, 2: indice of weight name
    const biasPrefix = this.weightsUrl.replace('nchw','nhwc') + nameArray[0];
    const weightsName = `${this.weightsUrl}const_fold_opt__${nameArray[2]}.npy`;
    let biasName = biasPrefix + '_bn_offset.npy';
    if (nameArray[0].includes('depthwise')) {
      biasName = `${biasPrefix}_bn_offset.npy`;
      if (nameArray[1] !== '') {
        biasName = `${biasPrefix}_${nameArray[1]}.npy`;
      }
    } else if (nameArray[0] === 'logits_semantic') {
      biasName = biasPrefix + '_biases.npy';
    }

    const weights = await buildConstantByNpy(this.builder, weightsName);
    const bias = await buildConstantByNpy(this.builder, biasName);

    options.bias = bias;
    if (activation === 'relu6') {
      // implement `relu6` by `clamp` of  WebNN API
      options.activation = this.builder.clamp({minValue: 0, maxValue: 6});
    } else if (activation === 'relu') {
      options.activation = this.builder.relu();
    } else {
      options.activation = undefined;
    }
    return this.builder.conv2d(input, weights, options);
  }

  async buildLinearBottleneck(input, nameArray, dwiseOptions, shortcut = true) {
    // nameArray: 0: indice of bias name, 1: indice of conv1x1Relu6's weight name,
    // 2: indice of dwise3x3Relu6's weight name, 3: indice of conv1x1Linear's weight name
    const biasPrefix = 'MobilenetV2_expanded_conv_' + nameArray[0];
    let dwBiasSuffix = 'depthwise_bn_offset';
    if (Number.parseInt(nameArray[0]) > 6) {
      dwBiasSuffix = 'BatchNorm_FusedBatchNorm';
    }
    const conv1x1Relu6 = await this.buildConv(
      input,
      [`${biasPrefix}_expand_Conv2D`, dwBiasSuffix, nameArray[1]]);
    const dwise3x3Relu6 = await this.buildConv(
      conv1x1Relu6,
      [`${biasPrefix}_depthwise`, dwBiasSuffix, nameArray[2]],
      'relu6',
      dwiseOptions);
    const conv1x1Linear = await this.buildConv(
      dwise3x3Relu6,
      [`${biasPrefix}_project_Conv2D`, dwBiasSuffix, nameArray[3]],
      'none');

    if (shortcut) {
      return this.builder.add(input, conv1x1Linear);
    }
    return conv1x1Linear;
  }

  async load(data) {
    const strides = [2, 2];
    const conv0 = await this.buildConv(
      data, ['MobilenetV2_Conv_Conv2D', '', '551'], 'relu6', {strides, padding: [1, 1, 1, 1]});
    const conv1 = await this.buildConv(
      conv0, ['MobilenetV2_expanded_conv_depthwise_depthwise', '', '543'], 'relu6',
      {padding: [1, 1, 1, 1], groups: 32});
    const conv2 = await this.buildConv(
      conv1, ['MobilenetV2_expanded_conv_project_Conv2D', '', '511'], 'none');
    const bottleneck0 = await this.buildLinearBottleneck(
      conv2, ['1', '537', '494', '534'], {strides, padding: [1, 1, 1, 1], groups: 96}, false);
    const bottleneck1 = await this.buildLinearBottleneck(
      bottleneck0, ['2', '447', '555', '523'], {padding: [1, 1, 1, 1], groups: 144});
    const bottleneck2 = await this.buildLinearBottleneck(
      bottleneck1, ['3', '520', '562', '542'], {strides, padding: [1, 1, 1, 1], groups: 144}, false);
    const bottleneck3 = await this.buildLinearBottleneck(
      bottleneck2, ['4', '503', '505', '489'], {padding: [1, 1, 1, 1], groups: 192});
    const bottleneck4 = await this.buildLinearBottleneck(
      bottleneck3, ['5', '446', '530', '522'], {padding: [1, 1, 1, 1], groups: 192});
    const bottleneck5 = await this.buildLinearBottleneck(
      bottleneck4, ['6', '491', '561', '538'], {padding: [1, 1, 1, 1], groups: 192}, false);
    const bottleneck6 = await this.buildLinearBottleneck(
      bottleneck5, ['7', '487', '560', '478'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]});
    const bottleneck7 = await this.buildLinearBottleneck(
      bottleneck6, ['8', '467', '536', '455'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]});
    const bottleneck8 = await this.buildLinearBottleneck(
      bottleneck7, ['9', '474', '524', '558'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]});
    const bottleneck9 = await this.buildLinearBottleneck(
      bottleneck8, ['10', '465', '556', '462'], {padding: [2, 2, 2, 2], groups: 384, dilations: [2, 2]}, false);
    const bottleneck10 = await this.buildLinearBottleneck(
      bottleneck9, ['11', '453', '532', '450'], {padding: [2, 2, 2, 2], groups: 576, dilations: [2, 2]});
    const bottleneck11 = await this.buildLinearBottleneck(
      bottleneck10, ['12', '441', '554', '517'], {padding: [2, 2, 2, 2], groups: 576, dilations: [2, 2]});
    const bottleneck12 = await this.buildLinearBottleneck(
      bottleneck11, ['13', '544', '509', '479'], {padding: [2, 2, 2, 2], groups: 576, dilations: [2, 2]}, false);
    const bottleneck13 = await this.buildLinearBottleneck(
      bottleneck12, ['14', '482', '552', '512'], {padding: [4, 4, 4, 4], groups: 960, dilations: [4, 4]});
    const bottleneck14 = await this.buildLinearBottleneck(
      bottleneck13, ['15', '475', '495', '563'], {padding: [4, 4, 4, 4], groups: 960, dilations: [4, 4]});
    const bottleneck15 = await this.buildLinearBottleneck(
      bottleneck14, ['16', '500', '459', '539'], {padding: [4, 4, 4, 4], groups: 960, dilations: [4, 4]}, false);

    const conv3 = await this.buildConv(bottleneck15, ['aspp0_Conv2D', '', '553'], 'relu');
    const averagePool2d = this.builder.averagePool2d(
      bottleneck15, {windowDimensions: [65, 65], layout: 'nchw'});
    const conv4 = await this.buildConv(averagePool2d, ['image_pooling_Conv2D', '', '546'], 'relu');
    const resample0 = this.builder.resample2d(
      conv4, {sizes: [65, 65], mode: 'linear'});
    const concat = this.builder.concat([resample0, conv3], 1);

    const conv5 = await this.buildConv(concat, ['concat_projection_Conv2D', '', '502'], 'relu');
    const conv6 = await this.buildConv(conv5, ['logits_semantic', '', '541'], 'none');
    const resample1 = this.builder.resample2d(
      conv6, {sizes: [65, 65], mode: 'linear'});
    return this.builder.resample2d(
      resample1, {sizes: [513, 513], mode: 'linear'});
  }
}
