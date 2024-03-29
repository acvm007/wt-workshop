import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, name, relu6 = true, options = {}) {
    const prefix = this.weightsUrl + 'conv_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = prefix + '_bias.npy';
    options.bias = await buildConstantByNpy(this.builder, biasName);
    if (relu6) {
      // implement `relu6` by `clamp` of  WebNN API
      options.activation = this.builder.clamp({minValue: 0, maxValue: 6});
    } else {
      options.activation = undefined;
    }
    return this.builder.conv2d(input, weights, options);
  }

  async buildGemm_(input, name) {
    const prefix = this.weightsUrl + 'gemm_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = prefix + '_bias.npy';
    const bias = await buildConstantByNpy(this.builder, biasName);
    const options = {c: bias, bTranspose: true};
    return this.builder.gemm(input, weights, options);
  }

  async buildLinearBottleneck_(
    input, convNameArray, group, stride, shortcut = true) {
    const conv1x1Relu6 = await this.buildConv(input, convNameArray[0]);
    const options = {
      padding: [1, 1, 1, 1],
      groups: group,
      strides: [stride, stride],
    };
    const dwise3x3Relu6 = await this.buildConv(
      conv1x1Relu6, convNameArray[1], true, options);
    const conv1x1Linear = await this.buildConv(
      dwise3x3Relu6, convNameArray[2], false);

    if (shortcut) {
      return this.builder.add(input, conv1x1Linear);
    }
    return conv1x1Linear;
  }

  async load(data) {
    const conv0 = await this.buildConv(
      data, '0', true, {padding: [1, 1, 1, 1], strides: [2, 2]});
    const conv1 = await this.buildConv(
      conv0, '2', true, {padding: [1, 1, 1, 1], groups: 32});
    const conv2 = await this.buildConv(conv1, '4', false);
    const bottleneck0 = await this.buildLinearBottleneck_(
      conv2, ['5', '7', '9'], 96, 2, false);
    const bottleneck1 = await this.buildLinearBottleneck_(
      bottleneck0, ['10', '12', '14'], 144, 1);
    const bottleneck2 = await this.buildLinearBottleneck_(
      bottleneck1, ['16', '18', '20'], 144, 2, false);
    const bottleneck3 = await this.buildLinearBottleneck_(
      bottleneck2, ['21', '23', '25'], 192, 1);
    const bottleneck4 = await this.buildLinearBottleneck_(
      bottleneck3, ['27', '29', '31'], 192, 1);
    const bottleneck5 = await this.buildLinearBottleneck_(
      bottleneck4, ['33', '35', '37'], 192, 2, false);
    const bottleneck6 = await this.buildLinearBottleneck_(
      bottleneck5, ['38', '40', '42'], 384, 1);
    const bottleneck7 = await this.buildLinearBottleneck_(
      bottleneck6, ['44', '46', '48'], 384, 1);
    const bottleneck8 = await this.buildLinearBottleneck_(
      bottleneck7, ['50', '52', '54'], 384, 1);
    const bottleneck9 = await this.buildLinearBottleneck_(
      bottleneck8, ['56', '58', '60'], 384, 1, false);
    const bottleneck10 = await this.buildLinearBottleneck_(
      bottleneck9, ['61', '63', '65'], 576, 1);
    const bottleneck11 = await this.buildLinearBottleneck_(
      bottleneck10, ['67', '69', '71'], 576, 1);
    const bottleneck12 = await this.buildLinearBottleneck_(
      bottleneck11, ['73', '75', '77'], 576, 2, false);
    const bottleneck13 = await this.buildLinearBottleneck_(
      bottleneck12, ['78', '80', '82'], 960, 1);
    const bottleneck14 = await this.buildLinearBottleneck_(
      bottleneck13, ['84', '86', '88'], 960, 1);
    const bottleneck15 = await this.buildLinearBottleneck_(
      bottleneck14, ['90', '92', '94'], 960, 1, false);

    const conv3 = await this.buildConv(bottleneck15, '95', true);
    const pool = this.builder.averagePool2d(conv3);
    const reshape = this.builder.reshape(pool, [1, null]);
    const gemm = await this.buildGemm_(reshape, '104');
    return this.builder.softmax(gemm);
  }
}
