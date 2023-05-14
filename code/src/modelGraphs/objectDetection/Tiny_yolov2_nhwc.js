import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, name, leakyRelu = true) {
    const prefix = this.weightsUrl + 'conv2d_' + name;
    const weightsName = prefix + '_kernel.npy';
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = await buildConstantByNpy(this.builder, biasName);
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
      autoPad: 'same-upper',
    };
    options.bias = bias;
    let conv = this.builder.conv2d(input, weights, options);
    if (leakyRelu) {
      // Fused leakyRelu is not supported by XNNPACK.
      conv = this.builder.leakyRelu(conv, {alpha: 0.10000000149011612});
    }
    return conv;
  }

  async load(data) {
    const poolOptions = {
      windowDimensions: [2, 2],
      strides: [2, 2],
      autoPad: 'same-upper',
      layout: 'nhwc',
    };
    const conv1 = await this.buildConv(data, '1');
    const pool1 = this.builder.maxPool2d(conv1, poolOptions);
    const conv2 = await this.buildConv(pool1, '2');
    const pool2 = this.builder.maxPool2d(conv2, poolOptions);
    const conv3 = await this.buildConv(pool2, '3');
    const pool3 = this.builder.maxPool2d(conv3, poolOptions);
    const conv4 = await this.buildConv(pool3, '4');
    const pool4 = this.builder.maxPool2d(conv4, poolOptions);
    const conv5 = await this.buildConv(pool4, '5');
    const pool5 = this.builder.maxPool2d(conv5, poolOptions);
    const conv6 = await this.buildConv(pool5, '6');
    const pool6 = this.builder.maxPool2d(conv6,
      {windowDimensions: [2, 2], autoPad: 'same-upper', layout: 'nhwc'});
    const conv7 = await this.buildConv(pool6, '7');
    const conv8 = await this.buildConv(conv7, '8');
    return await this.buildConv(conv8, '9', false);
  }
}
