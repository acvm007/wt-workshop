import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, name, options = {}) {
    const prefix = this.weightsUrl + name;
    const weightsName = prefix + '_kernel.npy';
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = await buildConstantByNpy(this.builder, biasName);
    options.inputLayout = 'nhwc';
    options.filterLayout = 'ohwi';
    options.bias = bias;
    options.activation = this.builder.relu();
    return this.builder.conv2d(input, weights, options);
  }

  async buildFire(input, name) {
    const convSqueeze = await this.buildConv(input, name + '_squeeze');
    const convE1x1 = await this.buildConv(convSqueeze, name + '_e1x1');
    const convE3x3 = await this.buildConv(
      convSqueeze, name + '_e3x3', {padding: [1, 1, 1, 1]});
    return this.builder.concat([convE1x1, convE3x3], 3);
  }

  async load(data) {
    const strides = [2, 2];
    const layout = 'nhwc';
    const conv1 = await this.buildConv(
      data, 'conv1', {strides, autoPad: 'same-upper'});
    const maxpool1 = this.builder.maxPool2d(
      conv1, {windowDimensions: [3, 3], strides, layout});
    const fire2 = await this.buildFire(maxpool1, 'fire2');
    const fire3 = await this.buildFire(fire2, 'fire3');
    const fire4 = await this.buildFire(fire3, 'fire4');
    const maxpool4 = this.builder.maxPool2d(
      fire4, {windowDimensions: [3, 3], strides, layout});
    const fire5 = await this.buildFire(maxpool4, 'fire5');
    const fire6 = await this.buildFire(fire5, 'fire6');
    const fire7 = await this.buildFire(fire6, 'fire7');
    const fire8 = await this.buildFire(fire7, 'fire8');
    const maxpool8 = this.builder.maxPool2d(
      fire8, {windowDimensions: [3, 3], strides, layout});
    const fire9 = await this.buildFire(maxpool8, 'fire9');
    const conv10 = await this.buildConv(fire9, 'conv10');
    const averagePool2d = this.builder.averagePool2d(
      conv10, {windowDimensions: [13, 13], layout});
    const reshape = this.builder.reshape(averagePool2d, [1, null]);
    return this.builder.softmax(reshape);
  }
}
