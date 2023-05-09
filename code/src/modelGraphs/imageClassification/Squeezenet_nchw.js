import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, name, options = {}) {
    const prefix = this.weightsUrl + 'squeezenet0_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = prefix + '_bias.npy';
    const bias = await buildConstantByNpy(this.builder, biasName);
    options.bias = bias;
    options.activation = this.builder.relu();
    return this.builder.conv2d(input, weights, options);
  }

  async buildFire(input, convName, conv1x1Name, conv3x3Name) {
    const conv = await this.buildConv(input, convName);
    const conv1x1 = await this.buildConv(conv, conv1x1Name);
    const conv3x3 = await this.buildConv(
      conv, conv3x3Name, {padding: [1, 1, 1, 1]});
    return this.builder.concat([conv1x1, conv3x3], 1);
  }

  async load(data) {
    const conv0 = await this.buildConv(data, 'conv0', {strides: [2, 2]});
    const pool0 = this.builder.maxPool2d(
      conv0, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire0 = await this.buildFire(pool0, 'conv1', 'conv2', 'conv3');
    const fire1 = await this.buildFire(fire0, 'conv4', 'conv5', 'conv6');
    const pool1 = this.builder.maxPool2d(
      fire1, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire2 = await this.buildFire(pool1, 'conv7', 'conv8', 'conv9');
    const fire3 = await this.buildFire(fire2, 'conv10', 'conv11', 'conv12');
    const pool2 = this.builder.maxPool2d(
      fire3, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire4 = await this.buildFire(pool2, 'conv13', 'conv14', 'conv15');
    const fire5 = await this.buildFire(fire4, 'conv16', 'conv17', 'conv18');
    const fire6 = await this.buildFire(fire5, 'conv19', 'conv20', 'conv21');
    const fire7 = await this.buildFire(fire6, 'conv22', 'conv23', 'conv24');
    const conv25 = await this.buildConv(fire7, 'conv25');
    const pool3 = this.builder.averagePool2d(
      conv25, {windowDimensions: [13, 13], strides: [13, 13]});
    const reshape0 = this.builder.reshape(pool3, [1, null]);
    return this.builder.softmax(reshape0);
  }
}
