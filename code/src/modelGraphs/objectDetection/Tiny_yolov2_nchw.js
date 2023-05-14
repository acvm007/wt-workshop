import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, name, useBias = false) {
    const prefix = this.weightsUrl + 'convolution' + name;
    const weightName = prefix + '_W.npy';
    const weight = await buildConstantByNpy(this.builder, weightName);
    const options = {autoPad: 'same-upper'};
    if (useBias) {
      const biasName = prefix + '_B.npy';
      options.bias = await buildConstantByNpy(this.builder, biasName);
    }
    return this.builder.conv2d(input, weight, options);
  }

  async buildBatchNorm(input, name) {
    const prefix = this.weightsUrl + 'BatchNormalization';
    const scaleName = `${prefix}_scale${name}.npy`;
    const biasName = `${prefix}_B${name}.npy`;
    const meanName = `${prefix}_mean${name}.npy`;
    const varName = `${prefix}_variance${name}.npy`;
    const scale = await buildConstantByNpy(this.builder, scaleName);
    const bias = await buildConstantByNpy(this.builder, biasName);
    const mean = await buildConstantByNpy(this.builder, meanName);
    const variance = await buildConstantByNpy(this.builder, varName);

    return this.builder.batchNormalization(
      input, mean, variance, {
        scale: scale, bias: bias,
        activation: this.builder.leakyRelu({alpha: 0.10000000149011612})
      });
  }

  async buildConvolutional(input, name) {
    const conv = await this.buildConv(input, name);
    return await this.buildBatchNorm(conv, name);
  }

  async load(data) {
    const mulScale = this.builder.constant({type: 'float32',
      dimensions: [1]}, new Float32Array([0.003921568859368563]));
    const addBias = this.builder.constant({type: 'float32',
      dimensions: [3, 1, 1]}, new Float32Array([0, 0, 0]));
    const poolOptions = {
      windowDimensions: [2, 2],
      strides: [2, 2],
      autoPad: 'same-upper',
    };
    const mul = this.builder.mul(data, mulScale);
    const add = this.builder.add(mul, addBias);
    const conv0 = await this.buildConvolutional(add, '');
    const pool0 = this.builder.maxPool2d(conv0, poolOptions);
    const conv1 = await this.buildConvolutional(pool0, '1');
    const pool1 = this.builder.maxPool2d(conv1, poolOptions);
    const conv2 = await this.buildConvolutional(pool1, '2');
    const pool2 = this.builder.maxPool2d(conv2, poolOptions);
    const conv3 = await this.buildConvolutional(pool2, '3');
    const pool3 = this.builder.maxPool2d(conv3, poolOptions);
    const conv4 = await this.buildConvolutional(pool3, '4');
    const pool4 = this.builder.maxPool2d(conv4, poolOptions);
    const conv5 = await this.buildConvolutional(pool4, '5');
    const pool5 = this.builder.maxPool2d(conv5,
      {windowDimensions: [2, 2], autoPad: 'same-upper'});
    const conv6 = await this.buildConvolutional(pool5, '6');
    const conv7 = await this.buildConvolutional(conv6, '7');
    return await this.buildConv(conv7, '8', true);
  }
}
