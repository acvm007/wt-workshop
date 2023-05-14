import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, name, stageName, options = undefined) {
    let prefix = '';
    if (stageName !== '') {
      prefix = this.weightsUrl + 'resnetv24_stage' + stageName + '_conv' +
        name;
    } else {
      prefix = this.weightsUrl + 'resnetv24_conv' + name;
    }
    const weightName = prefix + '_weight.npy';
    const weight = await buildConstantByNpy(this.builder, weightName);
    return this.builder.conv2d(input, weight, options);
  }

  async buildBatchNorm(input, name, stageName, relu = true) {
    let prefix = '';
    if (stageName !== '') {
      prefix = this.weightsUrl + 'resnetv24_stage' + stageName +
        '_batchnorm' + name;
    } else {
      prefix = this.weightsUrl + 'resnetv24_batchnorm' + name;
    }
    const scaleName = prefix + '_gamma.npy';
    const biasName = prefix + '_beta.npy';
    const meanName = prefix + '_running_mean.npy';
    const varName = prefix + '_running_var.npy';
    const scale = await buildConstantByNpy(this.builder, scaleName);
    const bias = await buildConstantByNpy(this.builder, biasName);
    const mean = await buildConstantByNpy(this.builder, meanName);
    const variance = await buildConstantByNpy(this.builder, varName);
    const options = {scale: scale, bias: bias};
    if (relu) {
      options.activation = this.builder.relu();
    }
    return this.builder.batchNormalization(input, mean, variance, options);
  }

  async buildGemm(input, name) {
    const prefix = this.weightsUrl + 'resnetv24_dense' + name;
    const weightName = prefix + '_weight.npy';
    const weight = await buildConstantByNpy(this.builder, weightName);
    const biasName = prefix + '_bias.npy';
    const bias = await buildConstantByNpy(this.builder, biasName);
    const options =
      {c: this.builder.reshape(bias, [1, null]), bTranspose: true};
    return this.builder.gemm(input, weight, options);
  }

  async buildBottlenectV2(
    input, stageName, nameIndices, downsample = false, stride = 1) {
    let residual = input;
    let strides = [1, 1];

    if (downsample) {
      strides = [stride, stride];
    }
    const bn1 = await this.buildBatchNorm(input, nameIndices[0], stageName);
    const conv1 = await this.buildConv(bn1, nameIndices[1], stageName);
    const bn2 = await this.buildBatchNorm(
      conv1, parseInt(nameIndices[0]) + 1, stageName);
    const conv2 = await this.buildConv(
      bn2, nameIndices[2], stageName, {padding: [1, 1, 1, 1], strides});
    const bn3 = await this.buildBatchNorm(
      conv2, parseInt(nameIndices[0]) + 2, stageName);
    const conv3 = await this.buildConv(bn3, nameIndices[3], stageName);
    if (downsample) {
      residual = await this.buildConv(
        bn1, parseInt(nameIndices[0]) + 3, stageName, {strides});
    }
    return this.builder.add(conv3, residual);
  }

  async load(data) {
    const bn1 = await this.buildBatchNorm(data, '0', '', false);
    const conv0 = await this.buildConv(
      bn1, '0', '', {padding: [3, 3, 3, 3], strides: [2, 2]});
    const bn2 = await this.buildBatchNorm(conv0, '1', '');
    const pool1 = await this.builder.maxPool2d(bn2,
      {windowDimensions: [3, 3], padding: [1, 1, 1, 1], strides: [2, 2]});

    // Stage 1
    const bottleneck1 = await this.buildBottlenectV2(
      pool1, '1', ['0', '0', '1', '2'], true);
    const bottleneck2 = await this.buildBottlenectV2(
      bottleneck1, '1', ['3', '4', '5', '6']);
    const bottleneck3 = await this.buildBottlenectV2(
      bottleneck2, '1', ['6', '7', '8', '9']);

    // Stage 2
    const bottleneck4 = await this.buildBottlenectV2(
      bottleneck3, '2', ['0', '0', '1', '2'], true, 2);
    const bottleneck5 = await this.buildBottlenectV2(
      bottleneck4, '2', ['3', '4', '5', '6']);
    const bottleneck6 = await this.buildBottlenectV2(
      bottleneck5, '2', ['6', '7', '8', '9']);
    const bottleneck7 = await this.buildBottlenectV2(
      bottleneck6, '2', ['9', '10', '11', '12']);

    // Stage 3
    const bottleneck8 = await this.buildBottlenectV2(
      bottleneck7, '3', ['0', '0', '1', '2'], true, 2);
    const bottleneck9 = await this.buildBottlenectV2(
      bottleneck8, '3', ['3', '4', '5', '6']);
    const bottleneck10 = await this.buildBottlenectV2(
      bottleneck9, '3', ['6', '7', '8', '9']);
    const bottleneck11 = await this.buildBottlenectV2(
      bottleneck10, '3', ['9', '10', '11', '12']);
    const bottleneck12 = await this.buildBottlenectV2(
      bottleneck11, '3', ['12', '13', '14', '15']);
    const bottleneck13 = await this.buildBottlenectV2(
      bottleneck12, '3', ['15', '16', '17', '18']);

    // Stage 4
    const bottleneck14 = await this.buildBottlenectV2(
      bottleneck13, '4', ['0', '0', '1', '2'], true, 2);
    const bottleneck15 = await this.buildBottlenectV2(
      bottleneck14, '4', ['3', '4', '5', '6']);
    const bottleneck16 = await this.buildBottlenectV2(
      bottleneck15, '4', ['6', '7', '8', '9']);

    const bn3 = await this.buildBatchNorm(bottleneck16, '2', '');
    const pool2 = await this.builder.averagePool2d(bn3);
    const reshape = this.builder.reshape(pool2, [1, null]);
    const gemm = await this.buildGemm(reshape, '0');
    return this.builder.softmax(gemm);
  }
}
