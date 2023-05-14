import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

const autoPad = 'same-upper';
const strides = [2, 2];
const layout = 'nhwc';

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, nameIndices, options = {}, relu = true) {
    let prefix = this.weightsUrl + 'resnet_v2_50_';
    // Items in 'nameIndices' represent the indices of block, unit, conv
    // respectively, except two kinds of specific conv names:
    // 1. contains 'shortcut', e.g.
    // resnet_v2_50_block1_unit_1_bottleneck_v2_shortcut_weights.npy
    // 2. contains 'logits', e.g. resnet_v2_50_logits_weights.npy
    if (nameIndices[0] !== '' && nameIndices[1] !== '') {
      prefix += `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_`;
    }
    if (nameIndices[2] === 'shortcut') {
      prefix += 'shortcut';
    } else if (nameIndices[2] === 'logits') {
      prefix += nameIndices[2];
    } else {
      prefix += 'conv' + nameIndices[2];
    }
    const weightsName = prefix + '_weights.npy';
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = await buildConstantByNpy(this.builder, biasName);
    options.inputLayout = layout;
    options.filterLayout = 'ohwi';
    options.bias = bias;
    if (relu) {
      options.activation = this.builder.relu();
    }
    return this.builder.conv2d(input, weights, options);
  }

  async buildFusedBatchNorm(input, nameIndices) {
    let prefix = this.weightsUrl + 'resnet_v2_50_';
    if (nameIndices[0] === 'postnorm') {
      prefix += 'postnorm';
    } else {
      prefix +=
        `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_preact`;
    }
    const mulParamName = prefix + '_FusedBatchNorm_mul_0_param.npy';
    const mulParam = await buildConstantByNpy(this.builder, mulParamName);
    const addParamName = prefix + '_FusedBatchNorm_add_param.npy';
    const addParam = await buildConstantByNpy(this.builder, addParamName);
    return this.builder.relu(
      this.builder.add(this.builder.mul(input, mulParam), addParam));
  }

  async buildBottleneckV2(
    input, nameIndices, downsample = false, shortcut = true) {
    let residual = input;

    const fusedBn = await this.buildFusedBatchNorm(input, nameIndices);
    const conv1 = await this.buildConv(
      fusedBn, nameIndices.concat(['1']), {autoPad});
    let conv2;
    if (downsample) {
      residual = await this.buildConv(
        fusedBn, nameIndices.concat(['shortcut']), {autoPad}, false);
    }
    if (!downsample && shortcut) {
      residual = this.builder.maxPool2d(
        input, {windowDimensions: [2, 2], strides, layout, autoPad});
      conv2 = await this.buildConv(
        conv1, nameIndices.concat(['2']), {strides, padding: [1, 1, 1, 1]});
    } else {
      conv2 = await this.buildConv(
        conv1, nameIndices.concat(['2']), {autoPad});
    }
    const conv3 = await this.buildConv(
      conv2, nameIndices.concat(['3']), {autoPad}, false);
    return this.builder.add(conv3, residual);
  }

  async load(data) {
    const conv1 = await this.buildConv(
      data, ['', '', '1'], {strides, padding: [3, 3, 3, 3]}, false);
    const pool = this.builder.maxPool2d(
      conv1, {windowDimensions: [3, 3], strides, layout, autoPad});
    // Block 1
    const bottleneck1 = await this.buildBottleneckV2(pool, ['1', '1'], true);
    const bottleneck2 = await this.buildBottleneckV2(
      bottleneck1, ['1', '2'], false, false);
    const bottleneck3 = await this.buildBottleneckV2(
      bottleneck2, ['1', '3']);

    // Block 2
    const bottleneck4 = await this.buildBottleneckV2(
      bottleneck3, ['2', '1'], true);
    const bottleneck5 = await this.buildBottleneckV2(
      bottleneck4, ['2', '2'], false, false);
    const bottleneck6 = await this.buildBottleneckV2(
      bottleneck5, ['2', '3'], false, false);
    const bottleneck7 = await this.buildBottleneckV2(
      bottleneck6, ['2', '4']);

    // Block 3
    const bottleneck8 = await this.buildBottleneckV2(
      bottleneck7, ['3', '1'], true);
    const loop = async (node, num) => {
      if (num > 5) {
        return node;
      } else {
        const newNode = await this.buildBottleneckV2(
          node, ['3', num.toString()], false, false);
        num++;
        return loop(newNode, num);
      }
    };
    const bottleneck9 = await loop(bottleneck8, 2);
    const bottleneck10 = await this.buildBottleneckV2(
      bottleneck9, ['3', '6']);

    // Block 4
    const bottleneck11 = await this.buildBottleneckV2(
      bottleneck10, ['4', '1'], true);
    const bottleneck12 = await this.buildBottleneckV2(
      bottleneck11, ['4', '2'], false, false);
    const bottleneck13 = await this.buildBottleneckV2(
      bottleneck12, ['4', '3'], false, false);

    const fusedBn =
      await this.buildFusedBatchNorm(bottleneck13, ['postnorm']);
    const mean = this.builder.averagePool2d(fusedBn, {layout});
    const conv2 = await this.buildConv(
      mean, ['', '', 'logits'], {autoPad}, false);
    const reshape = this.builder.reshape(conv2, [1, null]);
    return this.builder.softmax(reshape);
  }
}
