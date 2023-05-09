import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

const strides = [2, 2];
const autoPad = 'same-upper';

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, namePrefix, options = undefined, relu = true) {
    const weightsName = `${this.weightsUrl}/${namePrefix}_kernel.npy`;
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = `${this.weightsUrl}/${namePrefix}_Conv2D_bias.npy`;
    const bias = await buildConstantByNpy(this.builder, biasName);
    if (options !== undefined) {
      options.inputLayout = 'nhwc';
      options.filterLayout = 'ohwi';
      options.bias = bias;
    } else {
      options = {
        inputLayout: 'nhwc',
        filterLayout: 'ohwi',
        bias: bias,
      };
    }
    if (relu) {
      options.activation = this.builder.relu();
    }
    return this.builder.conv2d(input, weights, options);
  }

  async buildBlock35(input, indice) {
    const branch0 = await this.buildConv(
      input, `Block35_${indice}_Branch_0_Conv2d_1x1`, {autoPad});
    const branch1_0 = await this.buildConv(
      input, `Block35_${indice}_Branch_1_Conv2d_0a_1x1`, {autoPad});
    const branch1_1 = await this.buildConv(
      branch1_0, `Block35_${indice}_Branch_1_Conv2d_0b_3x3`, {autoPad});
    const branch2_0 = await this.buildConv(
      input, `Block35_${indice}_Branch_2_Conv2d_0a_1x1`, {autoPad});
    const branch2_1 = await this.buildConv(
      branch2_0, `Block35_${indice}_Branch_2_Conv2d_0b_3x3`, {autoPad});
    const branch2_2 = await this.buildConv(
      branch2_1, `Block35_${indice}_Branch_2_Conv2d_0c_3x3`, {autoPad});

    const concat = this.builder.concat([branch0, branch1_1, branch2_2], 3);
    const conv = await this.buildConv(
      concat, `Block35_${indice}_Conv2d_1x1`, {autoPad}, false);

    return this.builder.relu(this.builder.add(input, conv));
  }

  async buildBlock17(input, indice) {
    const branch0 = await this.buildConv(
      input, `Block17_${indice}_Branch_0_Conv2d_1x1`, {autoPad});
    const branch1_0 = await this.buildConv(
      input, `Block17_${indice}_Branch_1_Conv2d_0a_1x1`, {autoPad});
    const branch1_1 = await this.buildConv(
      branch1_0, `Block17_${indice}_Branch_1_Conv2d_0b_1x7`, {autoPad});
    const branch1_2 = await this.buildConv(
      branch1_1, `Block17_${indice}_Branch_1_Conv2d_0c_7x1`, {autoPad});

    const concat = this.builder.concat([branch0, branch1_2], 3);
    const conv = await this.buildConv(
      concat, `Block17_${indice}_Conv2d_1x1`, {autoPad}, false);

    return this.builder.relu(this.builder.add(input, conv));
  }

  async buildBlock8(input, indice, relu = true) {
    const branch0 = await this.buildConv(
      input, `Block8_${indice}_Branch_0_Conv2d_1x1`, {autoPad});
    const branch1_0 = await this.buildConv(
      input, `Block8_${indice}_Branch_1_Conv2d_0a_1x1`, {autoPad});
    const branch1_1 = await this.buildConv(
      branch1_0, `Block8_${indice}_Branch_1_Conv2d_0b_1x3`, {autoPad});
    const branch1_2 = await this.buildConv(
      branch1_1, `Block8_${indice}_Branch_1_Conv2d_0c_3x1`, {autoPad});

    const concat = this.builder.concat([branch0, branch1_2], 3);
    const conv = await this.buildConv(
      concat, `Block8_${indice}_Conv2d_1x1`, {autoPad}, false);

    let result = this.builder.add(input, conv);

    if (relu) {
      result = this.builder.relu(result);
    }
    return result;
  }

  async buildFullyConnected(input) {
    const weights = await buildConstantByNpy(this.builder,
      `${this.weightsUrl}/Bottleneck_kernel_transpose.npy`);
    const bias = await buildConstantByNpy(this.builder,
      `${this.weightsUrl}/Bottleneck_MatMul_bias.npy`);
    const options = {
      aTranspose: false,
      bTranspose: true,
      c: bias,
    };
    return this.builder.gemm(input, weights, options);
  }

  async load(data) {
    const poolOptions = {windowDimensions: [3, 3], strides, layout: 'nhwc'};

    const conv0 = await this.buildConv(data, 'Conv2d_1a_3x3', {strides});
    const conv1 = await this.buildConv(conv0, 'Conv2d_2a_3x3');
    const conv2 = await this.buildConv(conv1, 'Conv2d_2b_3x3', {autoPad});

    const pool0 = this.builder.maxPool2d(conv2, poolOptions);

    const conv3 = await this.buildConv(pool0, 'Conv2d_3b_1x1');
    const conv4 = await this.buildConv(conv3, 'Conv2d_4a_3x3');
    const conv5 = await this.buildConv(conv4, 'Conv2d_4b_3x3', {strides});

    // Block 35
    const block35_1 = await this.buildBlock35(conv5, 1);
    const block35_2 = await this.buildBlock35(block35_1, 2);
    const block35_3 = await this.buildBlock35(block35_2, 3);
    const block35_4 = await this.buildBlock35(block35_3, 4);
    const block35_5 = await this.buildBlock35(block35_4, 5);

    // Mixed 6a branches
    const mixed6a_branch0 = await this.buildConv(
      block35_5, 'Mixed_6a_Branch_0_Conv2d_1a_3x3', {strides});
    const mixed6a_pool = this.builder.maxPool2d(block35_5, poolOptions);
    const mixed6a_branch1_0 = await this.buildConv(
      block35_5, 'Mixed_6a_Branch_1_Conv2d_0a_1x1', {autoPad});
    const mixed6a_branch1_1 = await this.buildConv(
      mixed6a_branch1_0, 'Mixed_6a_Branch_1_Conv2d_0b_3x3', {autoPad});
    const mixed6a_branch1_2 = await this.buildConv(
      mixed6a_branch1_1, 'Mixed_6a_Branch_1_Conv2d_1a_3x3', {strides});
    const mixed6a = this.builder.concat(
      [mixed6a_branch0, mixed6a_branch1_2, mixed6a_pool], 3);

    // Block 17
    const block17_1 = await this.buildBlock17(mixed6a, 1);
    const block17_2 = await this.buildBlock17(block17_1, 2);
    const block17_3 = await this.buildBlock17(block17_2, 3);
    const block17_4 = await this.buildBlock17(block17_3, 4);
    const block17_5 = await this.buildBlock17(block17_4, 5);
    const block17_6 = await this.buildBlock17(block17_5, 6);
    const block17_7 = await this.buildBlock17(block17_6, 7);
    const block17_8 = await this.buildBlock17(block17_7, 8);
    const block17_9 = await this.buildBlock17(block17_8, 9);
    const block17_10 = await this.buildBlock17(block17_9, 10);

    // Mixed 7a branches
    const mixed7a_pool = this.builder.maxPool2d(block17_10, poolOptions);
    const mixed7a_branch0_0 = await this.buildConv(
      block17_10, 'Mixed_7a_Branch_0_Conv2d_0a_1x1', {autoPad});
    const mixed7a_branch0_1 = await this.buildConv(
      mixed7a_branch0_0, 'Mixed_7a_Branch_0_Conv2d_1a_3x3', {strides});
    const mixed7a_branch1_0 = await this.buildConv(
      block17_10, 'Mixed_7a_Branch_1_Conv2d_0a_1x1', {autoPad});
    const mixed7a_branch1_1 = await this.buildConv(
      mixed7a_branch1_0, 'Mixed_7a_Branch_1_Conv2d_1a_3x3', {strides});
    const mixed7a_branch2_0 = await this.buildConv(
      block17_10, 'Mixed_7a_Branch_2_Conv2d_0a_1x1', {autoPad});
    const mixed7a_branch2_1 = await this.buildConv(
      mixed7a_branch2_0, 'Mixed_7a_Branch_2_Conv2d_0b_3x3', {autoPad});
    const mixed7a_branch2_2 = await this.buildConv(
      mixed7a_branch2_1, 'Mixed_7a_Branch_2_Conv2d_1a_3x3', {strides});
    const mixed7a = this.builder.concat(
      [mixed7a_branch0_1, mixed7a_branch1_1,
        mixed7a_branch2_2, mixed7a_pool], 3);

    // Block 8
    const block8_1 = await this.buildBlock8(mixed7a, 1);
    const block8_2 = await this.buildBlock8(block8_1, 2);
    const block8_3 = await this.buildBlock8(block8_2, 3);
    const block8_4 = await this.buildBlock8(block8_3, 4);
    const block8_5 = await this.buildBlock8(block8_4, 5);
    const block8_6 = await this.buildBlock8(block8_5, 6, false);

    const mean = this.builder.reduceMean(block8_6, {axes: [1, 2]});
    // L2Normalization will be handled in post-processing
    return await this.buildFullyConnected(mean);
  }
}
