import {buildConstantByNpy} from "src/scripts/webnnFunctions.js";

export default class {
  constructor(weightsUrl,builder) {
    this.weightsUrl = weightsUrl;
    this.builder = builder
  }

  async buildConv(input, nameArray, relu6 = true, options = undefined) {
    // nameArray: 0: keyword, 1: indice, 2: weightSuffix, 3: biasSuffix
    let prefix = this.weightsUrl;
    let weightSuffix = '_mul_1.npy';
    let biasSuffix = `_sub__${nameArray[3]}.npy`;

    if (nameArray[0].includes('depthwise')) {
      prefix += `FeatureExtractor_MobilenetV1_MobilenetV1_Conv2d_
${nameArray[1]}_depthwise_BatchNorm_batchnorm`;
      weightSuffix = `_mul__${nameArray[2]}.npy`;
    } else if (nameArray[0].includes('pointwise')) {
      if (nameArray[0].includes('_')) {
        prefix += `FeatureExtractor_MobilenetV1_Conv2d_13_
${nameArray[0]}_Conv2d_${nameArray[1]}_BatchNorm_batchnorm`;
      } else {
        prefix += `FeatureExtractor_MobilenetV1_MobilenetV1_Conv2d_
${nameArray[1]}_pointwise_BatchNorm_batchnorm`;
      }
    } else if (nameArray[0].includes('Class')) {
      prefix += `/BoxPredictor_${nameArray[1]}_ClassPredictor`;
      weightSuffix = '_Conv2D.npy';
      biasSuffix = `_biases_read__${nameArray[3]}.npy`;
    } else if (nameArray[0].includes('BoxEncoding')) {
      prefix += `/BoxPredictor_${nameArray[1]}_BoxEncodingPredictor`;
      weightSuffix = '_Conv2D.npy';
      biasSuffix = `_biases_read__${nameArray[3]}.npy`;
    } else {
      prefix += `/FeatureExtractor_MobilenetV1_MobilenetV1_Conv2d_
${nameArray[1]}_BatchNorm_batchnorm`;
    }

    const weightsName = prefix + weightSuffix;
    const weights = await buildConstantByNpy(this.builder, weightsName);
    const biasName = prefix + biasSuffix;
    const bias = await buildConstantByNpy(this.builder, biasName);
    if (options !== undefined) {
      options.inputLayout = 'nhwc';
      options.filterLayout = 'ohwi';
      options.autoPad = 'same-upper';
    } else {
      options = {
        inputLayout: 'nhwc',
        filterLayout: 'ohwi',
        autoPad: 'same-upper',
      };
    }
    if (nameArray[0].includes('depthwise')) {
      options.filterLayout = 'ihwo';
    }
    options.bias = bias;
    if (relu6) {
      // implement `relu6` by `clamp` of  WebNN API
      options.activation = this.builder.clamp({minValue: 0, maxValue: 6});
    }
    return this.builder.conv2d(input, weights, options);
  }

  async load(data) {
    const strides = [2, 2];
    const conv0 = await this.buildConv(
      data, ['', '0', '', '165__cf__168'], true, {strides});
    const dwise0 = await this.buildConv(
      conv0, ['depthwise', '1', '161__cf__164', '162__cf__165'],
      true, {groups: 32});
    const conv1 = await this.buildConv(
      dwise0, ['pointwise', '1', '', '159__cf__162']);
    const dwise1 = await this.buildConv(
      conv1, ['depthwise', '2', '155__cf__158', '156__cf__159'],
      true, {strides, groups: 64});
    const conv2 = await this.buildConv(
      dwise1, ['pointwise', '2', '', '153__cf__156']);
    const dwise2 = await this.buildConv(
      conv2, ['depthwise', '3', '149__cf__152', '150__cf__153'],
      true, {groups: 128});
    const conv3 = await this.buildConv(
      dwise2, ['pointwise', '3', '', '147__cf__150']);
    const dwise3 = await this.buildConv(
      conv3, ['depthwise', '4', '143__cf__146', '144__cf__147'],
      true, {strides, groups: 128});
    const conv4 = await this.buildConv(
      dwise3, ['pointwise', '4', '', '141__cf__144']);
    const dwise4 = await this.buildConv(
      conv4, ['depthwise', '5', '137__cf__140', '138__cf__141'],
      true, {groups: 256});
    const conv5 = await this.buildConv(
      dwise4, ['pointwise', '5', '', '135__cf__138']);
    const dwise5 = await this.buildConv(
      conv5, ['depthwise', '6', '131__cf__134', '132__cf__135'],
      true, {strides, groups: 256});
    const conv6 = await this.buildConv(
      dwise5, ['pointwise', '6', '', '129__cf__132']);
    const dwise6 = await this.buildConv(
      conv6, ['depthwise', '7', '125__cf__128', '126__cf__129'],
      true, {groups: 512});
    const conv7 = await this.buildConv(
      dwise6, ['pointwise', '7', '', '123__cf__126']);
    const dwise7 = await this.buildConv(
      conv7, ['depthwise', '8', '119__cf__122', '120__cf__123'],
      true, {groups: 512});
    const conv8 = await this.buildConv(
      dwise7, ['pointwise', '8', '', '117__cf__120']);
    const dwise8 = await this.buildConv(
      conv8, ['depthwise', '9', '113__cf__116', '114__cf__117'],
      true, {groups: 512});
    const conv9 = await this.buildConv(
      dwise8, ['pointwise', '9', '', '111__cf__114']);
    const dwise9 = await this.buildConv(
      conv9, ['depthwise', '10', '107__cf__110', '108__cf__111'],
      true, {groups: 512});
    const conv10 = await this.buildConv(
      dwise9, ['pointwise', '10', '', '105__cf__108']);
    const dwise10 = await this.buildConv(
      conv10, ['depthwise', '11', '101__cf__104', '102__cf__105'],
      true, {groups: 512});
    const conv11 = await this.buildConv(
      dwise10, ['pointwise', '11', '', '99__cf__102']);

    const dwise11 = await this.buildConv(
      conv11, ['depthwise', '12', '95__cf__98', '96__cf__99'],
      true, {strides, groups: 512});
    const conv12 = await this.buildConv(
      dwise11, ['pointwise', '12', '', '93__cf__96']);
    const dwise12 = await this.buildConv(
      conv12, ['depthwise', '13', '89__cf__92', '90__cf__93'],
      true, {groups: 1024});
    const conv13 = await this.buildConv(
      dwise12, ['pointwise', '13', '', '87__cf__90']);

    const conv14 = await this.buildConv(
      conv13, ['pointwise_1', '2_1x1_256', '', '84__cf__87']);
    const conv15 = await this.buildConv(
      conv14, ['pointwise_2', '2_3x3_s2_512', '', '81__cf__84'],
      true, {strides});
    const conv16 = await this.buildConv(
      conv15, ['pointwise_1', '3_1x1_128', '', '78__cf__81']);
    const conv17 = await this.buildConv(
      conv16, ['pointwise_2', '3_3x3_s2_256', '', '75__cf__78'],
      true, {strides});
    const conv18 = await this.buildConv(
      conv17, ['pointwise_1', '4_1x1_128', '', '72__cf__75']);
    const conv19 = await this.buildConv(
      conv18, ['pointwise_2', '4_3x3_s2_256', '', '69__cf__72'],
      true, {strides});
    const conv20 = await this.buildConv(
      conv19, ['pointwise_1', '5_1x1_64', '', '66__cf__69']);
    const conv21 = await this.buildConv(
      conv20, ['pointwise_2', '5_3x3_s2_128', '', '63__cf__66'],
      true, {strides});

    // First concatenation
    const conv22 = await this.buildConv(
      conv11, ['BoxEncoding', '0', '', '177__cf__180'], false);
    const reshape0 = this.builder.reshape(conv22, [1, 1083, 1, 4]);
    const conv23 = await this.buildConv(
      conv13, ['BoxEncoding', '1', '', '175__cf__178'], false);
    const reshape1 = this.builder.reshape(conv23, [1, 600, 1, 4]);
    const conv24 = await this.buildConv(
      conv15, ['BoxEncoding', '2', '', '173__cf__176'], false);
    const reshape2 = this.builder.reshape(conv24, [1, 150, 1, 4]);
    const conv25 = await this.buildConv(
      conv17, ['BoxEncoding', '3', '', '171__cf__174'], false);
    const reshape3 = this.builder.reshape(conv25, [1, 54, 1, 4]);
    const conv26 = await this.buildConv(
      conv19, ['BoxEncoding', '4', '', '169__cf__172'], false);
    const reshape4 = this.builder.reshape(conv26, [1, 24, 1, 4]);
    const conv27 = await this.buildConv(
      conv21, ['BoxEncoding', '5', '', '167__cf__170'], false);
    const reshape5 = this.builder.reshape(conv27, [1, 6, 1, 4]);
    // XNNPACK doesn't support concat inputs size > 4.
    const concatReshape0123 = this.builder.concat(
      [reshape0, reshape1, reshape2, reshape3], 1);
    const concat0 = this.builder.concat(
      [concatReshape0123, reshape4, reshape5], 1);

    // Second concatenation
    const conv28 = await this.buildConv(
      conv11, ['Class', '0', '', '51__cf__54'], false);
    const reshape6 = this.builder.reshape(conv28, [1, 1083, 91]);
    const conv29 = await this.buildConv(
      conv13, ['Class', '1', '', '49__cf__52'], false);
    const reshape7 = this.builder.reshape(conv29, [1, 600, 91]);
    const conv30 = await this.buildConv(
      conv15, ['Class', '2', '', '47__cf__50'], false);
    const reshape8 = this.builder.reshape(conv30, [1, 150, 91]);
    const conv31 = await this.buildConv(
      conv17, ['Class', '3', '', '45__cf__48'], false);
    const reshape9 = this.builder.reshape(conv31, [1, 54, 91]);
    const conv32 = await this.buildConv(
      conv19, ['Class', '4', '', '43__cf__46'], false);
    const reshape10 = this.builder.reshape(conv32, [1, 24, 91]);
    const conv33 = await this.buildConv(
      conv21, ['Class', '5', '', '41__cf__44'], false);
    const reshape11 = this.builder.reshape(conv33, [1, 6, 91]);
    // XNNPACK doesn't support concat inputs size > 4.
    const concatReshape6789 = this.builder.concat(
      [reshape6, reshape7, reshape8, reshape9], 1);
    const concat1 = this.builder.concat(
      [concatReshape6789, reshape10, reshape11], 1);

    return {'boxes': concat0, 'scores': concat1};
  }
}
