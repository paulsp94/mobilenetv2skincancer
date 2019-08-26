import * as tf from '@tensorflow/tfjs-node';
import Jimp from 'jimp';

export default class SkinCancerPrediction {
  constructor() {
    this.model;
    this.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];
    this.modelPath = `file:///${__dirname}/../model/model.json`;
  }

  async initialize() {
    this.model = await tf.loadLayersModel(this.modelPath);
  }

  static async create() {
    const o = new SkinCancerPrediction();
    await o.initialize();
    return o;
  }

  loadImg = async imgURI => {
    return Jimp.read(imgURI).then(img => {
      img.resize(224, 224);
      const p = [];
      img.scan(0, 0, img.bitmap.width, img.bitmap.height, function test(
        x,
        y,
        idx
      ) {
        p.push(this.bitmap.data[idx + 0]);
        p.push(this.bitmap.data[idx + 1]);
        p.push(this.bitmap.data[idx + 2]);
      });

      return tf.tensor4d(p, [1, img.bitmap.width, img.bitmap.height, 3]);
    });
  };

  classify = async imgURI => {
    const img = await this.loadImg(imgURI);
    const predictions = await this.model.predict(img);
    const prediction = predictions
      .reshape([7])
      .argMax()
      .dataSync()[0];
    const result = this.classes[prediction];
    return result;
  };
}
