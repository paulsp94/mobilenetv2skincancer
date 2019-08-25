import { Image, createCanvas } from 'canvas';
import * as tf from '@tensorflow/tfjs';
import Jimp from 'jimp';

export default class SkinCancerPrediction {
  constructor() {
    this.model;
    this.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];
  }

  async initialize() {
    this.model = await tf.loadLayersModel(
      'https://raw.githubusercontent.com/paulsp94/mobilenetv2skincancer/master/model/model.json'
    );
  }

  static async create() {
    const o = new SkinCancerPrediction();
    await o.initialize();
    return o;
  }

  newLoadImg = async imgURI => {
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext('2d');
    try {
      var img = new Image();
      img.onload = () => ctx.drawImage(img, 0, 0);
      img.onerror = err => {
        throw err;
      };
      img.src = imgURI;
      const image = tf.browser.fromPixels(canvas).reshape([1, 224, 224, 3]);
      return image;
    } catch (err) {
      console.log(err);
    }
  };

  loadImg = async imgURI => {
    return Jimp.read(imgURI).then(() =>
      tf.tidy(img => {
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

        return tf
          .tensor3d(p, [img.bitmap.width, img.bitmap.height, 3])
          .reshape([1, img.bitmap.width, img.bitmap.height, 3]);
      })
    );
  };

  classify = async imgURI => {
    const img = await this.newLoadImg(imgURI);
    const predictions = await this.model.predict(img);
    const prediction = predictions
      .reshape([7])
      .argMax()
      .dataSync()[0];
    const result = this.classes[prediction];
    return result;
  };
}
