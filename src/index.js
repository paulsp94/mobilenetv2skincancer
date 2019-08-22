import * as tf from '@tensorflow/tfjs-node';
import Jimp from 'jimp';

export default class SkinCancerPrediction {
  constructor() {
    this.model = await tf.loadLayersModel('../model/model.json');
    this.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];
  }

  loadImg(imgURI) {
    return Jimp.read(imgURI).then(img =>
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
  }

  classify(imgURI) {
    const img = await loadImg(imgURI);
    const predictions = this.model.predict(img);
    const prediction = predictions.reshape([7]).argMax();
    const result = this.classes[prediction];
    return result;
  }
}
