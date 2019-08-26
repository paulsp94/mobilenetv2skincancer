
[![GSOC](https://img.shields.io/badge/GSOC-2019-yellow)](https://summerofcode.withgoogle.com/organizations/6137730124218368/?sp-page=2#4558376158101504)
![GitHub](https://img.shields.io/github/license/paulsp94/mobilenetv2skincancer)
![npm (tag)](https://img.shields.io/npm/v/mobilenetv2skincancer/latest)

The MobileNetV2 model pretrained on imagenet and fine-tuned on the skin cancer dataset.   
The default input size for this model is 224x224.   

Install `npm install mobilenetv2skincancer`

## How to use

```javascript
import SkinCancerPrediction from 'mobilenetv2skincancer';

const nvSample = './assets/nv_sample.jpg';

const run = async () => {
    const predictor = await ResNetPredictor.create();
    const prediction = await predictor.classify(nvSample);
    return prediction;
}
```odel(ResNetURL);
```