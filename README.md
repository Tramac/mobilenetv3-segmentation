# mobilenetv3-segmentation
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

An unofficial implement of [MobileNetV3](https://arxiv.org/abs/1905.02244) for semantic segmentation.

## Requisites
- PyTorch 1.1
- Python 3.x

## Usage
-----------------
### Train
- **Single GPU training**
```
python train.py --model mobilenetv3_small --dataset citys --lr 0.0001 --epochs 240
```
- **Multi-GPU training**
```
# for example, train mobilenetv3 with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model mobilenetv3_small --dataset citys --lr 0.0001 --epochs 240
```

### Evaluation
- **Single GPU training**
```
python eval.py --model mobilenetv3_small --dataset citys
```
- **Multi-GPU training**
```
# for example, evaluate mobilenetv3 with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS --model mobilenetv3_small --dataset citys
```

## Result
- **Cityscapes**

| Backbone  |  F   | Epochs |  OHEM  | mIoU  | Params(M) | Madds(G) | CPU(f) | GPU(f) |
| :-------: | :--: | :----: | :----: | :---: | :-------: | :------: | :----: | :----: |
| MV3-Small | 128  |   80   |   ✘    | 0.411 |    1.02   |   2.98   |        |        |
| MV3-Small | 128  |   80   |   ✓    | 0.476 |    1.02   |   2.98   |        |        |
| MV3-Large | 128  |   80   |   ✘    | 0.463 |    2.68   |   8.40   |        |        |
| MV3-Large | 128  |   80   |   ✓    | 0.529 |    2.68   |   8.40   |        |        |
| MV3-Large | 128  |   160  |   ✓    | 0.526 |    2.68   |   8.40   |        |        |
where: `lr=0.01, crop_size=768`

Note: Params and Madds are got using [torchscope](https://github.com/Tramac/torchscope). They are much larger than those reported in the [paper](https://arxiv.org/abs/1905.02244).

## To Do
- [ ] improve performance
- [x] train and eval
- [x] test madds

## References
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.0-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: http://dmlc.github.io/img/apache2.svg
[lic-url]: https://github.com/Tramac/mobilenetv3-segmentation/blob/master/LICENSE