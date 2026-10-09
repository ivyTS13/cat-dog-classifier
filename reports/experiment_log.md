# Cat vs Dog Image Classification — Experiment Log

## 1. Project Overview

**Goal:** Build an image classifier that distinguishes cats from dogs using Python and PyTorch.

**Dataset:** Kaggle cat and dog images.

**Environment:** Python, PyTorch, torchvision, Google Colab GPU for ResNet18 training.

**Evaluation metrics:** Training loss, validation loss, accuracy, precision, recall, F1-score, and confusion matrix.

## 2. Dataset

| Dataset split | Number of images |
| ------------- | ---------------: |
| Training      |            8,005 |
| Validation    |            2,023 |
| External test |              275 |
| **Total**     |       **10,303** |

The external test dataset contains 95 cat images and 180 dog images. It is imbalanced toward dogs, so accuracy is considered alongside per-class precision, recall, and F1-score.

## 3. Experiment Results

| Experiment | Model      | Main configuration                                            | Best validation accuracy | External test accuracy |
| ---------- | ---------- | ------------------------------------------------------------- | -----------------------: | ---------------------: |
| 1          | Custom CNN | Initial baseline                                              |                   69.45% |                 73.45% |
| 2          | Custom CNN | Augmentation, normalization, dropout; corrected preprocessing |                   79.68% |                      — |
| 3          | Custom CNN | Longer training                                               |                   82.65% |                      — |
| 4          | Custom CNN | Adaptive average pooling (1 × 1)                              |                   70.34% |                      — |
| 5          | Custom CNN | Adaptive average pooling (4 × 4)                              |                   82.35% |                      — |
| 6          | Custom CNN | 20-epoch Colab run                                            |                   82.16% |                      — |
| 7          | ResNet18   | ImageNet-pretrained backbone frozen; train final classifier   |                   97.73% |                 97.45% |

*Note: Record the exact checkpoint and configuration used for each external test result. The custom-CNN and ResNet18 external test results are from the same 275-image dataset.*

## 4. Best Model: ResNet18 Transfer Learning

**Best validation accuracy:** 97.73%

**External test accuracy:** 97.45%

**External test loss:** 0.0917

### External test confusion matrix

| Actual class | Predicted cat | Predicted dog |
| ------------ | ------------: | ------------: |
| Cat          |            91 |             4 |
| Dog          |             3 |           177 |

### Per-class results

| Class         | Precision | Recall | F1-score |
| ------------- | --------: | -----: | -------: |
| Cats          |      0.97 |   0.96 |     0.96 |
| Dogs          |      0.98 |   0.98 |     0.98 |
| Macro average |      0.97 |   0.97 |     0.97 |

## 5. Key Findings

1. The initial custom CNN learned useful features but had a noticeable gap between training and validation performance.
2. Correcting the preprocessing mismatch between training and validation significantly improved results.
3. Excessive compression with adaptive average pooling (1 × 1) resulted in underfitting. Using 4 × 4 pooling performed much better.
4. Training longer improved the custom CNN, but its validation accuracy remained around 82%.
5. ResNet18 transfer learning achieved substantially higher validation and external test accuracy by reusing pretrained visual features.

## 6. Limitations

* The external test set contains only 275 images.
* The cat and dog classes are imbalanced in the external test set.
* The external test results have already been examined for model comparison, so this dataset is no longer a completely untouched final benchmark.
* Further evaluation on a new, independent dataset would provide stronger evidence of generalization.

## 7. Next Steps

* Implement a script to predict the class of a single image.
* Compare frozen-backbone training with fine-tuning the final ResNet18 block.
* Visualize training and validation loss and accuracy.
* Preserve experiment configurations and checkpoint names so results can be reproduced.
