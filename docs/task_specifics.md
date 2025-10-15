# Task-specific information

This document contains information specific to each task when using the FHIBE Fairness Benchmark API.

## Model instructions

What your wrapped model's `__call__()` method must return is task-specific. Some additional methods are also required, depending on the task. Below we explain in detail what the expected outputs of these methods are for each task. The input is always a batch from the data loader returned by the `data_preprocessor()` method. See the demo scripts and notebooks for example implementations for each task.

### Person localization

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: `List[Dict[str, List[List[float]] | List[float] | List[int]]]`
- Return explanation: Returns a list of dictionaries, where each dictionary is the output from a single image in the batch, each with this format:

```python
{
    "bboxes": [[x0_i,y0_i,x1_i,y1_i]],
    "scores": [confidence_i],
    "labels": [label_i]
}
```

where:

- `[]` represents a Python list.
- `x0_i,y0_i,x1_i,y1_i` are the left, bottom, right, and top of the ith predicted bounding box, respectively, represented as Python floats. Note that there could be multiple bounding boxes predicted for a single image.
- `confidence_i` is the model's confidence score for the ith bounding box, represented as a Python float.
- `label_i` is the class label for the ith bounding box. This is relevant if the model detects multiple classes of objects, not just people. Map or set the class label for people in your model to be 0, or just hardcode this to 0 if your model does not produce class labels. All values must be Python integers.

### Person parsing

This task parses an entire person, not their individual body parts.

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: `List[Dict[str, List[np.ndarray[bool]] | List[float] | List[int]]]`
- Return explanation: Returns a list of dictionaries, where each dictionary is the output from a single image in the batch, each with this format:

```python
{
    "masks": [bool_array_i],
    "scores": [confidence_i],
    "labels": [label_i]
}
```

where:

- `[]` represents a Python list.
- `bool_array_i` is a boolean numpy array the same shape as the input image, where True indicates an object detection, False otherwise. There can be multiple masks per image.
- `confidence_i` is the model's confidence score for the ith mask, represented as a Python float.
- `label_i` is the class label for the ith mask, represented as a Python int. This is relevant if the model segments multiple classes of objects, not just people. Set the class label for people in your model to be 0, or just use 0 if your model does not produce class labels. Likewise if your model segments individual body parts, map all person body parts to label 0 in your `__call__()` method.

### Face localization

#### Method: `__call__()`

- Inputs: A batch from the data loader returnedby the `data_preprocessor()` method.
- Return type: `List[Dict[str, List[List[float]], List[float]]]`
- Return explanation: Returns a list of dictionaries, where each dictionary is the output from a single image in the batch, each with this format:

```python
{
    "detections": [[x0_i,y0_i,x1_i,y1_i]],
    "scores": [confidence_i],
}
```

where:

- `[]` represents a Python list.
- `"detections"` maps to a list of lists, where `x0_i,y0_i,x1_i,y1_i` are to the left, bottom, right, and top of the ith predicted bounding box, respectively, represented as Python floats.
- `confidence_i` is the model's confidence score for the ith bounding box, represented as a Python float.

### Body parts detection

This task evaluates a model's ability to detect the existence of various body parts (and some accessories) of each subject in each image. This is in contrast to the keypoint estimation task (see below) which evaluates the predicted (x,y) locations of body parts. For this task, the model predicts the probabilities that one or more of the following body parts are present in each image:

- Face
- Hand
- Upper body skin
- Left arm skin
- Right arm skin
- Left leg skin
- Right leg skin
- Head hair
- Left eyebrow
- Right eyebrow
- Left eye
- Right eye
- Nose
- Upper lip
- Lower lip
- Inner mouth
- Left shoe
- Right shoe
- Headwear
- Mask
- Eyewear
- Upper body clothes
- Lower body clothes
- Full body clothes
- Sock or legwarmer
- Neckwear
- Bag
- Glove
- Jewelry or timepiece

With the exception of "Face" and "Hand", these categories are taken from FHIBE's segmentation mask categories. At least part of the face is shown for each subject in each FHIBE image, so the "Face" category is always considered to be present in the list of visible ground truth body parts. The "Hand" is not an explicit mask category, so instead we construct it based on the following condition:

If any of the following keypoints are present:

- "Left pinky knuckle"
- "Left index knuckle"
- "Left thumb knuckle"
- "Right pinky knuckle"
- "Right index knuckle"
- "Right thumb knuckle

or if the "Glove" mask is present, then we consider a hand to be present in the ground truth. Note that with this construction, the "Hand" category refers to either hand (inclusive or).

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: `List[List[Dict[str, float]]]`
- Return explanation: Returns a list of lists of dictionaries, where each outer list element runs over the images. For each image, one must return a list of dictionaries, where each dictionary represents a single image subject (person). The dictionary must map the body part strings (refer to list above) to the predicted probability that that body part exists for a single subject in a single image.

The following shows an example return format for `__call__()` for a model that predicts the "Face" and "Hand" body parts. In this example, the batch contains two images. The first image contains two subjects, so its list contains two dictionaries. The second image in the batch contains a single subject, so its list only contains a single dictionary.

```python
[
  [ # first image in batch
    { # first subject in first image
      "Face": 0.85,
      "Hand": 0.22,
    },
    { # second subject in first image
      "Face": 0.99,
      "Hand": 0.78,
    }
  ],
  [ # second image in batch
    { # first subject in second image
      "Face": 0.10,
      "Hand": 0.47,
    },
    { # second subject in second image
      "Face": 0.98,
      "Hand": 0.99,
    }
  ]
]
```

where:

- `[]` represents a Python list.
- `{}` represents a Python dictionary.
- The detection probabilities are represented by Python floats.

### Keypoint estimation

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: `List[Dict[str, List[List[List[float]]]]]`
- Return explanation: Returns a list of dictionaries, where each dictionary is the output from a single image in the batch, each with this format:

```python
{
    "keypoints": [[[x_k,y_k]_j]_i],
    "scores": [[[confidence_k]_j]_i],
}
```

where:

- `[]` represents a Python list.
- keypoints are returned in a nested list structure. For each ground truth bounding box (`i`) in an image, there is a single set of predicted keypoints (`j`), represented by a list of lists `[[x_k,y_k]_j]`, where `x_k,y_k` denote the x and y coordiantes of a single keypoint. `k` runs over the keypoints in a set of keypoints detected within a single ground truth bounding box.
- **!! Important !!**: By default, the full set of keypoints available in this API will be used when evaluating your model. The API will assume that your model outputs all keypoints. The full set of keypoints [follows the COCO dataset format](https://cocodataset.org/#keypoints-eval), and is:

```
"Nose"
"Left eye"
"Right eye"
"Left ear"
"Right ear"
"Left shoulder"
"Right shoulder"
"Left elbow"
"Right elbow"
"Left wrist"
"Right wrist"
"Left hip"
"Right hip"
"Left knee"
"Right knee"
"Left ankle"
"Right ankle"
```

You can override the default by specifying a subset of these keypoints to use in the evaluation. To do this, specify the subset as a list via the parameter `custom_keypoints` in the `evaluate_task()` function. The order of the keypoints you specify in this list, e.g., `custom_keypoints=["Nose","Left eye"]` must follow the ordering of the full keypoint set above (this is enforced in the code). The order of the keypoints that your model returns in the output dictionary (see above) MUST FOLLOW the same order as specified in your `custom_keypoints` parameter.

- Confidence scores are similarly returned in a list of lists of lists. However, instead of the keypoint coordinates in the inner list, the confidence score for a single keypoint, `confidence_k` is reported.

### Face parsing

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: `List[Dict[str, np.ndarray[np.uint8]]]`
- Return explanation: Returns a list of dictionaries, where each dictionary is the output from a single image in the batch, each with this format:

```python
{
    "detections": np.ndarray[np.uint8],
}
```

where `detections` is a numpy array with uint8 datatype of the same size as the input face image. The value of each pixel in the array represents the predicted parsed object index. The indices run from 0-18 and correspond to the CelebAMask-HQ labels. For the ordering of the labels, see `CELEBA_MASK_LABELS` in [fhibe_eval_api/metrics/face_parsing/utils.py](../fhibe_eval_api/metrics/face_parsing/utils.py).

In addition, in `__init__()`, one must set `self.map_ears_to_skin` to `True` or `False` as done in the face parsing demo. The model used in the demo produces left and right ear masks, which are contained in the CelebAMask-HQ label set, but **the FHIBE annotations do not include ground truth ear mask annotations** (they are considered to be part of face skin). As a result, we provide functionality to map ear masks to face skin, which is triggered when `self.map_ears_to_skin=True`.

### Face verification

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: `List[Dict[str[List[float]]]]`
- Return explanation: Returns a list of dictionaries, where each dictionary is the output from a single image in the batch, each with this format:

```python
{
    "embeddings": [e_i],
}
```

where `embeddings` is a list representing the embedding for the image.

### Face encoding

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: `List[Dict[str,Any]]`
- Return explanation: Returns a list of dictionaries, where each dictionary is the output from a single image in the batch, each with this format:

```python
{
    "encodings": Any, but typically an ndarray,
}
```

where `encodings` represents the encoding for the image. This can be in any format.

#### Method: `save_encoding()`

- Return type: `None`.
- Return explanation: While this method does not return anything, **it must save the image as a png** that can be loaded via a torch dataloader. This method takes as input a single image encoding (of the format returned by the `__call__()` method) and a filename and saves the encoding to disk.

### Face super resolution

#### Method: `__call__()`

- Inputs: A batch from the data loader returned by the `data_preprocessor()` method.
- Return type: List[np.ndarray[np.uint8]]
- Return explanation: Returns a list of arrays, where each array represents the super resolution image corresponding to a single image in the batch. The array can be in any format one desires because for this task a `save_array()` method is also required for this task, as demonstrated in the face super resolution demo.

#### Method: `save_array()`

- Inputs: Super resolution array from a single image (of the format returned by the `__call__()` method) and a filename for where to save the array to disk.
- Return type: `None`
- Return explanation: - Return explanation: While this method does not return anything, **it must save the image as a png** that can be loaded via a torch dataloader. The dimensions of the saved image must match those of the input FHIBE-face crop+aligned face images, i.e., 512x512.

### Available metrics

Below, we describe the available metrics for each task. The metric string used in the API is provided, followed by a description of the metric and whether a list of thresholds is required. All metric strings are fully capitalized when used in the API.

- Person localization

  - `"AR_IOU"`: Average recall over intersection over union (IoU). For each ground truth person bounding box, the best IoU out of all predicted bounding boxes is obtained. At each value in a list of IoU thresholds between 0 and 1, each image is given a value of 1 (correct) or 0 (incorrect) based on whether IoU > threshold. Using these binary outcomes, the recall is calculated. The average recall over all thresholds is reported.
    - requires thresholds: True

- Person parsing

  - `"AR_MASK"`: Average recall over IoU of the person mask. For each ground truth person mask, the best IoU out of all predicted masks is obtained. Then, at each value in a list of IoU thresholds between 0 and 1, the recall is calculated. The average recall over all thresholds is reported.
    - requires thresholds: True

- Body parts detection

  - `"AR_DET"`: Average recall of the detection of body parts. At each value in a list of thresholds between 0 and 1, the recall is calculated for each body part predicted by the model, where a positive prediction (existence of body part) is determined by prob(body_part) >= threshold. The average recall over all thresholds is reported for each body part and averaged over all body parts.

  - `"ACC_DET"`: Accuracy of the detection of body parts. At each value in a list of thresholds between 0 and 1, the accuracy is calculated for each body part predicted by the model, where a positive prediction (existence of body part) is determined by prob(body_part) >= threshold. The accuracy reported is the average over all thresholds and reported for each body part as well as the average over all body parts.

- Keypoint estimation

  - `"PCK"`: Percentage correct keypoints. The distance between each ground truth keypoint and the closest predicted keypoint is compared to the product `thresh*face_bbox_diag`, where `thresh` is a threshold value and `face_bbox_diag` is the length of the diagonal of the ground truth face bounding box. If the distance is less than the product, a keypoint is considered correct. The fraction of correct keypoints in the set of ground truth keypoints in an image is the PCK at a single threshold for a single image. This is repeated for each threshold in a list of thresholds, and the mean over all thresholds is reported.
    - requires thresholds: True
  - `"AR_OKS"`: Average recall, using object keypoint similarity. The object keypoint similarity is calculated as in the COCO evaluation dataset: https://cocodataset.org/#keypoints-eval. At each value in a list of OKS thresholds between 0 and 1, each image is given a value of 1 (correct) or 0 (incorrect) based on whether OKS > threshold. Using these binary outcomes, the recall is calculated. The average recall over all thresholds is reported.
    - requires thresholds: True

- Face localization

  - `"AR_IOU"`: Average recall over IoU. For each ground truth face bounding box, the best IoU out of all predicted face bounding boxes is obtained. At each value in a list of IoU thresholds between 0 and 1, each image is given a value of 1 (correct) or 0 (incorrect) based on whether IoU > threshold. Using these binary outcomes, the recall is calculated. The average recall over all thresholds is reported.
    - requires thresholds: True

- Face parsing

  - `"F1"`: F1 score, averaged over each face part mask. For each face part, true positives are calculated as the number of pixels in the intersection of the ground truth and predicted masks, false positives are the intersection of the predicted mask and the logical not of the ground truth mask, false negatives are the intersection of the ground truth mask and the logical not of the predicted mask. The F1 score for each face part is calculated as the harmonic mean of precision and recall. The average F1 score over all face parts is reported.
    - requires thresholds: False

- Face verification

  - `"VAL"`: Validation rate. Calculates the validation rate at a false acceptance rate of 0.001 using k-fold cross-validation.

- Face encoding
  - `"LPIPS"`: Learned perceptual image patch similarity. This is calculated for each individual image using the Pytorch Image Quality (PIQ) library: https://piq.readthedocs.io/en/latest/modules.html#learned-perceptual-image-patch-similarity-lpips.
    - requires thresholds: False
  - `"PSNR"`: Peak signal-to-noise ratio. This is calculated for each individual image using the Pytorch Image Quality (PIQ) library: https://piq.readthedocs.io/en/latest/functions.html#peak-signal-to-noise-ratio-psnr.
  - `"CURRICULAR_FACE"`: CurricularFace embedding similarity score. This calculates the dot product between the embedding of an encoded image and the embedding its corresponding original image, using the CurricularFace backbone model to embed both images. **Note**: If you wish to use this metric, you must first download the model weights from this link: https://drive.google.com/file/d/1upOyrPzZ5OI3p6WkA5D5JFYCeiZuaPcp/view and then put the file in this directory of the repository: `fhibe_eval_api/metrics/face_verification/curricular_face/`. Otherwise, you will get an error stating that the weights file cannot be found.
- Face super resolution
  - `"LPIPS"`: Learned perceptual image patch similarity. This is calculated between each (input,super_resolution) image pair using the Pytorch Image Quality (PIQ) library: https://piq.readthedocs.io/en/latest/modules.html#learned-perceptual-image-patch-similarity-lpips.

### Model output file format

After running inference, the API saves model outputs to disk in formats that are specific to each task. Below are descriptions and examples of the outputs. If you are running your own custom model inference but you wish to use the API to generate a bias evaluation, ensure that you format your outputs on disk using the format specific to your task.

#### Person localization

This task saves a file called `model_outputs.json` to the directory: `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` (or `{results_rootdir}/mini/{task_name}/...` if `use_mini_dataset=True`), where each variable is a parameter provided to `evaluate_task`. This file contains the bounding box coordinates and confidence scores for **people only** in the following format:

```json
{
    "image1_filename": {
        "detections": [
            [
                x0_0,
                x1_0,
                y0_0,
                y1_0
            ],
            [
                x0_1,
                x1_1,
                y0_1,
                y1_1
            ],
            ...
        ],
        "scores": [
            score_0,
            score_1,
            ...
        ]
    },
    "image2_filename": {
        "detections": [
            [
                x0_0,
                x1_0,
                y0_0,
                y1_0
            ],
            [
                x0_1,
                x1_1,
                y0_1,
                y1_1
            ],
            ...
        ],
        "scores": [
            score_0,
            score_1,
            ...
        ]
    },
    ...
}
```

where `x0_i,y0_i,x1_i,y1_i` are the left, bottom, right, and top of the ith predicted bounding box, respectively, for a given image, represented as Python floats. `score_i` is the confidence score for the ith predicted bounding box.

For example:

```json
{
  "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images/c3cafc9a-42f3-465c-8867-280f066d5e81/f6a939bb-1d50-49d5-8385-e2e25053e0b6.png": {
    "detections": [
      [
        436.59100341796875, 344.2961730957031, 1041.4149169921875,
        1964.57177734375
      ]
    ],
    "scores": [0.9992664456367493]
  },
  "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images/0e2edcdc-3a64-4c2b-ac93-de03d710d89a/335a64ea-c038-4ca2-a8f5-a016ed8ea386.png": {
    "detections": [
      [690.509033203125, 329.8193054199219, 1478.013671875, 1878.9967041015625],
      [
        166.66336059570312, 327.9772644042969, 801.6248168945312,
        1881.0621337890625
      ]
    ],
    "scores": [0.9996370077133179, 0.9992431402206421]
  }
}
```

#### Person parsing

This task saves a file called `model_outputs.json` to the directory: `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` (or `{results_rootdir}/mini/{task_name}/...` if `use_mini_dataset=True`), where each variable is a parameter provided to `evaluate_task`. This file contains `detections` and `scores` keys for each unique image. The format is as follows:

```json
{
    "image1_filename": {
        "detections": [
            {
                "size": [
                    {image_width},
                    {image_height}
                ],
                "counts": "string_0"
            },
            {
                "size": [
                    {image_width},
                    {image_height}
                ],
                "counts": "string_1"
            }
        ],
        "scores": [
            score_0,
            score_1
        ]
    },
    ...
}
```

where `string_i` is the utf-8 decoded string from the run-length-encoded binary image masks segmenting the ith person from the background. See the [person parsing evaluation section of the code](https://github.com/SonyResearch/fairness-benchmark-internal/blob/main/fhibe_eval_api/evaluate/evaluate.py#L274) for code to obtain this string from a binary mask. `score_i` is the confidence score for the ith predicted binary mask.

For example:

```json
{
  "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images/fd014c3d-a037-4be2-b5f9-013261fbb867/077853b2-c618-4d40-90ad-76f558cb8aea.png": {
    "detections": [
      {
        "size": [2048, 1365],
        "counts": "ZoZo09Wo1P1nNS1UOe0^O;E<C=D<C=^Ob0ZOf0XOh0E:E;J4N1N2O1N2N2N2N2M3M3M3L4L4J6H8N2M3N2N2N2M3N2M3M3M3M3L4M3K5L4K5J7H7N2O00001O00dGX[N`5hd1[Ja[Nb5^d1YJl[Nb5Td1YJU\\Nc5kc1WJ_\\Ne5ac1VJh\\Nf5Xc1UJQ]Nh5nb1RJ[]Nk5eb1PJd]Nl5\\b1aI[^N[6ea1]If^N`6Za1\\In^N`6Ra1]IR_Nc6m`1YIW_Ng6j`1UIY_Nl6f`1QI]_No6kd10O001O010O001O0010O01O00010O10000O100`hNfJYd0Z5W[OgKWd0Y4Y[OiLVd0W3R[OSN\\d0m1jZOB`d0?Z[O[1Rc0eNh\\Oi1ob0XNl\\OT2mb0lMn\\O_2kb0aMQ]Oj2ib0VMR]OT3hb0mLS]O]3gb0dLV]O`3gb0bLU]Oa3kb0`LQ]Od3mb0^Ln\\Og3Qc0[Lj\\Oi3Tc0YLf\\Ol3Zc0ULa\\OP4]c0RL]\\OT4bc0nKW\\OW4hc0kKR\\O[4mc0fKm[O`4Qd0cKh[Ob4Xd0_Kb[Og4\\d0\\K][Oi4cd0YKW[Ol4gd0VKU[Om4kd0UKP[Oo4nd0UKkZOo4Te0UKeZOo4Ze0TK`ZOP5_e0TKYZOQ5fe0SKSZOQ5le0SKlYOR5Rf0SKeYOS5Yf0RK]YOU5bf0PKPYOZ5nf0kJfXO^5Xg0hJ[XOa5dg0cJPXOf5ng0`JdWOj5[h0_JPWOn5nh0`J[VOj5ei0eJbUOf5\\j0oJUSOm6il0RJaQOg6]n0_=M2O2N2N1O2N2lLWoNoVOjP1Qi0UPOnUOno0Qj0dPO\\UO^o0bj0UQOkTOln0Uk0P3O0O2O1O0O1O1O1O001O00001O1N101O1N2O1N2O11O01O0001O000010O0001O0010O01O001O11N10001O000O2O0000001N1000001O000O101O000000001O000000001O00000O101O00001O0000001O0000001O00001O0001O01O00001O00001O00010O00001O000010O001O010O001O010O001O00100O001O100O2N1O2N2N2N2N1O2O0O2N1O2N1O1O1O2N100O2N1O2N2N2O0O2N2N2N2N3N2M3M3M3M3M4L4L4M4K5K5K6J8H6J6J5K5L4K4L3M7I7I7I8H8H9G9F;F8H9G8H5K5K4L4L3M4L4L4K6bYO^mNT?hR1^@emN]?bR1S@mmNg?YR1j_OVnNo?RR1a_O]nNY`0jQ1X_OenNa`0bQ1S_OknNe`0[Q1Q_OPoNg`0XQ1m^ORoNn`0SQ1j^OToNRa0QQ1e^OWoNWa0mP1b^OZoNZa0jP1_^O\\oN]a0iP1\\^O]oNaa0gP1X^O^oNfa0eP1S^OboNia0dP1o]OboNna0bP1e]OkoNWb0SW1K5L:E<E;E<CX1iN`1_Nk0VOf0ZOn0QOd0]O9G8G9H9G<Da0^Oe0\\O8H7I7H7J7I6J6I3N3M2N2M3N1O2N1N3N1O2N1O1N3N1O2N1N3N1O2N1O2M3N2N3M2N2M3N2N2N1O2M2O1O1O2N1N101N1O2N101N1O2M2O2N1O2M2O2M2N3M2N3M4K5L4K5K5K6J6J8G:G9F:F9F:B>]Ob0^OPXli0"
      },
      {
        "size": [2048, 1365],
        "counts": "]hik0l0on1k0TO<E:E;E:E<B>Aa0^O?@?C=D;E;E;E9G:F9F;E:H9G8J7H7H8F:F:F;E:F:H8J6J6H8G:B=E;h_OoBW]O[=bb0YDg[OQ<Sd0dDR[Oh;id0jD`ZO`;\\e0PEmYOZ;oe0ZEjXO^;Qg0YEdUO`=Vj0TCcTOe=Wk0kBUTO_=gk0oBgSOY=Vl0SCYSOT=el0WCmROo<Pm0ZCdROl<[m0[CYROj<fm0^CmQOi<Pn0aCcQOd<\\n0eCWQO`<hn0kChPO\\<Uo0gDeoN^;YP1U:O1O01O00O1I6A7I76K5_Na1G9H7J5K6K8I7J6F;J7J5bZOclNV>cS1eAglNQ>]S1kAmlNk=XS1oASmNg=QS1UBYmNb=kR1XB`mN^=eR1PBRnNg=RR1oA[nNk=iQ1nAanNl=aQ1PBgnNi=\\Q1UBjnNe=XQ1[BlnN^=XQ1aBjnNZ=YQ1gBhnNS=\\Q1mBgnNl<]Q1WCbnNf<^Q1cC\\nNZ<eQ1mCVnNP<jQ1WDRnNf;nQ1`DnmN];SR1hDkmNU;UR1PEimNl:XR1YEfmNb:\\R1bEcmNZ:_R1hEamN_:VR1cEjmNe:mQ1\\ERnNm:dQ1UE\\nNS;ZQ1oDfnNX;RQ1iDnnN\\;kP1fDUoN`;cP1bD]oNc;]P1]DdoNo;no0SDRPOj>Pm0VAQSOU?bl0l@_SO`?Tl0a@mSOj?ek0W@\\TOk?ak0U@`TOn?\\k0R@eTOP`0Wk0Q@kTOP`0Rk0P@oTOR`0nj0n_OTUOT`0hj0l_OZUOV`0bj0j_O`UOX`0\\j0h_OeUO[`0Wj0e_OlUO]`0ni0d_OTVO_`0gi0a_OZVOb`0bi0^_O`VOc`0]i0]_OeVOc`0Zi0\\_OiVOc`0Ui0]_OmVOc`0Qi0^_OPWOb`0nh0^_OTWOb`0jh0^_OXWOb`0fh0^_O\\WOb`0ch0]_O_WOc`0_h0\\_OcWOd`0\\h0\\_OfWOc`0Yh0\\_OiWOc`0Wh0\\_OkWOd`0Th0[_OnWOd`0Rh0\\_OPXOc`0og0\\_OSXOd`0lg0[_OWXOc`0ig0\\_OYXOc`0gg0]_O\\XO]`0fg0c_O^XOW`0eg0h_O`XOQ`0bg0P@`XOi?eg0V@^XOd?eg0]@^XOZ?hg0f@YXOR?mg0n@VXOh>Rh0WAPXOb>Vh0]AmWOZ>Zh0eAhWOR>`h0mAbWOk=fh0SB\\WO`=Pi0_BRWOS=[i0lBgVOg<ei0XC]VO[<oi0dCSVOQ<Wj0oCjUOd;cj0[D^UOW;oj0hDSUOf:^k0ZEdTOS:nk0lESTOb9^l0^FcSOT9kl0kFWSOe8Wm0ZGlROX8am0gG`ROl7lm0SHXRO]7Vn0aHnQOl6bn0SIbQOW6Ro0gISQOb5`o0]JfPOg4RP1WKSPOP4cP1mKboN_3oP1`LVoNl2\\Q1QMinNW2mQ1fMYnNZ1dR1bNdmN:]S1AnlNQOZT1j0YlNeMjT1V2lkN_LmT1]3o;J5L5J6I8I7H8H9G:F<C<D>ZOf0UOUVU]1"
      }
    ],
    "scores": [0.9985802173614502, 0.963715672492981]
  },
  "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images/c3cafc9a-42f3-465c-8867-280f066d5e81/f6a939bb-1d50-49d5-8385-e2e25053e0b6.png": {
    "detections": [
      {
        "size": [2048, 1536],
        "counts": "WjZm0:gn1Y1POP1beNZOjd0S1_ZOAWe0k0SZOHee0c0hYOLQf0?\\YO0]f0:SYO3ff06oXO2lf06iXO2Rg05dXO2Xg05_XO1]g05\\XO0ag05XXO0dg06TXO0ig06nWO0og05iWO2Rh06cWO1Yh06\\WO3_h05UWO4gh02lVO:mh0NeVO>Vi0JZVOd0`i0CRVOh0ii0AhUOk0Rj0]O`UOn0[j0[OVUOP1ej0XOQUOo0jj0YOkTOo0Qk0YOeTOl0Xk0YO`TOn0\\k0XO]TOl0`k0YOYTOl0dk0YOVTOl0gk0WOTTOl0kk0XOoSOk0Ql0XOhSOl0Wl0XObSOl0]l0XO\\SOl0cl0XOUSOn0jl0VOlROP1Sm0TOcROS1\\m0ROZROT1fm0POPROV1om0POgQOU1Xn0PO^QOV1bn0POSQOV1kn0ROiPOS1eh0]CbYOf;^MR1lh0lCUYOY;dMP1Si0ZDjXOk:iMP1Zi0eD_XOa:hMT1fi0aD^XOa:^MW1Rj0]D]XO`:TM\\1\\j0ZD\\XO`:kL_1fj0UD]XO`:aLc1Pk0QD\\XOb:VLf1\\k0mC[XOoi0cg0UVOZXOli0dg0XVOZXOhi0cg0]VOZXOdi0dg0`VOYXOai0dg0dVOZXO\\i0cg0jVOYXOWi0cg0oVOZXORi0cg0SWO[XOmh0ag0YWO\\XOhh0`g0^WO]XOch0`g0bWO]XO_h0`g0fWO^XOZh0_g0lWO]XOUh0ag0oWO\\XORh0bg0RXO[XOog0cg0UXOZXOlg0dg0XXOYXOig0fg0ZXOWXOgg0hg0]XOTXOdg0kg0fXOjWO\\g0Uh0oXO_WOSg0`h0YYOSWOif0lh0bYOhVO`f0Wi0kYO]VOWf0ci0S8O1O1O1O100O1O1O100O1O100O1O1O001O1N2O1O1N2O0O2N2N2N2N2N2M2O2N2N2N2N20O00000O10001O0O1O2O0O1O2N1N3N1N2N3N1N3N2N2O1N2N1O2N2N2N2M3N2N2M3M3N2M3M3M2N3M3L4N2N2O1N3N1O1O1N2O1O1O1O1O1N2O_\\OekN_;ZT1ZDilNk:VS1nDkmNZ:SR1`E\\nNX:cQ1bEknNW:TQ1dEXoNV:gP1fEcoNU:]P1fEloNW:TP1dEUPOW:lo0bE_POZ:bo0`EgPO[:Zo0_EQQO\\:on0^E\\QO]:en0\\EfQO_:Zn0[EQRO_:Qn0ZEYROb:hm0WEaROf:_m0SElROi:Um0oDUSOn:ll0kD^SOQ;dl0fDfSOV;\\l0_DQTO^;Ql0UD]TOi;fk0jCeTOT<ek0YCgTOd<bk0kBhTOS=bk0[BhTOc=ck0kAfTOT>dV1000O100O100O1O1O001O00001O000000000O1000O01O001O1O0010O01O1O001O010O1O001O1O00100O2N2N2N2N2ZcNh@`X1[?WgNYA[X1j>\\gNiAWX1[>^gNXBVX1Q>[gNaBYX1k=TgN`BhX1k=ffN_BWY1m=VfN\\BhY1o=eeN[BYZ1ha0N2N1O2N1O1O2N0O2O0O101N101N100O2N100O2N100O2N1O101N1O1O1gKlYOdnNVf0UQ1YZO`nNje0YQ1eZO]nN]e0ZQ1U[OZnNnd0]Q1e[OVnN_d0_Q1V\\O\\mNdd0YR1g4K5K5K5K5L4L4L4L4M3L4M3M3M3N2M3M3M4M2M3N2N2N2N2M3N2O1N2N2N2N2O1N2O1N2N2N2M3N2M3L4L4L4L4L4M3M3M3M3M3N2M3N2N3M2N2N2N2M3N2M3L4L4M3L4M3M3M3M3N2M3N2N2N2N2N2N3M2N2M3M3L4L4K5I8I6J6J6J6K5K5K5K6J5L4L4M4K4M4K5J6J5K6I7H8F:TOl0XKYhNZ@cX1j>h4VOj0]Oc0_Oa0@a0A>D=E:F;G9F;D;I8I6K4L5J6K4M4K4L5K4M3L4M3M3M3L4M3M2O2N2M2O2M3N2M3M2O2N2M3N2N2N2M4M2N2N2N3N1N3M4M3L3N4K4M2M4M2M4M2M2O2N2M2O2M2O2N1N2O1O2N1N3N2N2N2N2N3M3L4M3M4L3M3M3M2N2M3N2N2N1O001O1O001O1O1O001O1O001O1O001O1O001O1O1O1O2N2N2M3N2N3M3M4L3M3M3M3M2N2N2N2N2N1O2N1N2N3M2O2M2N3M3L4M3M3M4K5L4K5K6Jb0^O`0@?A=BTR[P1"
      }
    ],
    "scores": [0.9994397759437561]
  }
}
```

#### Face localization

This task saves a file called `model_outputs.json` to the directory: `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` (or `{results_rootdir}/mini/{task_name}/...` if `use_mini_dataset=True`), where each variable is a parameter provided to `evaluate_task`. This file contains the bounding box coordinates and confidence scores for people's faces in the same format as the model outputs in the person localization section (see that section above).

#### Body parts detection

This task saved a file called `model_outputs.json` to the directory: `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` (or `{results_rootdir}/mini/{task_name}/...` if `use_mini_dataset=True`), where each variable is a parameter provided to `evaluate_task`. This file contains the model's predicted probabilities for the existence of each body part in the following format:

```
{
    "image1_filename": {
        "detections": [
            {
                "body_part_0_0": prob_0,
                "body_part_0_0": prob_1,
                ...
            },
            {
                "body_part_0_0": prob_0,
                "body_part_0_0": prob_1,
                ...
            },
        ]
    },
    "image2_filename": {
        "detections": [
            {
                "body_part_0_0": prob_0,
                "body_part_0_0": prob_1,
                ...
            }
        ]
    },
    ...
}
```

where `body_part_i_j` is the string for the ith body part for jth person, and `prob_i` is the predicted probability (float) that that body part is present in the image for that person. Note that the same set of body parts must be present in each sub dictionary. Here is an example `model_outputs.json` content with two images, where the first image has two people and the model predicts the "Face" and "Hand" body parts:

```
{
    "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images/52d4716b-984d-4abf-829f-2b041be07205/0f77296e-75f8-41a9-87fb-ea73f09c1004.png": {
        "detections": [
            {
                "Face": 0.7305874228477478,
                "Hand": 0.11991582810878754
            },
            {
                "Face": 0.12332239001989365,
                "Hand": 0.580467164516449
            }
        ]
    },
    "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images/9c598529-0a53-4437-8826-d131b9dc4482/0f7cd9f7-c6dd-422c-b823-3887eb7e8dc0.png": {
        "detections": [
            {
                "Face": 0.04996178299188614,
                "Hand": 0.016766566783189774
            }
        ]
    }
}
```

See the Body parts detection section above for a full list of supported body parts.

#### Keypoint estimation

This task saves a file called `model_outputs.json` to the directory: `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` (or `{results_rootdir}/mini/{task_name}/...` if `use_mini_dataset=True`), where each variable is a parameter provided to `evaluate_task`. This file contains the keypoint coordinates and confidence scores for the subset of keypoints that are detected by the model for each image. Here is the format of `model_outputs.json`:

```

{
    "image1_filename": {
        "detections": [
            [
                [
                    x_nose_0,
                    y_nose_0
                ],
                [
                    x_left_eye_0,
                    y_left_eye_0
                ],
                ...
            ],
            [
                [
                    x_nose_1,
                    y_nose_1
                ],
                [
                    x_left_eye_1,
                    y_left_eye_1
                ],
                ...
            ]
        ],
        "scores": [
            [
                score_nose_0,
                score_left_eye_0,
                ...
            ],
            [
                score_nose_1,
                score_left_eye_1,
                ...
            ]
        ]
    },
    ...
}
```

where `x_nose_i` and `y_nose_i` are the x,y coordinates of the predicted nose of the ith person in a given image, and `score_nose_i` is the confidence score of the ith person in a given image. The model must output coordinates and scores for all 14 COCO keypoints (see Keypoint estimation section above). With the current metrics implemented in the API, only the ground truth keypoints that appear in the image will be evaluated, so the coordinates and scores of the remaining keypoints in the model output for that image will be ignored. For keypoints that the model predicts do not exist for a given subject in an image, use large negative values for the coordinates and scores, e.g., [-999.,-999.] for x,y and [-999.] for the score. This will ensure that the metrics will treat this as a false prediction for this keypoint.

#### Face parsing

```
{
    "image1_filename": {
        "detections_rle": {
            "0": {
                "size": [
                    512,
                    512
                ],
                "counts": "string_0"
            },
            "1": {
                "size": [
                    512,
                    512
                ],
                "counts": "string_1"
            },
            ...
        }
    },
    "image2_filename": {
        "detections_rle": {
            "0": {
                "size": [
                    512,
                    512
                ],
                "counts": "string_0"
            },
            "1": {
                "size": [
                    512,
                    512
                ],
                "counts": "string_1"
            },
            ...
        }
    },
    ...
}
```

where each integer key of the `detection_rle` dictionary is a label mapping to a CelebA-HQ mask category. For the ordering of the labels, see `CELEBA_MASK_LABELS` in [fhibe_eval_api/metrics/face_parsing/utils.py](../fhibe_eval_api/metrics/face_parsing/utils.py).

**The FHIBE annotations do not include ground truth ear mask annotations** (they are considered to be part of face skin). As a result, you should map your left and right ear masks to the face skin mask. Feel free to keep or remove the ear masks in your output file -- we will ignore them either way.

Each face part dictionary has two components:

- `size` - lists the width and height of the image.
- `counts` - a utf-8 decoded string representing the run-length-encoded binary mask for the specific face part.

See the [face parsing evaluation section of the code](https://github.com/SonyResearch/fairness-benchmark-internal/blob/main/fhibe_eval_api/evaluate/evaluate.py#L364) section of the code for more details.

#### Face verification

This task saves a file called `model_outputs.json` to the directory: `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` (or `{results_rootdir}/mini/{task_name}/...` if `use_mini_dataset=True`), where each variable is a parameter provided to `evaluate_task`. This file contains the 512-dimensional face embedding vectors for each image in the following format:

```
{
    "image1_filename": {
        "detections": [
            0.022667212411761284,
            0.07811176031827927,
            0.0029736971482634544,
            ...
        ]
    },
    "image2_filename": {
        "detections": [
            0.01853184960782528,
            0.020853571593761444,
            0.009344225749373436,
            ...
        ]
    },
    ...
}
```

Where the `detections` key maps to a list of length 512, representing the face embedding.

#### Face encoding

This task does not save a `model_outputs.json` file. Instead, it stores the face encodings as `.png` images of the same shape as the FHIBE-face images (512x512) in a subfolder `face_encodings` of the `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` directory, where the variables are parameters provided to `evaluate_task`. The filename is formatted as `{image_id}_{subject_id}`, where this information can be found in the `fhibe_face_crop_align.csv` metadata file.

#### Face super resolution

This task saves a file called `model_outputs.json` to the directory: `{results_rootdir}/{task_name}/{dataset_name}/{model_name}/` (or `{results_rootdir}/mini/{task_name}/...` if `use_mini_dataset=True`), where each variable is a parameter provided to `evaluate_task`. This file contains a mapping from original image filename to super resolution filename, for example:

```
{
    "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images_faces/crop_and_align/a77705fa-051a-41d4-a561-45d9727b0999/fdf6ccea-33e6-483b-b13e-15b79950d99c_a77705fa-051a-41d4-a561-45d9727b0999.png": {
        "super_res_filename": "/home/ubuntu/fairness-benchmark-internal/results/mini/face_super_resolution/fhibe_face_crop_align/super_resolution_demo_2025Jan06/super_fdf6ccea-33e6-483b-b13e-15b79950d99c_a77705fa-051a-41d4-a561-45d9727b0999.png"
    },
    "/home/ubuntu/anonymized_fhibe/DEC24_RELEASE/images_faces/crop_and_align/fe92ac04-8491-4ae9-a03a-54e4f5629d6d/342ab175-6609-4a41-a1d1-4a92ca9f7b33_fe92ac04-8491-4ae9-a03a-54e4f5629d6d.png": {
        "super_res_filename": "/home/ubuntu/fairness-benchmark-internal/results/mini/face_super_resolution/fhibe_face_crop_align/super_resolution_demo_2025Jan06/super_342ab175-6609-4a41-a1d1-4a92ca9f7b33_fe92ac04-8491-4ae9-a03a-54e4f5629d6d.png"
    },
    ...
}
```

where the super resolution images must be saved to disk and referenced in this file. The super resolution images must be of the same shape as the input FHIBE-face image (512x512).
