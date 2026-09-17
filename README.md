# CT Anatomy Gap Segmentation

| | |
| --- | --- |
| Final rank | #13 |
| Domain | Computer Vision |
| Difficulty | Hard |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 6 |
| Last submission | 2026-06-18 |

## Problem statement

### Overview

You receive RGB images built from two neighboring axial CT slices. The center slice has been withheld. Your task is to predict the center-slice anatomy masks for eight broad anatomical groups.

Each input image uses the red channel for the superior context slice, the green channel for the inferior context slice, and the blue channel for their absolute difference. The target mask is not a segmentation of a visible image. Strong solutions need to learn 3D anatomical continuity, organ location priors, CT appearance, and patient-to-patient shape variation from the training cases.

Use only the files in `dataset/public/`. Do not use external datasets, source lookup, original source files, source subject identifiers, or internet retrieval. Open-source pretrained image encoders or segmentation weights are allowed only when they are already available in the execution environment or installed packages; do not download data or weights during the run, and do not use weights trained on this exact benchmark source corpus.

### Evaluation

Submissions are scored with **High-Fidelity Gap Anatomy Segmentation Score**. Higher is better.

The score uses a stricter high-fidelity composite:

`quality = 0.45 * mean_class_dice + 0.30 * mean_case_class_dice + 0.20 * strict_boundary_f1 + 0.05 * area_score`

The final score is `quality ** 2`, clipped to `[0, 1]`.

`mean_class_dice` is the macro average foreground Dice score across anatomical groups with at least one positive hidden pixel. `mean_case_class_dice` is the average Dice over case/class pairs where the hidden mask has foreground pixels, so missing a small structure in one case is penalized directly.

`strict_boundary_f1` is computed from globally accumulated boundary true positives, false positives, and false negatives across all cases and classes with no pixel tolerance. A predicted boundary pixel counts as correct only when it lands on a true boundary pixel.

For each case and class, `area_error = abs(predicted_pixels - true_pixels) / (height * width)`. `mean_area_error` is the mean of those errors across all cases and classes, and `area_score = max(0, 1 - 12 * mean_area_error)`.

### Dataset

Files in `dataset/public/`:

| File | Description |
| --- | --- |
| `train.csv` | Training metadata with `case_id`, hashed `volume_id`, image path, mask path, dimensions, and context gap. |
| `test.csv` | Test metadata with `case_id`, image path, dimensions, and context gap. |
| `label_schema.csv` | Integer label IDs and anatomical group names. |
| `sample_submission.csv` | Valid random placeholder submission template. |
| `train_images/` | RGB context PNGs for training cases. |
| `train_masks/` | Center-slice label PNGs for training cases. Pixel value `0` is background; values `1` through `8` match `label_schema.csv`. |
| `test_images/` | RGB context PNGs for hidden test cases. |

The anatomical groups are:

`lung_airway`, `cardiomediastinal`, `hepatosplenic`, `renal_adrenal`, `bowel_pancreas`, `pelvic_urinary`, `axial_bone`, `appendicular_muscle`.

CSV column details:

| File | Column | Type | Description |
| --- | --- | --- | --- |
| `train.csv` | `case_id` | string | Anonymized training case identifier. |
| `train.csv` | `volume_id` | string | Hashed source-volume identifier for volume-aware validation. |
| `train.csv` | `image_path` | string | Relative path under `dataset/public/` to the RGB context PNG. |
| `train.csv` | `mask_path` | string | Relative path under `dataset/public/` to the center-slice label PNG. |
| `train.csv` | `height`, `width` | integer | Image and mask dimensions in pixels. All prepared cases are 128 by 128. |
| `train.csv` | `context_gap_slices` | integer | Number of source CT slices between the withheld center slice and each visible context slice. |
| `test.csv` | `case_id` | string | Anonymized test case identifier to use in `submission.csv`. |
| `test.csv` | `image_path` | string | Relative path under `dataset/public/` to the RGB context PNG. |
| `test.csv` | `height`, `width` | integer | Test image dimensions in pixels. |
| `test.csv` | `context_gap_slices` | integer | Number of source CT slices between the hidden center slice and the visible context slices. |
| `label_schema.csv` | `class_id` | integer | Pixel value for the anatomical group in train masks. |
| `label_schema.csv` | `class_name` | string | Submission column name for that anatomical group. |

The split holds out entire volumes from training. Use the hashed `volume_id` column in `train.csv` for volume-aware validation. Public rows are hash-shuffled, so row order must not be used as a source-volume signal.

### Submission

Submit `submission.csv` with exactly the same number of rows as `test.csv` and these columns in order:

| Column | Type | Description |
| --- | --- | --- |
| `case_id` | string | Test case identifier from `test.csv`. |
| `lung_airway` | string | RLE mask for this anatomical group. |
| `cardiomediastinal` | string | RLE mask for this anatomical group. |
| `hepatosplenic` | string | RLE mask for this anatomical group. |
| `renal_adrenal` | string | RLE mask for this anatomical group. |
| `bowel_pancreas` | string | RLE mask for this anatomical group. |
| `pelvic_urinary` | string | RLE mask for this anatomical group. |
| `axial_bone` | string | RLE mask for this anatomical group. |
| `appendicular_muscle` | string | RLE mask for this anatomical group. |

RLE format uses 1-indexed row-major pixel positions as `start length` pairs. An empty string means an empty mask for that class. Each `case_id` must appear exactly once, and extra or missing IDs are invalid.

Predicted class masks for the same case must not overlap. A pixel can belong to at most one anatomical group.

Example:

| case_id | lung_airway | cardiomediastinal | hepatosplenic | renal_adrenal | bowel_pancreas | pelvic_urinary | axial_bone | appendicular_muscle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap_f5af448c110e974293 | 10741 2 10754 2 |  | 958 3 4252 5 |  |  |  | 393 5 |  |
| gap_5296f1202eee3792d4 |  | 3378 5 |  | 11288 2 | 5187 3 6094 2 |  |  | 6613 5 |
