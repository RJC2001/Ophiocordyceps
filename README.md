# Ophiocordyceps

A small local web app for training and testing detectors on images and CSV data.

## Image detector (learned layout)

The image mode learns a **normalized layout** from your annotations instead of
training a pixel classifier.  You only need to annotate a modest number of
images (around 20 is plenty):

1. Open `/train` and switch to **Image** mode.
2. Upload the photos.  The thumbnails show their split with a localized short
   label (Train / Validation / Test).
3. Annotate the objects with the **same number of boxes** on every Train (and
   optionally Validation) image.
4. Set the **Train / Validation / Test** split (default 70% / 15% / 15%).
   The split is calculated from **all uploaded images**, in list order:
   Test is the last part, Validation the part before it.  Counts use
   largest-remainder apportionment, so when the percentages do not divide
   evenly the extra image goes to Train first, then Validation, then Test.
   For 24 images the default gives 17 / 4 / 3.
5. The Test thumbnails are **locked** and must stay unannotated — they are the
   unseen check set.  Reordering the list changes which images are held out.
6. Click **Train Detector**.  There are no model hyper-parameters to configure.

The layout is fitted only on annotated **Train** images.  Validation images give
an extra generalization score when they are annotated.  Test predictions are
shown in a gallery after training, without ever being used for fitting or
scoring.

Training learns where each box sits relative to an anchor.  At prediction time
the anchor is located with a Hough-circle search for round objects; if it is not
visible in a frame, the learned anchor is used as a fallback.
The detector therefore always returns exactly K boxes (K = number of boxes per
annotated training image), including frames where a pixel model would return
1 or 2.

The predicted boxes are also snapped to the circles visible in the frame.  Both
the stored layout and the same layout rotated by 180 degrees are tried, so one
model handles both plate orientations (one well on top or two wells on top).
A small camera shift does therefore **not** require retraining: the circle
search corrects the position inside the model's search window.  Retrain only
when the objects change shape/size or when the number of boxes per image
changes.

### Detector configuration

The defaults use a fixed image point for a centred round object, but the
**Detector configuration** panel under Step 3 exposes the anchor settings so
other cameras, object sizes or shapes can be trained without changing code:

- anchor detection: a fixed image point or circular Hough search,
- anchor centre X/Y (fraction of the image),
- anchor radius (fraction of the shorter image side),
- Hough search min/max radius.

The settings are stored inside the trained model, so prediction uses exactly the
same configuration.

Metrics shown after training:

- **Mean IoU (train)** – how closely the learned layout fits the training images.
- **Mean IoU (validation)** – the same layout scored on annotated validation
  images (when present).

The page also reports how many images landed in each split and previews
predictions on the locked Test images.  Test images do not produce a training
score because they were never annotated.

Save the model to the server, then open `/test`, select it under **Image**, and
upload test photos.  Results are shown on the page and can be downloaded as a
ZIP with annotated images and object crops.

## CSV mode

CSV training and prediction work as before: upload a CSV, choose the target and
feature columns, train, save the model, then predict from `/test`.  The CSV form
keeps its configuration exposed (task type, shuffle/random seed, Train /
Validation / Test split or k-fold, algorithm and hyper-parameters).

Text/categorical feature columns are one-hot encoded automatically, so columns
such as `Formulation`, `Formulation_type` or `Medium` can be used directly;
unknown categories in prediction data are ignored instead of raising an error.
Algorithms: Random Forest, Logistic/Linear Regression, SVM, Neural Network (MLP)
and XGBoost (when installed).  There is no Dummy baseline in the list.

## Run

```bash
pip install -r requirements.txt
python3 main.py
```

The server listens on `http://localhost:8000` (configurable in
`res/static/config/train.json`).
