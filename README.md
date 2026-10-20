# Paige

Paige reads a photographed page. You drop in a picture of a sheet of paper. The app finds the page, straightens it, and reads the words with two deep learning models.

1. **Page outline.** A contour search finds the four corners of the sheet and warps the photo flat.
2. **Text detection.** [`db_mobilenet_v3_large`](https://github.com/mindee/doctr) (Differentiable Binarization with a MobileNetV3 backbone) finds each line.
3. **Text recognition.** `crnn_mobilenet_v3_small` (a convolutional recurrent network) reads the characters on that line.

The first read downloads the pretrained weights, about 25 MB, into the docTR cache.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 and choose a photo, or use **Try a sample page**. The sample is a page shot at an angle. Paige flattens it, draws a box around each line, and prints the transcript with a confidence score.

`samples/straight_page.png` is the same page already flat, if you want to try a scan instead of a photo.

## Tests

```bash
pytest
```

The sample test loads the real detection and recognition models and checks that the angled photo comes back as the words on the page.
