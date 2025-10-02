# CT Website

A lightweight FastAPI web application for uploading CT scan archives and classifying each series with the bundled 2.5D ResNet model.

## Features

- Upload a `.zip` archive that contains one or more DICOM series.
- Automatically groups slices into series and converts them to Hounsfield units.
- Runs the `resnet2p5d.pt` model to obtain a probability of pathology for every series.
- Presents the results in an HTML table together with any parsing warnings.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

By default the app looks for the model weights at `../ct-service/models/resnet2p5d.pt`. Override the path with the `CT_MODEL_PATH` environment variable if necessary.
