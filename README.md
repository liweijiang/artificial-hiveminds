# Artificial Hiveminds: The Open-Ended Homogeneity of Language Models (and Beyond)

This directory contains the source code for the **Artificial Hiveminds** project, which analyzes human preference annotations and evaluates model calibration in diverse response generation tasks.

## Directory Overview

```
artificial_hiveminds/
└── src/
    ├── data_construction/
    ├── human_annotations/
    └── model_calibration_analysis/
```

### `data_construction/`
Contains scripts and utilities for constructing the dataset used in the project. This includes:
- Data preprocessing
- Response aggregation
- Formatting for annotation

### `human_annotations/`
Code and configuration files for collecting and managing human annotations. Tasks in this folder may include:
- Annotation schema definition
- Interfaces for annotation collection
- Parsing and processing raw annotation data

### `model_calibration_analysis/`
Houses scripts for analyzing the alignment between model-generated scores (or rankings) and human preferences. Tasks typically include:
- Correlation analysis
- Calibration error computation
- Visualizations and summary metrics
