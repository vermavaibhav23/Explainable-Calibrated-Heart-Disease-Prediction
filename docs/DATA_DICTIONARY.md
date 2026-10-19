# Input data dictionary

| Column | Meaning / units | Encoding |
|---|---|---|
| Age | Age, years | Numeric |
| Sex | Sex as recorded in source | F, M |
| ChestPainType | Chest pain category | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure, mm Hg | Numeric; zero treated as missing |
| Cholesterol | Serum cholesterol, mg/dL | Numeric; zero treated as missing |
| FastingBS | Fasting blood sugar above 120 mg/dL | 0 or 1 |
| RestingECG | Resting ECG summary | Normal, ST, LVH |
| MaxHR | Maximum achieved heart rate, bpm | Numeric |
| ExerciseAngina | Exercise-induced angina | N, Y |
| Oldpeak | ST depression measurement | Numeric; source contains negative values |
| ST_Slope | Peak exercise ST-segment slope | Up, Flat, Down |
| HeartDisease | Dataset label, training/evaluation only | 0 or 1 |

Blank inputs are missing. Unknown categories, infinities, out-of-schema numeric values and all-missing records are rejected. The validation ranges in `data.py` are broad software bounds rather than clinical thresholds. Preserve the source's units and definitions. Identifiers and extra columns are discarded before modelling; do not upload identifying information.

The demonstration CSV contains fictional examples and is not a validation dataset.

