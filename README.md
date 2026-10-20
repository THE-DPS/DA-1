# Senior Data Analytics Engineer Internship — Five-Week Project

## Project
Short-term-rental listing analysis using a user-supplied Inside Airbnb CSV extract that appears to cover the Albany, New York area.

## Important data note
The supplied source contains 500 rows and 90 columns. The coordinates and neighbourhood labels indicate an Albany, New York-area extract. Treat this as a limited extract, not a complete market census. Findings are descriptive and should not be generalized beyond the file without representative data.

## Weekly reports
1. Week 1 — Data Cleaning and Pre-processing
2. Week 2 — Exploratory Data Analysis
3. Week 3 — Data Modelling
4. Week 4 — Data Visualization
5. Week 5 — Data Strategy

## Project layout
- `reports/`: five Word reports
- `data/listings_cleaned.csv`: cleaned dataset
- `src/`: reproducible analysis script
- `figures/`: charts used in reports
- `Submission_Descriptions_200plus_words.txt`: descriptions for portal submission

## Reproducibility
Install dependencies with `pip install -r requirements.txt`. Place the original source file at `data/listings.csv.gz` and run `python src/run_analysis.py`. Review all outputs before submitting, especially if the input dataset changes.

## Caveats
- Listing price is not the same as realized booking revenue.
- Price outliers are flagged, not automatically deleted.
- The modelling results use one random train/test split and are a prototype, not production validation.
- Verify the source licence/terms and your internship's submission rules before publishing data or code.
