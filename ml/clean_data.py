"""
SmartEdu - Dataset Cleaning Script
Cleans the raw student dataset: fixes whitespace, casing, invalid ranges,
duplicates, missing values, and inconsistent Result labels.
"""
import pandas as pd
import numpy as np

RAW_PATH = "/mnt/user-data/uploads/smartedu_unclean_dataset.csv"
CLEAN_PATH = "/home/claude/smartedu/data/clean_dataset.csv"

# Valid ranges based on the marking scheme used in the raw data
RANGES = {
    "Attendance_Percentage": (0, 100),
    "Quiz_Marks": (0, 20),
    "Assignment_Marks": (0, 20),
    "Midterm_Marks": (0, 30),
    "Final_Marks": (0, 30),
}


def clean():
    df = pd.read_csv(RAW_PATH)

    # 1. Trim whitespace from string columns
    for col in ["Roll_Number", "Student_Name", "Result"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "": np.nan})

    # 2. Standardize Roll_Number casing (STUxxxxx)
    df["Roll_Number"] = df["Roll_Number"].str.upper()

    # 3. Standardize name casing (Title Case)
    df["Student_Name"] = df["Student_Name"].str.title()

    # 4. Standardize Result labels
    result_map = {
        "PASS": "Pass", "P": "Pass", "PASS ": "Pass",
        "FAIL": "Fail", "F": "Fail",
    }
    df["Result"] = df["Result"].str.upper().map(
        lambda x: "Pass" if x in ("PASS", "P") else ("Fail" if x in ("FAIL", "F") else np.nan)
        if pd.notna(x) else np.nan
    )

    # 5. Convert numeric columns, strip stray spaces first
    for col in RANGES:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Remove out-of-range values (set to NaN so they can be imputed)
    for col, (lo, hi) in RANGES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # 7. Drop rows with no Roll_Number (primary key) and de-duplicate
    df = df.dropna(subset=["Roll_Number"])
    df = df.drop_duplicates(subset=["Roll_Number"], keep="first")

    # 8. Impute missing numeric values with column median
    for col in RANGES:
        df[col] = df[col].fillna(df[col].median())

    # 9. Impute missing names
    df["Student_Name"] = df["Student_Name"].fillna("Unknown Student")

    # 10. Recompute Result from Final_Marks where missing (pass mark = 40% of 30 = 12)
    PASS_MARK = 15
    df.loc[df["Result"].isna(), "Result"] = np.where(
        df.loc[df["Result"].isna(), "Final_Marks"] >= PASS_MARK, "Pass", "Fail"
    )

    # 11. Round numeric columns
    for col in RANGES:
        df[col] = df[col].round(1)

    # 12. Derive Risk_Level (target label for ML) from academic indicators
    def risk_level(row):
        score = (
            row["Attendance_Percentage"] * 0.25
            + (row["Quiz_Marks"] / 20 * 100) * 0.20
            + (row["Assignment_Marks"] / 20 * 100) * 0.15
            + (row["Midterm_Marks"] / 30 * 100) * 0.25
            + (row["Final_Marks"] / 30 * 100) * 0.15
        )
        if score >= 70:
            return "Low Risk"
        elif score >= 50:
            return "Medium Risk"
        else:
            return "High Risk"

    df["Risk_Level"] = df.apply(risk_level, axis=1)

    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"Cleaned dataset saved: {CLEAN_PATH}")
    print(f"Rows: {len(df)}")
    print(df["Result"].value_counts())
    print(df["Risk_Level"].value_counts())
    return df


if __name__ == "__main__":
    clean()
