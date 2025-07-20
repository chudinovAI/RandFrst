import pandas as pd
import logging


def load_and_combine_data(smoke_path: str, no_smoke_path: str) -> pd.DataFrame:
    smoke_df = pd.read_csv(smoke_path)
    logging.info(f"Load smoke_df: {smoke_df.shape}")
    no_smoke_df = pd.read_csv(no_smoke_path)
    logging.info(f"Load no_smoke_df: {no_smoke_df.shape}")
    no_smoke_df["label"] = 0
    df = pd.concat([smoke_df, no_smoke_df], ignore_index=True)
    logging.info(f"United DataFrame: {df.shape}")
    # clearing useless feat
    cols_to_drop = [
        col
        for col in ["frame", "region_id", "whitePixelCount", "movementRatio"]
        if col in df.columns
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logging.info(f"Deleted: {cols_to_drop}")
    df["label"] = df["label"].astype(int)
    return df


def create_dynamic_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    df = df.copy()
    if "region_id" not in df.columns:
        df["_fake_region_id"] = 0
        group_col = "_fake_region_id"
    else:
        group_col = "region_id"
    sort_cols = [group_col]
    if "frame_number" in df.columns:
        sort_cols.append("frame_number")
    df = df.sort_values(sort_cols)
    # list for dynamic feat
    feature_cols = [
        col for col in df.columns if col not in ["label", group_col, "frame_number"]
    ]
    for col in feature_cols:
        df[f"{col}_diff"] = df.groupby(group_col)[col].diff(periods=window_size)
        df[f"{col}_std"] = (
            df.groupby(group_col)[col]
            .rolling(window=window_size)
            .std()
            .reset_index(level=0, drop=True)
        )
    df = df.dropna()
    for col in ["region_id", "frame_number", "_fake_region_id"]:
        if col in df.columns:
            df = df.drop(columns=col)
    return df
