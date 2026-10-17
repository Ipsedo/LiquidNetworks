from pandas import Series


def standardize_column(s: Series) -> Series:
    c_data = s.dropna().astype(float)
    c_mean = c_data.mean()
    c_std = c_data.std()

    return (s.fillna(c_mean).astype(float) - c_mean) / (c_std + 1e-8)


def min_max_normalize_column(s: Series) -> Series:
    deltas_no_na = s.dropna().astype(float)

    deltas_min = deltas_no_na.min()
    deltas_max = deltas_no_na.max()
    deltas_mean = deltas_no_na.mean()

    if deltas_max != deltas_min:
        return (s.fillna(deltas_mean).astype(float) - deltas_min) / (deltas_max - deltas_min)

    return s.fillna(deltas_mean).astype(float) / deltas_max
