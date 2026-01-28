def make_lags(df, col_name, lags):
    for lag in lags:
        df[f'lag_{lag}'] = df[col_name].shift(lag)
    return df


def make_rollings(df, col_name, rolls):
    for roll in rolls:
        df[f'rolling_{roll}'] = df[col_name].shift(1).rolling(roll).mean()
    return df