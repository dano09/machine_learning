import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)

# For console
# C:/Users/Justin/PycharmProjects/basics/time_series/
# df = pd.read_csv('C:/Users/Justin/PycharmProjects/basics/time_series/AMZN.csv')

base = 'C:/Users/Justin/PycharmProjects/basics/time_series/'
path = base + 'AMZN.csv'
voo_path = base + 'VOO.csv'
voo_div_path = base + 'VOO_Divs.csv'

def cleaning(path='AMZN.csv'):
    df = pd.read_csv(path)
    df = df.sort_values('Date', ascending=False)
    df = df[['Date', 'Adj Close']]
    df.rename(columns={'Date': 'date', 'Adj Close': 'adj_close'}, inplace=True)
    df['adj_close'] = df['adj_close'].astype(np.int64)
    return df


def dividend_cleaning(path='VOO.csv'):
    df = pd.read_csv(path)
    df = df.sort_values('Date', ascending=False)
    df = df[['Date', 'Adj Close', 'Close']]
    df.rename(columns={'Date': 'date', 'Adj Close': 'adj_close', 'Close': 'close'}, inplace=True)
    return df


def single_period_returns(df):
    """
        Assumes dataframe is decreasing in time
        2019-01-09
             |
        1998-01-02
    """
    # One-Period Simple Returns
    price_df = df.copy()
    # Pt / Pt-1

    # Gross Returns
    price_df['simple_gross_return'] = price_df.adj_close / price_df.adj_close.shift(-1)
    # Net Returns
    price_df['simple_net_return'] = (price_df.adj_close / price_df.adj_close.shift(-1)) - 1
    # df['pct_change'] = df.adj_close.pct_change(-1)  # simple_net_return === pct_change
    return price_df


def multi_period_returns(k, df):
    price_df = df.copy()
    # k periods simple gross return
    gross_col_name = str(k) + '_period_gross_returns'
    net_col_name = str(k) + '_period_net_returns'

    # p(t) / p(t-k)
    price_df[gross_col_name] = price_df.adj_close.pct_change(-k) + 1

    # [ p(t) - p(t-k) ] / p(t-k)
    price_df[net_col_name] = price_df.adj_close.pct_change(-k)
    return price_df


def _get_last_price_every_year(df):
    """ used to calculate annualized_return """
    price_df = df.copy()
    first_year = price_df['date'].values[-1][:4]
    this_year = price_df['date'].values[0][:4]
    annual_df = pd.DataFrame(columns=['date', 'adj_close'])
    for year in range(int(this_year), int(first_year), -1):
        year_df = price_df[price_df['date'].str.contains(str(year))]
        annual_df = annual_df.append(year_df.iloc[0])
    return annual_df


def annual_returns(df):
    """ simply calculates the yearly returns """
    range_df = df.copy()
    range_df = _get_last_price_every_year(range_df)
    return single_period_returns(range_df)


def avg_annualized_return(k, df):
    """
        Dataframe should have annual gross returns

        Params:
            k - int: years
            df - dataframe
                    date    gross_return

    Also known as Geometric Mean
    1) Multiple all gross returns for k time intervals [rows]
    2) Take the k-th root
    3) Subtract 1
    """
    ann_df = df[::-1]  # Reverse Series
    col_name = str(k) + 'yr_annualized_return'
    ann_df[col_name] = (pd.Series.rolling(ann_df['simple_gross_return'], window=k).apply(lambda x: np.prod(x))) ** (1/k) - 1

    # More complicated version
    # 1) Sum up the natural logarithm of returns
    # 2) Divide by k periods
    # 3) Take exponential
    # 4) Subtract 1
    col_name2 = str(k) + 'yr_annualized_log_return'
    ann_df[col_name2] = np.exp((1/k)*(pd.Series.rolling(ann_df['simple_gross_return'], window=k).apply(lambda x: np.sum(np.log(x))))) - 1

    return ann_df[::-1]


def singe_period_log_returns(df):
    """ Also called Continuously Compinded Returns """
    price_df = df.copy()
    price_df['log_returns'] = np.log(price_df.adj_close / price_df.adj_close.shift(-1))
    return price_df


def multi_period_log_returns(k, df):
    # k periods simple gross return
    price_df = df.copy()
    price_df = price_df[::-1]  # Reverse series for using rolling window
    col_name = str(k) + '_period_log_returns'
    # Property of logs, we can simply add here
    price_df[col_name] = pd.Series.rolling(price_df['log_returns'], window=k).apply(lambda x: np.sum(x))

    # Equivalent to taking logarithm of the product of gross returns
    # using single_period_returns
    return price_df[::-1]


def calculate_dividend_payment(df):
    """ Compare Adjusted Close with calculated adjusted close using dividend forumla
        [(Price + Div) / Previous Price] - 1
    """
    pass
    #TODO SKIP

if __name__ == '__main__':
    df = cleaning(path)

    # 1 Calculate some returns
    daily_returns = single_period_returns(df)
    weekly_returns = multi_period_returns(7, df)

    # 2 Calculate Average Annualized Returns
    annual_returns = annual_returns(df)
    annualized_returns = avg_annualized_return(4, annual_returns)

    # 3 Calculate Dividend Payments
    df = dividend_cleaning(voo_path)
    df = pd.read_csv(voo_path)
    voo_divs = pd.read_csv(voo_div_path)
    #  2018-12-17      1.289
    #  2018-09-26      1.207

