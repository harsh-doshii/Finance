import pandas as pd
import numpy as np

def drawdown(return_series: pd.Series):
    wealth_index = 1000*(1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
                    "Wealth": wealth_index,
                    "Previous Peaks": previous_peaks,
                    "Drawdowns": drawdowns})

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                     header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi 

def get_ind_file(filetype):
    known_types = ["returns", "nfirms", "size"]
    if filetype not in known_types:
        raise ValueError(f"filetype must be one of:{','.join(known_types)}")
    if filetype is "returns":
        name = "vw_rets"
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1
                         
    ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind
                         
def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    return get_ind_file("returns")

def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms")

def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size")

                         
def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return

def skewness(r):
    """
    Calculating skewness
    """
    demean = r - r.mean()
    demean_cubed = demean**3
    demean_final = demean_cubed.mean()
    risk = r.std(ddof=0)**3
    return demean_final/risk

def kurtosis(r):
    """
    Calculating kurtosis
    """
    demean = r - r.mean()
    demean_cubed = demean**4
    demean_final = demean_cubed.mean()
    risk = r.std(ddof=0)**4
    return demean_final/risk

import scipy.stats

def is_normal(r, level=0.01):
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semideviation(r):
    is_negative = r < 0 
    return r[is_negative].std(ddof=0)

def var_historic(r, level=5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    else:
        return -np.percentile(r, level)

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def annual_rets(r,periods_per_year):
    compd_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compd_growth**(periods_per_year/n_periods) - 1

def annual_vol(r,periods_per_year):
    return r.std()*np.sqrt(periods_per_year)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annual_rets(excess_ret, periods_per_year)
    ann_vol = annual_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_returns(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**0.5

def plotef2(n_points, er, cov):
    weights = [np.array([w, 1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_returns(w,er) for w in weights]
    vol = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns":rets,"Volatility":vol})
    return ef.plot.scatter(x="Volatility", y="Returns",style=".-")

from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    returns_is_target = {
                        'type':'eq',
                        'args': (er,),
                        'fun':lambda weights, er: target_return - portfolio_returns(weights,er)
                        }
    weights_sum_to_1 = {
                        'type':'eq',
                        'fun':lambda weights: np.sum(weights) - 1
                       }
    weights = minimize(portfolio_vol, init_guess, 
                      args=(cov,), method='SLSQP',
                      options={'disp': False},
                      constraints=(weights_sum_to_1,returns_is_target),
                      bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, riskfree_rate=0 ,show_cml=False,show_ew=False,show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style='.-', legend=False)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_returns(w_gmv, er)
        vol_gmv= portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
    if show_ew:
        n= er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_returns(w_ew, er)
        vol_ew= portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_returns(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    return ax                  

def gmv(cov):
    n = cov.shape[0]
    return msr(0, np.repeat(1,n),cov)

def msr(riskfree_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1 = {
                        'type':'eq',
                        'fun':lambda weights: np.sum(weights) - 1
                       }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_returns(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
        
    weights = minimize(neg_sharpe, init_guess, 
                      args=(riskfree_rate, er, cov), method='SLSQP',
                      options={'disp': False},
                      constraints=(weights_sum_to_1,),
                      bounds=bounds)
    return weights.x
                         
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.80, riskfree_rate=0.03,drawdown=None):
                    dates = risky_r.index
                    n_steps = len(dates)
                    account_value= start
                    floor_value = start*floor
                    if drawdown is not None:
                        peak = np.maximum(peak, account_value)
                        floor_value = peak*(1-drawdown)
                    if isinstance(risky_r, pd.Series): 
                        risky_r = pd.DataFrame(risky_r, columns=["R"])

                    if safe_r is None:
                        safe_r = pd.DataFrame().reindex_like(risky_r)
                        safe_r.values[:] = riskfree_rate/12
                    
                    account_history=pd.DataFrame().reindex_like(risky_r)
                    cushion_history=pd.DataFrame().reindex_like(risky_r)
                    risky_w_history=pd.DataFrame().reindex_like(risky_r)
                    
                    for step in range(n_steps):
                        cushion = (account_value - floor_value)/account_value
                        risky_w = m*cushion
                        risky_w = np.minimum(risky_w,1)
                        risky_w = np.maximum(risky_w,0)
                        safe_w = 1 - risky_w
                        risky_alloc = account_value*risky_w
                        safe_alloc = account_value*safe_w
                        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
                        cushion_history.iloc[step] = cushion
                        risky_w_history.iloc[step] = risky_w
                        account_history.iloc[step] = account_value
                    
                    risky_wealth = start*(1+risky_r).cumprod()    
                    
                    backtest_result = {
                                "Wealth": account_history,
                                "Risky Wealth": risky_wealth, 
                                "Risk Budget": cushion_history,
                                "Risky Allocation": risky_w_history,
                                "m": m,
                                "start": start,
                                "floor": floor,
                                "risky_r":risky_r,
                                "safe_r": safe_r
                                }     
                    return backtest_result
                         
def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annual_rets, periods_per_year=12)
    ann_vol = r.aggregate(annual_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdowns.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    #hist_cvar5 = r.aggregate(cvar_historic)#
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        #"Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

                         
def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
                         
def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, steps_per_year=12, y_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = run_cppi(risky_r=pd.DataFrame(risky_r),riskfree_rate=riskfree_rate,m=m, start=start, floor=floor)
    wealth = btr["Wealth"]

    # calculate terminal wealth stats
    y_max=wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios

    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0

    # Plot!
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9),xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85),xycoords='axes fraction', fontsize=24)
    if (floor > 0.01):
            hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
            hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(.7,                 .7), xycoords='axes fraction', fontsize=24)
    
    return plt.subplots_adjust

#cppi_controls(show_cppi): 
#                        widgets.interactive(show_cppi,
#                                   n_scenarios=widgets.IntSlider(min=1, max=1000, step=5, value=50), 
#                                   mu=(0., +.2, .01),
#                                   sigma=(0, .3, .05),
#                                   floor=(0, 2, .1),
#                                   m=(1, 5, .5),
#                                   riskfree_rate=(0, .05, .01),
#                                   steps_per_year=widgets.IntSlider(min=1, max=12, step=1, value=12,
#                                                          description="Rebals/Year"),
#                                   y_max=widgets.IntSlider(min=0, max=100, step=1, value=100,
#                                                          description="Zoom Y Axis")
#                                            )
#display(cppi_controls)                         
                         
def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time t where t is in years and r is the annual interest rate
    """
    return (1+r)**(-t)

def pv(l, r):
    """
    Compute the present value of a list of liabilities given by the time (as an index) and amounts
    """
    dates = l.index
    discounts = discount(dates, r)
    return (discounts*l).sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return assets/pv(liabilities, r)
                         
def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

import math
def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices
                         
                  
def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
    
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    """
    cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
    return pv(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)
     
def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

                         