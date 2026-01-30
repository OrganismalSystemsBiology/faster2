""" Basic functions for making summary
"""
import os
import matplotlib
import numpy as np
from scipy import stats
import stage


def test_two_sample(x, y):
    # test.two.sample: Performs two-sample statistical tests according to our labratory's standard.
    ##
    # Arguments:
    # x: first samples
    # y: second samples
    ##
    # Return:
    # A dict of (p.value=p.value, method=method (string))
    ##

    # remove nan
    xx = np.array(x)
    yy = np.array(y)
    xx = xx[~np.isnan(xx)]
    yy = yy[~np.isnan(yy)]

    # If input data length < 2, any test is not applicable.
    if (len(xx) < 2) or (len(yy) < 2):
        p_value = np.nan
        stars = ''
        method = None
    else:
        # If input data length < 3, Shapiro test is not applicable,
        # so we assume false normality of the distribution.
        if (len(xx) < 3) or (len(yy) < 3):
            # Forced rejection of distribution normality
            normality_xx_p = 0
            normality_yy_p = 0
        elif np.var(xx) == 0 or np.var(yy) == 0:
            # Forced rejection of distribution normality
            normality_xx_p = 0
            normality_yy_p = 0
        else:
            normality_xx_p = stats.shapiro(xx)[1]
            normality_yy_p = stats.shapiro(yy)[1]

        equal_variance_p = var_test(xx, yy)['p_value']

        if not ((normality_xx_p < 0.05) or (normality_yy_p < 0.05) or (equal_variance_p < 0.05)):
            # When any null-hypotheses of the normalities of x and of y,
            # and the equal variance of (x,y) are NOT rejected,
            # use Student's t-test
            method = "Student's t-test"
            p_value = stats.ttest_ind(xx, yy, equal_var=True)[1]
        elif not ((normality_xx_p < 0.05) or (normality_yy_p < 0.05)) and (equal_variance_p < 0.05):
            # When null-hypotheses of the normality of x and of y are NOT rejected,
            # but that of the equal variance of (x,y) is rejected,
            # use Welch's t-tet
            method = "Welch's t-test"
            p_value = stats.ttest_ind(xx, yy, equal_var=False)[1]
        else:
            # If none of above was satisfied, use Wilcoxon's ranksum test.
            method = "Wilcoxon test"
            # same as stats.mannwhitneyu() with alternative='two-sided', use_continuity=False
            # or R's wilcox.test(x, y, exact=F, correct=F)
            p_value = stats.ranksums(xx, yy)[1]

        # stars
        stars = stat_stars(p_value)

    res = {'p_value': p_value, 'stars': stars, 'method': method}
    return res

    
def var_test(x, y):
    """ Performs an F test to compare the variances of two samples.
        This function is equivalent to R's var.test()
    """
    df1 = len(x) - 1
    df2 = len(y) - 1
    v1 = np.var(y, ddof=1)
    v2 = np.var(x, ddof=1)

    if v2 > 0:
        F = v1/v2
        if F > 1:
            p_value = stats.f.sf(F, df2, df1)*2  # two-sided
        else:
            p_value = (1-stats.f.sf(F, df2, df1))*2  # two-sided
    else:
        F = np.nan
        p_value = np.nan
    return {'F': F, 'df1': df1, 'df2': df2, 'p_value': p_value}


def set_common_features_domain_power_timeseries(ax, x_max, y_min, y_max):
    if y_min > 0:
        # if y_min is positive, set the y-axis minimum to 0 
        y_min = 0
    else:
        # add some bottom margin
        y_min = y_min - 0.1*(y_max - y_min)

    # add some top margin
    y_max = y_max + 0.1*(y_max - y_min)


    y_tick_interval = np.power(10, np.ceil(np.log10(y_max - y_min))-1)
    ax.set_yticks(np.arange(y_min, y_max, y_tick_interval))
    ax.set_xticks(np.arange(0, x_max+1, 6))
    ax.grid(dashes=(2, 2))

    light_bar_base = matplotlib.patches.Rectangle(
        xy=[0, y_min -0.1*y_tick_interval], width=x_max, height=0.1*y_tick_interval, fill=True, color=stage.COLOR_DARK)
    ax.add_patch(light_bar_base)
    for day in range(int(x_max/24)):
        light_bar_light = matplotlib.patches.Rectangle(
            xy=[24*day, y_min -0.1*y_tick_interval], width=12, height=0.1*y_tick_interval, fill=True, color=stage.COLOR_LIGHT)
        ax.add_patch(light_bar_light)

    ax.set_ylim(y_min -0.1*y_tick_interval, y_max)


def savefig(output_dir, basefilename, fig):
    # JPG
    filename = f'{basefilename}.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0.02,
                bbox_inches='tight', dpi=100, pil_kwargs={"quality":85, "optimize":True})
    # PDF
    filename = f'{basefilename}.pdf'
    fig.savefig(os.path.join(output_dir, 'pdf', filename), pad_inches=0.02,
                bbox_inches='tight', dpi=100)    


def x_shifts(values, y_min, y_max, width):
    #    print_log(y_min, y_max)
    counts, _ = np.histogram(values, range=(
        np.min([y_min, np.min(values)]), np.max([y_max, np.max(values)])), bins=25)
    sorted_values = sorted(values)
    shifts = []
#    print_log(counts)
    non_zero_counts = counts[counts > 0]
    for c in non_zero_counts:
        if c == 1:
            shifts.append(0)
        else:
            p = np.arange(1, c+1)  # point counts
            s = np.repeat(p, 2)[:p.size] * (-1)**p * width / \
                10  # [-1, 1, -2, 2, ...] * width/10
            shifts.extend(s)

#     print_log(shifts)
#     print_log(sorted_values)
    return [np.array(shifts), sorted_values]


def scatter_datapoints(ax, w, x_pos, values):
    s, v = x_shifts(values, *ax.get_ylim(), w)
    ax.scatter(x_pos + s, v, color='dimgrey')


def stat_stars(p_value):
    # stars
    if not np.isnan(p_value) and p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = ''
    return stars


def mack_skillings(dat):
    ''' Calculate the Mack-Skillings statistics and p-value.
    This function calculates the Mack-Skillings statistics and p-value. 
    The Mack-Skillings test is a nonparametric test for the equality of k treatments in a two-way layout 
    with at least one observation for every treatment-block combination.
    Args:
        dat: 3D numpy array of shape (n, k, r) where n is the number of blocks, k is the number of treatments, and r is the number of repeats.
    Returns:
        ms: Mack-Skillings statistics
        p: p-value
    
    Reference: P334-335 of "Nonparametric Statistical Methods" by Myles Hollander and Douglas A. Wolfe, 2nd Edition, 1999.
    '''
    n, k, r = dat.shape # number of blocks (n), treatments (k), repeats (r)
    cs = np.apply_along_axis(lambda x: np.count_nonzero(~np.isnan(x)), 2, dat) # count of repeated observaions in each cell
    qs = np.apply_along_axis(np.nansum, 1, cs) # sum of observation counts in each block
    rnk = np.array([stats.rankdata(x, nan_policy='omit').reshape(k, r) for x in dat]) # ranks in each block
    vs = np.sum((np.nansum(rnk, axis=2).T/qs), axis=1) # mean rank-sums in each treatment
    ve = [np.sum(cs[:,j]*(qs + 1)/(2*qs)) for j in range(k-1)] # expected mean rank-sums of each treatment
    v = vs[:-1] - ve
    css = cs[:, :-1] # The degree of freedom of rank-sums of treatments is k-1
    sigma = -np.dot(css.T*(qs+1)/(12*qs**2), css) # covariance of rank-sums between treatments
    sigma_diag = np.array([np.sum(css[:,s]*(qs - css[:,s])*(qs + 1)/(12*qs**2)) for s in range(k-1)]) # variance of rank-sums of each treatment
    np.fill_diagonal(sigma, sigma_diag)
    ms = v @ np.linalg.inv(sigma) @ v.T # Mack-Skillings statistics: the sum of normalized rank-sums
    p = 1 - stats.chi2.cdf(ms, k-1) # p-value
    return ms, p


# calculate max correlation
def _calc_max_corr(vec, norm_base_s, norm_base_c):
    """ _calc_max_corr: calculate the max correlation between a vector and sine and cosine waves
	
		vec: input vector
		norm_base_s: normalized sine wave
		norm_base_c: normalized cosine wave
		return: max correlation value, phase of max correlation
	"""
    vec_m = vec - np.mean(vec)
    norm_vec = (vec_m)/np.sqrt(np.sum((vec_m)*(vec_m)))
    max_corr_value = np.sqrt(np.power(np.dot(norm_base_c, norm_vec), 2) + np.power(np.dot(norm_base_s, norm_vec), 2))
    max_corr_phase = np.arctan2(np.dot(norm_base_s, norm_vec), np.dot(norm_base_c, norm_vec))
    return max_corr_value, max_corr_phase


def _norm_bases(n_dp, n_dp_in_per):
    """ _norm_bases: calculate the normalized sine and cosine waves
		n_dp: number of data points
		n_dp_in_per: number of data points in a period
		return: normalized sine wave, normalized cosine wave
	"""
    base_s = np.sin(np.arange(n_dp)/n_dp_in_per*2*np.pi)
    base_c = np.cos(np.arange(n_dp)/n_dp_in_per*2*np.pi)
    base_s_m = base_s - np.mean(base_s)
    base_c_m = base_c - np.mean(base_c)
    norm_base_s = (base_s_m)/np.sqrt(np.sum((base_s_m)*(base_s_m)))
    norm_base_c = (base_c_m)/np.sqrt(np.sum((base_c_m)*(base_c_m)))
    return norm_base_s, norm_base_c


def _max_corr_pval(num_datapoints, max_corr):
    """ pvalue of the max Pearson correlations 
        max corr: the value of max correlation
        return: cumulative density
    """
    n = num_datapoints - 3
    p = np.power((1 - np.power(max_corr, 2)), n / 2)
    return p


def _sem_ratio(avg_vector, sem_vector):
    """ ratio of the avg_vecotr at which the vector reaches the SEM sphere
        avg_vector: average vector
        sem_vector: SEM vector
        return: the ratio of the average vector from the origin to the SEM sphere over the average vector
    """
   
    ratio = 1.0 - 1.96 / np.sqrt(np.sum(np.power(avg_vector / sem_vector, 2))) # 1.96 is the 95% confidence interval
    return max(0, ratio)


def costest(avg_vec, sem_vec, n_dp, n_dp_in_per):
    """ costest: calculate the correlation between the average vector and the sine and cosine waves
		avg_vec: average vector
		sem_vec: SEM vector
		n_dp: number of data points
		n_dp_in_per: number of data points in a period
		return: max correlation value, phase of max correlation, original p, SEM adjusted p
	"""
    # prepare bases
    norm_base_s, norm_base_c = _norm_bases(n_dp, n_dp_in_per)

    ## prepare vectors to handle nan
    avg_vector = avg_vec.copy()
    sem_vector = sem_vec.copy()
    # each SEM needs to be >0
    sem_vector[~(sem_vector>0)] = np.nanmax(sem_vector)
    # replace nan in SEM with an effectively infinite SEM
    sem_vector[np.isnan(avg_vector)] = np.nanmax(sem_vector)*1000000
    # replace nan in AVG with median
    avg_vector[np.isnan(avg_vector)] = np.nanmedian(avg_vector)
    # taking the SEM into account
    sem_r = _sem_ratio(avg_vector, sem_vector)

    # tuple of max-correlation value and the phase of max correlation
    mc, mc_ph = _calc_max_corr(avg_vector, norm_base_s, norm_base_c)

    adj_mc = mc * sem_r
    p_org = _max_corr_pval(n_dp, mc)
    p_sem_adj = _max_corr_pval(n_dp, adj_mc)
    
	# get 24h phase for the phase of max correlation
    mc_ph_24h = np.mod(24*mc_ph/(2*np.pi),24)

    # max-correlation value, phase of max correlation, original p, SEM adjusted p
    return mc, mc_ph_24h, p_org, p_sem_adj