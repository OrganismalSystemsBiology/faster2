""" Basic functions for making summary
"""
import os
import numpy as np
from scipy import stats


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
        if not np.isnan(p_value) and p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.06:
            stars = '*'
        else:
            stars = ''

    res = {'p_value': p_value, 'stars': stars, 'method': method}
    return res

    
def var_test(x, y):
    """ Performs an F test to compare the variances of two samples.
        This function is same as R's var.test()
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


def set_common_features_domain_power_timeseries(ax, x_max, y_max):
    y_tick_interval = np.power(10, np.ceil(np.log10(y_max))-1)
    ax.set_yticks(np.arange(0, y_max, y_tick_interval))
    ax.set_xticks(np.arange(0, x_max+1, 6))
    ax.grid(dashes=(2, 2))

    light_bar_base = matplotlib.patches.Rectangle(
        xy=[0, -0.1*y_tick_interval], width=x_max, height=0.1*y_tick_interval, fill=True, color=stage.COLOR_DARK)
    ax.add_patch(light_bar_base)
    for day in range(int(x_max/24)):
        light_bar_light = matplotlib.patches.Rectangle(
            xy=[24*day, -0.1*y_tick_interval], width=12, height=0.1*y_tick_interval, fill=True, color=stage.COLOR_LIGHT)
        ax.add_patch(light_bar_light)

    ax.set_ylim(-0.15*y_tick_interval, y_max)


def savefig(output_dir, basefilename, fig):
    # JPG
    filename = f'{basefilename}.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0.01,
                bbox_inches='tight', dpi=100, quality=85, optimize=True)
    # PDF
    filename = f'{basefilename}.pdf'
    fig.savefig(os.path.join(output_dir, 'pdf', filename), pad_inches=0,
                bbox_inches='tight', dpi=100)    


