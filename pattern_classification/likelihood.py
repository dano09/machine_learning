from scipy.stats import norm
import pandas as pd


def calc_likelihood(x, mean, sd):
    '''
    Normal Gaussian, takes mean and SD [not variance]
    :param x:
    :param mean:
    :param sd:
    :return:
    '''
    return norm.pdf(x, mean, sd)


def calc_evidence(lhood1, lhood2, prior1, prior2):
    return lhood1 * prior1 + lhood2 * prior2


def calc_posterior(lhood, prior, evidence):
    return (lhood * prior) / evidence


def calc_likelihood_ratio(lhood1, lhood2):
    return lhood1 / lhood2


def calc_likelihood_threshold(loss_11, loss_12, loss_21, loss_22, prior1, prior2):
    return ((loss_12 - loss_22) / (loss_21 - loss_11)) * (prior2 / prior1)


def main():
    m1, m2 = 0, 2      # mean
    v1, v2 = 1, 4      # variance
    sd1, sd2 = 1, 2
    p1, p2 = 0.6, 0.4  # priors
    #samples = [-2, -1, 0, 1, 3, 5, 7]
    samples = [5]
    results = []
    for s in samples:
        l1 = calc_likelihood(s, m1, sd1)
        l2 = calc_likelihood(s, m2, sd2)
        ratio = calc_likelihood_ratio(l1, l2)
        evd = calc_evidence(l1, l2, p1, p2)
        pos1 = calc_posterior(l1, p1, evd)
        pos2 = calc_posterior(l2, p2, evd)
        results.append((evd, l1, l2, ratio, pos1, pos2))

    return pd.DataFrame(data=results, index=samples, columns=['p(x)', 'p(x|w1)', 'p(x|w2)', 'ratio', 'p(w1|x)', 'p(w2|x)'])
    #print(results)

main()