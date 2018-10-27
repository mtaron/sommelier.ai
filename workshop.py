import pandas as pd
from operator import itemgetter

def boxplot_sorted(df, by, column, **kwds):
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values()
    df2[meds.index].boxplot(**kwds)

def show_most_informative_features(model, vectorizer=None, classifier=None, n=20):
    # Source: https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
    vectorizer = vectorizer or model.steps[0][1]
    classifier = classifier or model.steps[-1][1]

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {}.".format(
                classifier.__class__.__name__))

    tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True)

    # Get the top n and bottom n coef, name pairs
    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    output = []

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append("{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn))

    print("\n".join(output))