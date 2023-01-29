#############################################################
# Overview
# 
# This scripts mimics results from the publication An Alternative
# Cross Entropy Loss for Learning-to-Rank.
#
# Ended up getting a better normalized discounted cumulative gain score
# than publication. There have probably been training improvments since 
# publication was printed.
#
# Builds a ranking boosted model with GPU. "I wanna go fast" - 
# Ricky Bobby.
#############################################################

# %%
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from plotnine import ggplot, aes, labs, geom_boxplot, scale_y_continuous

##############
# Load data into memeory
##############
# %%
folder = 'S:/Python/projects/microsoftLTR/data/'
train = pl.read_ipc(folder + 'train.ipc')
valid = pl.read_ipc(folder + 'vali.ipc')
test = pl.read_ipc(folder + 'test.ipc')
del folder
train.head()

# %%
# Combine to only train and test only
halfIndex = int(np.floor(valid.shape[0] / 2))
train = pl.concat([train, valid[0:halfIndex]])
test = pl.concat([valid, valid[halfIndex:valid.shape[0]]])
del valid, halfIndex

##############
# Minimal exploration
##############
# %%
(
    train.
    select([pl.col('qid').n_unique()])
)

# %%
(
    test.
    select([pl.col('qid').n_unique()])
)

# %%
temp = (
    pl.concat([
        train.
        groupby('qid').
        agg([pl.col('label').count().alias('documentCount')]).
        with_column(pl.lit('train').alias('dataset')),
        test.
        groupby('qid').
        agg([pl.col('label').count().alias('documentCount')]).
        with_column(pl.lit('test').alias('dataset')),
    ]).
    to_pandas()
)
graph = (
    ggplot(temp, aes(x = 'dataset', y = 'documentCount'))
    + geom_boxplot()
    + scale_y_continuous(breaks = np.arange(0, 2000, 100))
    + labs(x = "Dataset", y = "Count Of Documents Per Query")
)
graph

# %%
del temp, graph

##############
# prep for lightgbm
##############
# %%
def prep_data_ranking(df, tup = False):
    groups =  (
        df.
        groupby('qid').
        agg(pl.col('label').count().alias('count')).
        select([pl.col('count')]).
        to_dict(as_series=False).
        values()
    )
    groups = list(groups)[0]

    X = df[:, 2:df.shape[1]].to_numpy()
    y = df.select([pl.col('label')]).to_numpy()
    y = y.reshape((y.shape[0],))
    
    return X, y, groups

X_train, y_train, groups_train = prep_data_ranking(train)
X_test, y_test, groups_test = prep_data_ranking(test)

##############
# Train ranking boosted model w/ gpu
##############
# %%
model = lgb.LGBMRanker(objective = 'rank_xendcg', n_estimators = 500, 
                       learning_rate = .02, subsample = .5, max_bin = 63,
                       device_type = 'gpu')
model.fit(X = X_train, y = y_train, group = groups_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

##############
# tidy predictions for gain
##############
# Before: One row is one document.
# After: One row is one query ID.
# %%
def tidy(labels, predictions, groups):
    qids = []
    scoreIDs = []
    qid = -1
    for group in groups:
        qid+=1
        scoreID = 0
        for i in range(0, group):
            scoreID+=1
            scoreIDs.append(scoreID)
            qids.append(qid)
    qids = np.array(qids)
    scoreIDs = np.array(scoreIDs)

    df = {'scoreID': scoreIDs, 'label':labels, 'prediction':predictions, 'qid':qids}
    df = pl.DataFrame(data = df)

    labels_df = (
        df.
        select([pl.col('scoreID'), pl.col('qid'), pl.col('label')]).
        pivot(index = 'qid', values = 'label', columns='scoreID').
        drop(columns='qid')
    )

    prediction_df = (
        df.
        select([pl.col('scoreID'), pl.col('qid'), pl.col('prediction')]).
        pivot(index = 'qid', values = 'prediction', columns='scoreID').
        drop(columns='qid')
    )

    # Some quids have more document than other quids
    # Need a matrix w/ zero missing values.
    # When quid is smaller than the largest quid, imputing 0 for label and --9999999.0 for prediction
    for col in labels_df.columns:
        labels_df = labels_df.with_column(pl.when(pl.col(col).is_null()).then(pl.lit(0)).otherwise(pl.col(col)).alias(col))
    labels_array = labels_df.to_numpy()

    for col in prediction_df.columns:
        prediction_df = prediction_df.with_column(pl.when(pl.col(col).is_null()).then(pl.lit(-9999999.0)).otherwise(pl.col(col)).alias(col))
    prediction_array = prediction_df.to_numpy()

    return labels_array, prediction_array

labels_train, preds_train = tidy(y_train, pred_train, groups_train)
labels_test, preds_test = tidy(y_test, pred_test, groups_test)


# %% Sanity checks
(
    train.
    select([pl.col('qid')]).
    unique().
    shape[0] == labels_train.shape[0]
)

# %%
(
    train.
    select([pl.col('qid')]).
    unique().
    shape[0] == preds_train.shape[0]
)

# %%
(
    test.
    select([pl.col('qid')]).
    unique().
    shape[0] == labels_test.shape[0]
)

# %%
(
    test.
    select([pl.col('qid')]).
    unique().
    shape[0] == preds_test.shape[0]
)

# %%
(
    int(
        train.
        groupby('qid').
        agg([pl.col('label').count().alias('documentCount')]).
        select([pl.col('documentCount').max()])[0,0]
     ) == preds_train.shape[1]
)

# %%
(
    int(
        test.
        groupby('qid').
        agg([pl.col('label').count().alias('documentCount')]).
        select([pl.col('documentCount').max()])[0,0]
     ) == preds_test.shape[1]
)

# %%
labels_train.shape[1] == preds_train.shape[1] 

# %%
labels_test.shape[1] == labels_test.shape[1] 

##############
# Performance stats
##############
# %%
ndcg_score(y_true = labels_train, y_score = preds_train)

# %%
ndcg_score(y_true = labels_test, y_score = preds_test)

# %%
ndcg_score(y_true = labels_train, y_score = preds_train, k = 5)

# %%
ndcg_score(y_true = labels_test, y_score = preds_test, k = 5)

# %%
ndcg_score(y_true = labels_train, y_score = preds_train, k = 10)

# %%
ndcg_score(y_true = labels_test, y_score = preds_test, k = 10)


# %%
