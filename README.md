
# Summary

This repo builds a ranking boosted model with LightGPU on the
Microsoft’s Web30K dataset. Gain at the top 10 scores is around .583 on
unseen data. This is considerably higher than the publication this code
is based on indicating improvements in source code after printing.

# Data Overview

## Query and Documents

Learning to rank M.L. has its origin in search engine optimization.
Because of this, there are two key ideas. Documents and queries. A
document is a web page. The web page is crawled and features are
created. Example include covered query term number and term frequency,
stream length. A query is the string the user types into Bing. There is
a many to one relationship between documents and queries stemming from
the fact the user sees many web pages per search. Further, a document
may show up in many queries.

- Train N: 2,643,905 (documents)

- Test N: 1,120,827 (documents)

- Train queries: 22,036

- Test queries: 6,306

For these data, the number of documents per query looks similar.

![](README_files/figure-commonmark/cell-3-output-1.png)

    <ggplot: (145661259283)>

This project uses fold one. The vali.txt file is split into two pieces
and put into train and test datasets. Raw data can be found
[here](https://www.microsoft.com/en-us/research/project/mslr/).

## Cleaning Process

- Step 1: Load .txt files.
- Step 2: Name columns.
- Step 3: Remove a few columns.
- Step 4: Remove leading string in values.
- Step 5: Convert variables to numeric.
- Step 6: Write ipc files for quick loads.

# Model Summary

## Results

Data shaping was the major hurdle for this project. Model training is
made easy by LightGPM. For ranking the major changes are using
LGBMRanker and setting the objective to “rank_xendcg”. Without much
tuning, the model had a NCDG of .58 for top 5 scores and .58 for top 10
scores on the unseen test data.

For reference, the publication An Alternative Cross Entropy Loss for
Learning-to-Rank has NCDG around .48 and this was bleeding edge
performance when it was printed. It appears Microsoft has improved model
training since then.

On a technical front, the data passed into LGBMRanker has one row per
document. The performance metric requires one row per query and document
scores are the columns. This intermediate data shaping between
predictions (one row per document) to performance calculations (one row
per query ID) would require a custom metric creation to use sci-kit
learn’s cross validation functionality.

## Hardware Considerations

For the Windows platform, the pip install process includes all the
necessary parts for training on a GPU. At writing, the Linux and Mac
don’t have this option. In addition, installing with conda on Windows
does not provide GPU support.
