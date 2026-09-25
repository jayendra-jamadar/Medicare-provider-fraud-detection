# Medicare Provider Fraud Detection

Predicting potentially fraudulent Medicare providers from the claims they file,
plus a second study that asks whether those providers work in groups.

The dataset is the Kaggle release of Medicare Part A and Part B claims for one
benefit year. It holds 5,410 providers, 558,211 claims, and 138,556
beneficiaries. 506 providers, about 9.35 percent, are labelled as potential
fraud.

## Two parts

**Part one, the original case study.** Claim level feature engineering and
classical models. Every claim is turned into a row, features are aggregated at
provider, beneficiary and physician level, and Logistic Regression, Decision
Tree and Random Forest are compared. Best result is Random Forest at 0.9518
test AUC with an F1 of 0.5679, scored at claim level. All of this lives in the
`CS_1_*` notebooks.

**Part two, the collusion graph study.** A newer question. Fraud in healthcare
is often organised, so a group of providers can route claims through the same
small set of physicians and the same patients while each provider on its own
looks completely normal. Claim totals cannot see that. A network can. Part two
builds that network and tests the idea properly. It lives in `graph_fraud/`.

## What part two found

Nothing. And that turned out to be the interesting part.

| check | result on real claims |
| --- | --- |
| groups flagged by the unsupervised sweep | 1, holding 18 providers, 2 of them labelled fraud |
| ranking providers by ring risk alone | ROC AUC 0.458, worse than a coin toss |
| adding all 30 graph features to the baseline | PR AUC 0.715 to 0.706 |
| GraphSAGE message passing | PR AUC 0.619 |
| fusion of everything | PR AUC 0.711 |

Every graph variant lands at or below the plain claim totals baseline. The
single group that cleared the flagging bar has 2 fraudulent providers out of
18, which is exactly the base rate.

### Why

A provider to provider link can only exist where two providers share a
physician or a patient. In this file they mostly do not.

**92.4 percent of the 100,737 physicians bill for exactly one provider, and the
most widely shared one reaches only 10.** Patients do move between providers,
66 percent of them see more than one, but that is ordinary patient behaviour
rather than collusion. Fraudulent providers actually share slightly less than
clean ones, 39.5 percent of their claim volume against 44.0 percent.

There is almost no shared structure in this dataset for a collusion graph to
work with.

### Proving the detector is not just broken

A null result only means something if the instrument works. So the identical
code was run against generated claims built to the measured shape of the real
file, with 4 collusion rings planted inside. The thresholds were fixed on this
control and then applied to the real file without changing them.

| check | control data |
| --- | --- |
| ring sweep, no labels used | precision 0.84, recall 0.67 over 48 planted members |
| adding graph features | PR AUC 0.633 to 0.673 |
| fusion | PR AUC 0.683 |

The method finds rings when rings exist. It finds none here because there are
none to find.

## How to run part two

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle, [Healthcare Provider Fraud Detection
Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis),
and put these four files in `Dataset/Train/`:

```
Train-1542865627584.csv
Train_Beneficiarydata-1542865627584.csv
Train_Inpatientdata-1542865627584.csv
Train_Outpatientdata-1542865627584.csv
```

Then:

```bash
cd graph_fraud
python run_all.py     # real run plus control run, about 7 minutes
python build_page.py  # writes report/ring_sheet.html
python verify.py      # leakage and significance checks
```

Without the CSVs the code falls back to a generated stand in and tells you so,
so everything still runs end to end.

Open `graph_fraud/report/ring_sheet.html` in a browser for the interactive
write up: the evidence, the sharing breakdown, the network viewer, the model
comparison for both datasets, and the audit queue.

## What is in the graph module

| file | what it does |
| --- | --- |
| `dataset_profile.py` | the real file's measured shape, read out of the notebook outputs |
| `synth_data.py` | control data generator, same schema and shape, rings planted |
| `data_loader.py` | finds the CSVs wherever they are, falls back to the generator |
| `baseline_features.py` | provider level claim features, the comparison baseline |
| `graph_builder.py` | bipartite graph and the specificity weighted projection |
| `ring_detection.py` | Louvain communities, ring scoring, the flagging rule |
| `graph_features.py` | structural features, sharing concentration, node2vec |
| `gnn.py` | GraphSAGE in NumPy, full batch, backprop written out by hand |
| `pipeline.py` | the four way model comparison and cross validation |
| `run_all.py` | both runs plus the report payload in one command |
| `build_page.py` | renders the HTML report |
| `verify.py` | label independence, fold hygiene, paired significance test |

## How the graph is built

Two providers are connected when they share physicians or patients. A plain
shared count is useless, because a physician working with two hundred providers
would link everyone to everyone. So each shared person is weighted by how rare
they are:

```
w(p, q) = sum over shared entities e of  min(c_pe, c_qe) / log(1 + deg(e))
```

That weight is then divided by `sqrt(n_p * n_q)`, so a busy provider does not
connect to the whole network just by filing more claims, and the result is cut
to a mutual top 12 per provider. On the real claims this gives 37,170 edges
across 5,410 providers in 18 components.

Communities come from Louvain and get scored on four things that never touch
the label: internal density, how much of a member's sharing stays inside the
group, how specific the shared people are, and how similarly members bill. A
group is flagged only when it is both dense and highly ranked.

## Leakage control

The graph, the ring scores and the embeddings never read the fraud label.
`verify.py` proves it by rebuilding everything with the labels shuffled and
confirming the structure comes out identical. The GNN score used by the fusion
model is cross fitted inside the training providers, so no provider is ever
scored by a model that saw its own label. GNN training is transductive, meaning
neighbour features are visible for every node, but the loss only ever touches
training labels.

## Things worth knowing

Part one scores at claim level across 558,211 rows. Part two scores at provider
level across 5,410 rows, which is where the label actually lives. The two sets
of numbers are not comparable, so do not read 0.9518 against 0.9394 as a drop.

The control data plants exactly the pattern the graph module is built to find,
so its recall figure measures the detector rather than the world. The real run
is the honest number.

Louvain is stochastic. The seed is fixed, but another seed will shift community
boundaries a little.

node2vec embeddings did not help the classifier and are off by default in the
fusion model. They are kept because they give the report its provider map.

This result is about this dataset. The Kaggle release is anonymised and
partially reconstructed, so the absence of physician sharing may be an artefact
of how it was prepared rather than a fact about Medicare. On a real claims
warehouse with true provider and physician identifiers the same code could tell
a different story, and that is the obvious next step.

## Credit

Original case study and the `CS_1_*` notebooks by the repository author.
Dataset from Kaggle, Healthcare Provider Fraud Detection Analysis.
