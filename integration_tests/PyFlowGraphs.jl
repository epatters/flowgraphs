# Copyright 2018 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module TestPyFlowGraphs
using Test
using Nullables

using Catlab.Diagram
using SemanticFlowGraphs

import ..IntegrationTest: db

const py_dir = joinpath(@__DIR__, "python")

function read_py_raw_graph(name::String)
  read_raw_graph(joinpath(py_dir, "$name.py.graphml"))
end
function read_py_semantic_graph(name::String)
  read_semantic_graph(joinpath(py_dir, "$name.graphml"); elements=false)
end

# Raw flow graph
################

# Deserialize Python raw flow graph from GraphML.
diagram = read_py_raw_graph("pandas_read_sql")
@test nboxes(diagram) == 2
b1, b2 = boxes(diagram)
@test isnull(b1.value.annotation)
@test get(b2.value.annotation) == "python/pandas/read-sql-table"
@test [ get(p.annotation) for p in output_ports(b1) ] ==
  [ "python/sqlalchemy/engine" ]
@test [ get(p.annotation) for p in input_ports(b2)[1:2] ] ==
  [ "python/builtins/str", "python/sqlalchemy/engine" ]
@test [ get(p.annotation) for p in output_ports(b2) ] ==
  [ "python/pandas/data-frame" ]

# Semantic flow graph
#####################

# Read SQL table using pandas and SQLAlchemy.
semantic = read_py_semantic_graph("pandas_read_sql")
d = WiringDiagram([], concepts(db, ["table"]))
engine = add_box!(d, Box([], concepts(db, ["sql-database"])))
cons = add_box!(d, construct(pair(concept(db, "sql-table-database"),
                                  concept(db, "sql-table-name"))))
read_table = add_box!(d, concept(db, "read-table"))
add_wires!(d, [
  (engine, 1) => (cons, 1),
  (cons, 1) => (read_table, 1),
  (read_table, 1) => (output_id(d), 1),
])
@test semantic == d

# K-means clustering on the Iris dataset using NumPy and SciPy.
# FIXME: The boxes are added to `d` in the exact order of `semantic`. That
# won't be necessary when we implement graph isomorphism for wiring diagrams.
semantic = read_py_semantic_graph("scipy_clustering_kmeans")
kmeans_fit = Hom("fit",
  otimes(concept(db, "k-means"), concept(db, "data")),
  concept(db, "k-means"))
d = WiringDiagram([], concepts(db, ["array","array"]))
clusters = add_box!(d, concept(db, "clustering-model-clusters"))
transform = add_box!(d, Box(concepts(db, ["array"]), concepts(db, ["array"])))
kmeans = add_box!(d, construct(pair(concepts(db,
  ["k-means", "clustering-model-n-clusters"])...)))
read_file = add_box!(d, concept(db, "read-tabular-file"))
file = add_box!(d, construct(pair(concepts(db, ["tabular-file", "filename"])...)))
fit = add_box!(d, kmeans_fit)
centroids = add_box!(d, concept(db, "k-means-centroids"))
add_wires!(d, [
  (file, 1) => (read_file, 1),
  (read_file, 1) => (transform, 1),
  (kmeans, 1) => (fit, 1),
  (transform, 1) => (fit, 2),
  (fit, 1) => (clusters, 1),
  (fit, 1) => (centroids, 1),
  (clusters, 1) => (output_id(d), 2),
  (centroids, 1) => (output_id(d), 1),
])
@test semantic == d

# K-means clustering on the Iris dataset using pandas and scikit-learn.
semantic = read_py_semantic_graph("sklearn_clustering_kmeans")
d = WiringDiagram([], concepts(db, ["array"]))
file = add_box!(d, construct(pair(concepts(db, ["tabular-file", "filename"])...)))
read_file = add_box!(d, concept(db, "read-tabular-file"))
kmeans = add_box!(d, construct(concept(db, "k-means")))
transform = add_box!(d, Box(concepts(db, ["table"]), concepts(db, ["array"])))
fit = add_box!(d, concept(db, "fit"))
clusters = add_box!(d, concept(db, "clustering-model-clusters"))
add_wires!(d, [
  (file, 1) => (read_file, 1),
  (read_file, 1) => (transform, 1),
  (kmeans, 1) => (fit, 1),
  (transform, 1) => (fit, 2),
  (fit, 1) => (clusters, 1),
  (clusters, 1) => (output_id(d), 1),
])
@test semantic == d

# Compare sklearn clustering models using a cluster similarity metric.
# FIXME: Box order, as mentioned above.
semantic = read_py_semantic_graph("sklearn_clustering_metrics")
clustering_fit = Hom("fit",
  otimes(concept(db, "clustering-model"), concept(db, "data")),
  concept(db, "clustering-model"))
d = WiringDiagram([], concepts(db, ["array", "k-means", "agglomerative-clustering"]))
make_data = add_box!(d, Box([], concepts(db, ["array", "array"])))
kmeans_fit = add_box!(d, clustering_fit)
agglom = add_box!(d, construct(concept(db, "agglomerative-clustering")))
agglom_clusters = add_box!(d, concept(db, "clustering-model-clusters"))
agglom_fit = add_box!(d, clustering_fit)
score = add_box!(d, Box(concepts(db, ["array", "array"]), []))
kmeans = add_box!(d, construct(concept(db, "k-means")))
kmeans_clusters = add_box!(d, concept(db, "clustering-model-clusters"))
add_wires!(d, [
  (make_data, 1) => (kmeans_fit, 2),
  (make_data, 1) => (agglom_fit, 2),
  (make_data, 2) => (output_id(d), 1),
  (kmeans, 1) => (kmeans_fit, 1),
  (kmeans_fit, 1) => (kmeans_clusters, 1),
  (kmeans_fit, 1) => (output_id(d), 2),
  (kmeans_clusters, 1) => (score, 1),
  (agglom, 1) => (agglom_fit, 1),
  (agglom_fit, 1) => (agglom_clusters, 1),
  (agglom_fit, 1) => (output_id(d), 3),
  (agglom_clusters, 1) => (score, 2),
])
@test semantic == d

# Errors metrics for linear regression using sklearn.
semantic = read_py_semantic_graph("sklearn_regression_metrics")
d = WiringDiagram([], [])
file = add_box!(d, construct(pair(concepts(db, ["tabular-file", "filename"])...)))
data_x = add_box!(d, Box(concepts(db, ["table"]), concepts(db, ["table"])))
data_y = add_box!(d, Box(concepts(db, ["table"]), concepts(db, ["column"])))
ols = add_box!(d, construct(concept(db, "least-squares")))
fit = add_box!(d, concept(db, "fit-supervised"))
predict = add_box!(d, concept(db, "predict"))
error_l1 = add_box!(d, concept(db, "mean-absolute-error"))
error_l2 = add_box!(d, concept(db, "mean-squared-error"))
read_file = add_box!(d, concept(db, "read-tabular-file"))
add_wires!(d, [
  (file, 1) => (read_file, 1),
  (read_file, 1) => (data_x, 1),
  (read_file, 1) => (data_y, 1),
  (ols, 1) => (fit, 1),
  (data_x, 1) => (fit, 2),
  (data_y, 1) => (fit, 3),
  (fit, 1) => (predict, 1),
  (data_x, 1) => (predict, 2),
  (data_y, 1) => (error_l1, 1),
  (predict, 1) => (error_l1, 2),
  (data_y, 1) => (error_l2, 1),
  (predict, 1) => (error_l2, 2),
])
@test semantic == d

# Linear regression on an R dataset using statsmodels.
semantic = read_py_semantic_graph("statsmodels_regression")
d = WiringDiagram([], concepts(db, ["linear-regression"]))
r_data = add_box!(d, construct(pair(concept(db, "r-dataset-name"),
                                    concept(db, "r-dataset-package"))))
ols = add_box!(d, construct(concept(db, "least-squares")))
eval_formula = add_box!(d, concept(db, "evaluate-formula-supervised"))
fit = add_box!(d, concept(db, "fit-supervised"))
read_table = add_box!(d, concept(db, "read-table"))
add_wires!(d, [
  (r_data, 1) => (read_table, 1),
  (read_table, 1) => (eval_formula, 2),
  (ols, 1) => (fit, 1),
  (eval_formula, 1) => (fit, 2),
  (eval_formula, 2) => (fit, 3),
  (fit, 1) => (output_id(d), 1),
])
@test semantic == d

end
