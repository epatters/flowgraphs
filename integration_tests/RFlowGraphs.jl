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

module TestRFlowGraphs
using Test

using Catlab.WiringDiagrams
using SemanticFlowGraphs

using ..IntegrationTest: db, read_semantic_graph

# Semantic flow graph
#####################

# K-means clustering on the Iris dataset using base R.
# FIXME: The encapsulated box should only have one input port.
# FIXME: Box order, as noted in PyFlowGraphs.
semantic = read_semantic_graph(joinpath("r", "clustering_kmeans"))
d = WiringDiagram([], [])
kmeans = add_box!(d, construct(pair(concepts(db,
  ["k-means", "clustering-model-n-clusters"])...)))
read_file = add_box!(d, concept(db, "read-tabular-file"))
file = add_box!(d, construct(pair(concepts(db, ["tabular-file", "filename"])...)))
centroids = add_box!(d, concept(db, "k-means-centroids"))
clusters = add_box!(d, concept(db, "clustering-model-clusters"))
fit = add_box!(d, Hom("fit",
  otimes(concept(db, "k-means"), concept(db, "data")),
  concept(db, "k-means")))
transform = add_box!(d, Box(concepts(db, ["table"]), concepts(db, ["table"])))
add_wires!(d, [
  (file, 1) => (read_file, 1),
  (read_file, 1) => (transform, 1),
  (kmeans, 1) => (fit, 1),
  (transform, 1) => (fit, 2),
  (fit, 1) => (clusters, 1),
  (fit, 1) => (centroids, 1),
])
@test semantic == d

end
