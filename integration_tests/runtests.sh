#!/usr/bin/env bash

flowgraphs.jl record --graph-outputs=simplify python
flowgraphs.jl record r

flowgraphs.jl enrich python
flowgraphs.jl enrich r

julia IntegrationTest.jl
