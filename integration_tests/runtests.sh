#!/usr/bin/env bash

flowgraphs.jl record python
flowgraphs.jl record r

flowgraphs.jl enrich python
flowgraphs.jl enrich r

julia IntegrationTest.jl
