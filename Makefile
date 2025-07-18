# Makefile for STARSDataFusion.jl

# Run unit tests
unit-test:
	julia --project -e 'using Pkg; Pkg.test()'

# Alias for unit-test
test: unit-test
