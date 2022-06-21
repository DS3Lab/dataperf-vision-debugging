# Makefile for the data cleaning challenge.

.PHONY: run-all
## Create baseline results, run evaluation, and plot results
run: 
	@echo "Creating baseline results..."
	@python create_baselines.py
	@echo "Evaluating results..."
	@python main.py
	@echo "Plotting results..."
	@python plotter.py
	
clean:
	@echo "Removing..."
	rm -rf results/*
	rm -rf outdir/*
	rm -rf submissions/*