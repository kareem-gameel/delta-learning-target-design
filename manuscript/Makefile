MAIN = main
BUILD = build

.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(MAIN).tex

clean:
	latexmk -C -outdir=$(BUILD) $(MAIN).tex

