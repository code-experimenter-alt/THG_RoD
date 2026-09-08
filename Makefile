LATEXMK ?= latexmk
FLAGS = -pdf -interaction=nonstopmode -halt-on-error
.PHONY: all main supplement clean
all: main supplement
main:
	$(LATEXMK) $(FLAGS) main.tex
supplement:
	$(LATEXMK) $(FLAGS) Supplementary_Material.tex
clean:
	$(LATEXMK) -c main.tex
	$(LATEXMK) -c Supplementary_Material.tex
