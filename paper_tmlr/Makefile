# Build both versions with tectonic (brew install tectonic) or latexmk.
#   make            submission.pdf (anonymous, TMLR) and arxiv.pdf (named)
#   make figures    regenerate figures and tables from the result files
#   make check      fail if a \\todo or a de-anonymising link is left in the sources
LATEX ?= tectonic

all: submission.pdf arxiv.pdf

%.pdf: %.tex body.tex preamble.tex references.bib
	$(LATEX) $<

figures:
	cd .. && python benchmarks/paper_figures.py

check:
	@! grep -n '\\todo{' body.tex preamble.tex || { echo "TODO markers left (above)"; exit 1; }
	@! grep -n -i -E "github|pypi|arxiv\\.org|copetti|polimi|simocop" body.tex submission.tex \
		|| { echo "named links in the anonymous sources (above)"; exit 1; }
	@echo "check passed"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out submission.pdf arxiv.pdf

.PHONY: all figures check clean
