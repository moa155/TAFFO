# Makefile for the COT 2025/26 project report.
#
# Usage:
#   make            # build report.pdf
#   make view       # build and open the PDF in the system viewer
#   make clean      # remove auxiliary files, keep the PDF
#   make distclean  # remove everything that `make` produced
#
# Requirements:
#   A full LaTeX distribution with pgfplots and tikz. On macOS:
#       brew install --cask mactex-no-gui
#   or the smaller BasicTeX + the required packages:
#       brew install --cask basictex
#       sudo tlmgr update --self
#       sudo tlmgr install pgfplots pgf xcolor booktabs tabularx enumitem \
#                          hyperref listings amsmath mathtools lm
#   then restart your shell so /Library/TeX/texbin/ is in $PATH.

TEX       := report.tex
PDF       := report.pdf
LATEXMK   := $(shell command -v latexmk 2>/dev/null)
PDFLATEX  := pdflatex -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all view clean distclean check

all: $(PDF)

# Prefer latexmk when present (handles reruns automatically), fall back
# to two plain pdflatex passes otherwise.
$(PDF): $(TEX)
ifeq ($(LATEXMK),)
	$(PDFLATEX) $(TEX)
	$(PDFLATEX) $(TEX)
else
	$(LATEXMK) -pdf -interaction=nonstopmode $(TEX)
endif

view: $(PDF)
	open $(PDF) 2>/dev/null || xdg-open $(PDF) 2>/dev/null || echo "Open $(PDF) manually"

clean:
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz \
	      *.bbl *.blg *.nav *.snm *.vrb *.lof *.lot *.run.xml *-blx.bib

distclean: clean
	rm -f $(PDF)

# Sanity check: is the LaTeX toolchain installed?
check:
	@if ! command -v pdflatex >/dev/null 2>&1; then \
	  echo "ERROR: pdflatex not found. Install MacTeX or BasicTeX:"; \
	  echo "       brew install --cask mactex-no-gui"; \
	  exit 1; \
	fi
	@echo "pdflatex: $$(pdflatex --version | head -1)"
	@if command -v kpsewhich >/dev/null 2>&1; then \
	  for pkg in pgfplots tikz booktabs tabularx hyperref listings mathtools; do \
	    if ! kpsewhich $$pkg.sty >/dev/null 2>&1; then \
	      echo "WARNING: LaTeX package '$$pkg' not found"; \
	    fi; \
	  done; \
	fi
	@echo "All good. Run 'make' to build $(PDF)."
