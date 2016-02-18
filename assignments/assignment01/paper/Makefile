TITLE="hw1_shishido_partI"

all:
	pdflatex $(TITLE).tex
	bibtex $(TITLE)
	pdflatex $(TITLE).tex
	pdflatex $(TITLE).tex

clean:
	rm -f *.{aux,log,bbl,lof,lot,blg,out}
