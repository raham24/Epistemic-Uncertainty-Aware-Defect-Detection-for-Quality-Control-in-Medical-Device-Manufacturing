MAIN = generic-color
TEX  = $(MAIN).tex
PDF  = $(MAIN).pdf

LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -halt-on-error -synctex=1 

.PHONY: all clean cleanall watch

all: $(PDF)

$(PDF): $(TEX)
	$(LATEXMK) $(LATEXMK_FLAGS) $(TEX)

clean:
	$(LATEXMK) -c 
	rm -rf "%OUTDIR%"
	rm indent.log

cleanall:
	$(LATEXMK) -C
	rm -f *.bbl
	rm -rf "%OUTDIR%"
	rm indent.log
	
watch:
	$(LATEXMK) -pdf -pvc $(TEX)
