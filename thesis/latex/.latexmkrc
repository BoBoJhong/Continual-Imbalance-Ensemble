$pdf_mode = 5;
$out_dir = 'build';
$aux_dir = 'build';
$xelatex = 'xelatex -no-shell-escape -synctex=1 -recorder -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex_use = 2;
$max_repeat = 8;
@default_files = ('main.tex');
