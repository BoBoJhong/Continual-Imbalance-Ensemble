% NCU IM Master's Thesis (community adaptation, October 2026)
% New implementation informed by the 2013 NCU template; original author notice
% and unmodified historical source are kept under legacy_reference/.
% This is NOT an officially approved template.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ncuimthesis}[2026/10/14 NCU IM thesis template with font selection (unofficial)]
\LoadClass[12pt,a4paper,oneside]{report}
\RequirePackage{iftex}
\ifXeTeX\else\ClassError{ncuimthesis}{Please use XeLaTeX}{Change compiler to XeLaTeX in Overleaf settings.}\fi
\RequirePackage{fontspec}
\RequirePackage{xeCJK}
\RequirePackage{zhnumber}
% ===== Font choices: Traditional Chinese BiauKai + English Times New Roman =====
% Overleaf supports Times New Roman by system font name.
% DFKai-SB (Windows BiauKai) is proprietary and cannot be bundled here.
% To use the exact Traditional Chinese font, upload legally-usable fonts/kaiu.ttf.
% If none is supplied, an Overleaf-compatible kai-script font is used instead.
\defaultfontfeatures{Ligatures=TeX}
% English: real Times New Roman on Overleaf; graceful fallback for local compilers.
\IfFontExistsTF{Times New Roman}{%
  \setmainfont{Times New Roman}%
  \typeout{NCU FONT: English = Times New Roman (installed)}%
}{%
  \IfFontExistsTF{TeX Gyre Termes}{%
    \setmainfont{TeX Gyre Termes}%
    \typeout{NCU FONT: English fallback = TeX Gyre Termes (NOT actual Times New Roman)}%
  }{%
    \IfFontExistsTF{Tinos}{%
      \setmainfont{Tinos}%
      \typeout{NCU FONT: English fallback = Tinos (NOT actual Times New Roman)}%
    }{%
      \setmainfont{Liberation Serif}%
      \typeout{NCU FONT: English fallback = Liberation Serif (NOT actual Times New Roman)}%
    }%
  }%
}

% Chinese: regular, headings and sans faces use the same calligraphic font.
\newcommand{\ncuSetKaiByName}[1]{%
  \setCJKmainfont[AutoFakeBold=2]{#1}%
  \setCJKsansfont[AutoFakeBold=2]{#1}%
  \setCJKmonofont[AutoFakeBold=2]{#1}%
}
\newcommand{\ncuSetKaiByFile}[1]{%
  \setCJKmainfont[Path=fonts/,AutoFakeBold=2]{#1}%
  \setCJKsansfont[Path=fonts/,AutoFakeBold=2]{#1}%
  \setCJKmonofont[Path=fonts/,AutoFakeBold=2]{#1}%
}
\IfFileExists{fonts/kaiu.ttf}{%
  \ncuSetKaiByFile{kaiu.ttf}%
  \typeout{NCU FONT: Chinese = actual BiauKai (fonts/kaiu.ttf)}%
}{%
  \IfFontExistsTF{DFKai-SB}{%
    \ncuSetKaiByName{DFKai-SB}%
    \typeout{NCU FONT: Chinese = DFKai-SB (installed)}%
  }{%
    \IfFontExistsTF{AR PL KaitiM Big5}{%
      \ncuSetKaiByName{AR PL KaitiM Big5}%
      \typeout{NCU FONT: Chinese fallback = AR PL KaitiM Big5 (NOT actual BiauKai)}%
    }{%
      \IfFontExistsTF{TW-Kai}{%
        \ncuSetKaiByName{TW-Kai}%
        \typeout{NCU FONT: Chinese fallback = TW-Kai (NOT actual BiauKai)}%
      }{%
        \ncuSetKaiByName{FandolKai-Regular}%
        \typeout{NCU FONT: Chinese fallback = FandolKai-Regular (NOT actual BiauKai)}%
      }%
    }%
  }%
}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
\RequirePackage[a4paper,top=2.5cm,bottom=2.5cm,left=3cm,right=2cm,footskip=1cm]{geometry}
\RequirePackage{setspace}
\setstretch{1.5}
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\RequirePackage{amsmath,amssymb,amsthm,booktabs,array,longtable}
\RequirePackage{graphicx,float}
\RequirePackage{caption}
\RequirePackage{pdfpages}
\RequirePackage{chngcntr}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{etoolbox}
\RequirePackage{fancyhdr}
\RequirePackage{hyperref}
\hypersetup{unicode=true,colorlinks=false,pdfborder={0 0 0},bookmarksnumbered=true}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0pt}}
\renewcommand{\contentsname}{目錄}
\renewcommand{\listfigurename}{圖目錄}
\renewcommand{\listtablename}{表目錄}
\renewcommand{\bibname}{參考文獻}
\renewcommand{\figurename}{圖}
\renewcommand{\tablename}{表}
\counterwithout{figure}{chapter}
\counterwithout{table}{chapter}
\counterwithout{equation}{chapter}
\renewcommand{\thefigure}{\arabic{figure}}
\renewcommand{\thetable}{\arabic{table}}
\renewcommand{\theequation}{\arabic{equation}}
\renewcommand{\thechapter}{\zhnumber{\arabic{chapter}}}
\renewcommand{\thesection}{\arabic{chapter}-\arabic{section}}
\renewcommand{\thesubsection}{\thesection-\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection-\arabic{subsubsection}}
\titleformat{\chapter}[hang]{\centering\bfseries\Large}{\thechapter、}{0.5em}{}
\titleformat{name=\chapter,numberless}[hang]{\centering\bfseries\Large}{}{0pt}{}
\titlespacing*{\chapter}{0pt}{0pt}{1.1cm}
\titleformat{\section}{\bfseries\large}{\thesection}{1em}{}
\titleformat{\subsection}{\bfseries\normalsize}{\thesubsection}{1em}{}
\captionsetup{font=normalsize,labelfont=normalfont,labelsep=space}
\captionsetup[table]{position=top}
\captionsetup[figure]{position=bottom}
\renewcommand{\cftchapaftersnum}{、}
\setlength{\cftchapnumwidth}{3.5em}
\renewcommand{\cftfigpresnum}{圖~}
\renewcommand{\cfttabpresnum}{表~}
\setlength{\cftfignumwidth}{3.2em}
\setlength{\cfttabnumwidth}{3.2em}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\newcommand{\ncuCNtitle}{請填寫中文論文題目}
\newcommand{\ncuENtitle}{Please Enter the English Thesis Title}
\newcommand{\ncuStudent}{請填寫研究生姓名}
\newcommand{\ncuAdvisor}{請填寫指導教授姓名}
\newcommand{\ncuDate}{中華民國〇〇〇年〇月}
\newif\ifncuTBP
\ncuTBPfalse
\newcommand{\ncuTitleBlock}{%
  \thispagestyle{empty}%
  \begin{center}
  \vspace*{1.2cm}
  {\LARGE\bfseries 國\quad 立\quad 中\quad 央\quad 大\quad 學\par}
  \vspace{3.0cm}
  {\Large 資\quad 訊\quad 管\quad 理\quad 學\quad 系\par}
  \vspace{0.4cm}
  {\Large 碩\quad 士\quad 論\quad 文\par}
  \vfill
  {\Large\bfseries\ncuCNtitle\par}
  \vfill
  \begin{tabular}{rl}
  研\quad 究\quad 生：& \ncuStudent\\[0.5cm]
  指導教授：& \ncuAdvisor
  \end{tabular}
  \vspace{1.65cm}\par
  {\large\ncuDate\par}
  \vspace*{0.6cm}
  \end{center}
}
\newcommand{\ncuCover}{\begin{titlepage}\ncuTitleBlock\end{titlepage}}
\newcommand{\ncuInnerTitle}{\begin{titlepage}\ncuTitleBlock\end{titlepage}}
\newcommand{\ncuInsertForm}[1]{\IfFileExists{forms/#1.pdf}{\includepdf[pages=-,pagecommand={}]{forms/#1.pdf}}{\typeout{NCU NOTICE: Signed form forms/#1.pdf not inserted.}}}
\newcommand{\ncuAbstractPage}[3]{%
  \clearpage\phantomsection\addcontentsline{toc}{chapter}{#2}%
  \begin{center}
  {\large\bfseries #1\par}
  \vspace{2cm}
  {\Large\bfseries #3\par}
  \end{center}
  \vspace{1.1cm}
}
\newcommand{\ncuFrontTitle}[1]{\chapter*{#1}\phantomsection\addcontentsline{toc}{chapter}{#1}}
\newcommand{\ncuStartFront}{\clearpage\pagenumbering{roman}\hypersetup{pageanchor=true}\pagestyle{plain}}
\newcommand{\ncuStartMain}{\clearpage\pagenumbering{arabic}\pagestyle{plain}}
\newcommand{\ncuBeginAppendices}{%
  \appendix
  \renewcommand{\thechapter}{\Alph{chapter}}
  \titleformat{\chapter}[hang]{\centering\bfseries\Large}{附錄\thechapter}{0.6em}{}
}
