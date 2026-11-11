@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo  Reading Sign Language - environment setup
echo ============================================
echo.

rem ---- 1. Python 3.14 ----------------------------------------------------
py -3.14 -c "import sys" >nul 2>&1
if not errorlevel 1 goto :have_python

echo Python 3.14 was not found. Installing it with winget...
winget install --id Python.Python.3.14 --scope user --accept-package-agreements --accept-source-agreements
py -3.14 -c "import sys" >nul 2>&1
if not errorlevel 1 goto :have_python

echo.
echo Python 3.14 is installed but this window cannot see it yet.
echo Close this window, open a new one, and run setup.bat again.
echo If it still fails, install Python 3.14 from https://www.python.org/downloads/
goto :fail

:have_python
for /f "delims=" %%v in ('py -3.14 -c "import sys; print(sys.version.split()[0])"') do echo Found Python %%v

rem ---- 2. Virtual environment ---------------------------------------------
set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" (
    "%VPY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 14) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Existing .venv uses another Python version. Recreating it...
        rmdir /s /q .venv
    ) else (
        echo Using existing .venv
    )
)
if not exist "%VPY%" (
    echo Creating .venv with Python 3.14...
    py -3.14 -m venv .venv
    if errorlevel 1 goto :fail
)

rem ---- 3. Packages --------------------------------------------------------
echo.
echo Installing packages from requirements.txt - this can take several minutes the first time...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
"%VPY%" -c "import numpy, pandas, sklearn, matplotlib, seaborn, torch; print('All packages import correctly. torch', torch.__version__)"
if errorlevel 1 goto :fail

rem ---- 4. Data ------------------------------------------------------------
if not exist "data\raw" mkdir "data\raw"
set "MISSING="
if not exist "data\raw\sign_mnist_train.csv" set "MISSING=1"
if not exist "data\raw\sign_mnist_test.csv" set "MISSING=1"
echo.
if defined MISSING (
    echo Data is missing. Download the dataset from
    echo   https://www.kaggle.com/datasets/datamunge/sign-language-mnist
    echo unzip it, and put sign_mnist_train.csv and sign_mnist_test.csv in data\raw\
) else (
    echo Data found in data\raw\
)

rem ---- 5. LaTeX for the reports: MiKTeX, Perl for latexmk, Cursor extension
echo.
set "MTX=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"
if not exist "%MTX%\pdflatex.exe" if exist "%ProgramFiles%\MiKTeX\miktex\bin\x64\pdflatex.exe" set "MTX=%ProgramFiles%\MiKTeX\miktex\bin\x64"
where pdflatex >nul 2>&1
if not errorlevel 1 goto :have_tex
if exist "%MTX%\pdflatex.exe" goto :have_tex
echo Installing MiKTeX with winget...
winget install --id MiKTeX.MiKTeX --scope user --silent --accept-package-agreements --accept-source-agreements
:have_tex
if exist "%MTX%\initexmf.exe" "%MTX%\initexmf.exe" --set-config-value=[MPM]AutoInstall=1 >nul 2>&1
if exist "%MTX%\pdflatex.exe" (echo MiKTeX found.) else (where pdflatex >nul 2>&1 || echo WARNING: MiKTeX is not installed. Install it from https://miktex.org/download)

where perl >nul 2>&1
if not errorlevel 1 goto :have_perl
if exist "C:\Strawberry\perl\bin\perl.exe" goto :have_perl
echo Installing Strawberry Perl with winget - needed by latexmk. Click Yes if Windows asks for permission...
winget install --id StrawberryPerl.StrawberryPerl --silent --accept-package-agreements --accept-source-agreements
:have_perl

where cursor >nul 2>&1
if errorlevel 1 (
    echo Cursor command not found. In Cursor, install the extension "LaTeX Workshop" by James Yu by hand.
) else (
    call cursor --install-extension James-Yu.latex-workshop
)

echo.
echo Setup finished. To use the environment in a terminal, run:
echo   .venv\Scripts\activate
echo If LaTeX was just installed, restart Cursor so it can find pdflatex and latexmk.
echo.
pause
exit /b 0

:fail
echo.
echo Setup failed. Read the messages above for the reason.
pause
exit /b 1
