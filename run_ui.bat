@echo off
echo Starting DD Doc Assist Chat UI...
echo.
echo Note: This is a simplified chat interface that assumes documents are already processed.
echo You can process documents in advance using the main.py command line tool.
echo.
echo Note: Make sure all requirements are installed by running:
echo pip install -r requirements.txt
echo.
echo Press Ctrl+C to exit the application when finished.
echo.

rem Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set STREAMLIT_WATCHDOG_EXCLUDE_PATHS=torch,transformers,openai,llama_index
set PYTHONWARNINGS=ignore::RuntimeWarning

rem Start Streamlit
streamlit run src/ui.py 