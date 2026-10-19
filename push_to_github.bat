@echo off
echo ========================================================
echo   PUSHING HOTEL CANCELLATION ML PLATFORM TO GITHUB
echo ========================================================
echo.
echo Remote repository: https://github.com/AbdulBasit-5/hotel-model.git
echo Branch: main
echo.
echo Pushing commits to GitHub...
git push origin main
echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo   SUCCESS! Pushed successfully to GitHub!
    echo   Repository: https://github.com/AbdulBasit-5/hotel-model
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   Push failed or requires GitHub login authentication.
    echo   If prompted, sign in via browser or paste a GitHub Token.
    echo ========================================================
)
echo.
pause
