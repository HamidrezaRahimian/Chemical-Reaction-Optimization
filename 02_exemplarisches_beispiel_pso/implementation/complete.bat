@echo off
setlocal enabledelayedexpansion

:: Create or clear the output file
echo. > complete.java

:: Loop through each .java file in the current and subdirectories
for /r %%f in (*.java) do (
    echo Processing: %%f
    echo. >> complete.java
    type "%%f" >> complete.java
    echo. >> complete.java
)

echo All Java files have been processed and written to complete.java

endlocal