$m2Repo = Join-Path $env:USERPROFILE ".m2\repository"
$jfx = (Get-ChildItem -Path (Join-Path $m2Repo "org\openjfx") -Recurse -Filter "*-win.jar").FullName -join ";"
$jack = (Get-ChildItem -Path (Join-Path $m2Repo "com\fasterxml\jackson") -Recurse -Filter "*.jar").FullName -join ";"

$jarPath = "target\panoramic-segmentation-gui-1.0.0-SNAPSHOT.jar"
if (Test-Path $jarPath) {
    $cp = "$jarPath;$jack"
} else {
    $cp = "target\classes;$jack"
}

Write-Host "Launching JavaFX Desktop GUI..."
java --module-path "$jfx" --add-modules javafx.controls,javafx.fxml,javafx.swing -cp "$cp" edu.iitm.cs5013.panoramic.Main
