$jfx = (Get-ChildItem -Path "C:\Users\balaj\.m2\repository\org\openjfx" -Recurse -Filter "*-win.jar").FullName -join ";"
$jack = (Get-ChildItem -Path "C:\Users\balaj\.m2\repository\com\fasterxml\jackson" -Recurse -Filter "*.jar").FullName -join ";"
$cp = "target\classes;$jack"

Write-Host "Launching JavaFX Desktop GUI..."
java --module-path "$jfx" --add-modules javafx.controls,javafx.fxml,javafx.swing -cp "$cp" edu.iitm.cs5013.panoramic.Main
