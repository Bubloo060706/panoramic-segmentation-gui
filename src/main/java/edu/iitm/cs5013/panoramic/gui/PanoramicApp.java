package edu.iitm.cs5013.panoramic.gui;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.net.URL;

/**
 * JavaFX Application entry point for the Desktop GUI.
 */
public class PanoramicApp extends Application {

    @Override
    public void start(Stage primaryStage) {
        MainViewController controller = new MainViewController(primaryStage);
        Scene scene = new Scene(controller.getRoot(), 1240, 820);

        URL css = getClass().getResource("/styles/app.css");
        if (css != null) {
            scene.getStylesheets().add(css.toExternalForm());
        }

        primaryStage.setTitle("Panoramic Image Segmentation Framework (CS5013)");
        primaryStage.setScene(scene);
        primaryStage.setMinWidth(960);
        primaryStage.setMinHeight(640);
        primaryStage.show();
    }
}
