package edu.iitm.cs5013.panoramic;

import edu.iitm.cs5013.panoramic.gui.PanoramicApp;
import javafx.application.Application;

/**
 * Main application launcher.
 * Launches PanoramicApp without requiring javafx runtime modules in classpath.
 */
public class Main {
    public static void main(String[] args) {
        Application.launch(PanoramicApp.class, args);
    }
}
