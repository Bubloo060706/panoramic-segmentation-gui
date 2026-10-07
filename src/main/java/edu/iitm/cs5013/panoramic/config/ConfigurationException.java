package edu.iitm.cs5013.panoramic.config;

/**
 * Thrown when tool configuration is malformed, missing, or fails validation.
 */
public class ConfigurationException extends Exception {
    public ConfigurationException(String message) {
        super(message);
    }

    public ConfigurationException(String message, Throwable cause) {
        super(message, cause);
    }
}
