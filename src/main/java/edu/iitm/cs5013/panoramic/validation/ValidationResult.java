package edu.iitm.cs5013.panoramic.validation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Result of validating inputs, configurations, or operations.
 */
public final class ValidationResult {
    private final boolean valid;
    private final List<String> errors;
    private final List<String> warnings;

    public ValidationResult(boolean valid, List<String> errors, List<String> warnings) {
        this.valid = valid;
        this.errors = errors != null ? Collections.unmodifiableList(new ArrayList<>(errors)) : Collections.emptyList();
        this.warnings = warnings != null ? Collections.unmodifiableList(new ArrayList<>(warnings)) : Collections.emptyList();
    }

    public static ValidationResult success() {
        return new ValidationResult(true, List.of(), List.of());
    }

    public static ValidationResult successWithWarnings(List<String> warnings) {
        return new ValidationResult(true, List.of(), warnings);
    }

    public static ValidationResult failure(String error) {
        return new ValidationResult(false, List.of(error), List.of());
    }

    public static ValidationResult failure(List<String> errors) {
        return new ValidationResult(false, errors, List.of());
    }

    public boolean isValid() {
        return valid;
    }

    public List<String> getErrors() {
        return errors;
    }

    public List<String> getWarnings() {
        return warnings;
    }

    public String getFormattedErrors() {
        return String.join(System.lineSeparator(), errors);
    }

    public String getFormattedWarnings() {
        return String.join(System.lineSeparator(), warnings);
    }
}
