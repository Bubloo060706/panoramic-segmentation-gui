package edu.iitm.cs5013.panoramic.execution;

/**
 * Thrown when an external process model execution fails or exits with a non-zero code.
 */
public class ProcessExecutionException extends Exception {
    private final int exitCode;
    private final String stdout;
    private final String stderr;
    private final boolean timedOut;

    public ProcessExecutionException(String message, int exitCode, String stdout, String stderr, boolean timedOut) {
        super(message);
        this.exitCode = exitCode;
        this.stdout = stdout;
        this.stderr = stderr;
        this.timedOut = timedOut;
    }

    public ProcessExecutionException(String message, Throwable cause) {
        super(message, cause);
        this.exitCode = -1;
        this.stdout = "";
        this.stderr = "";
        this.timedOut = false;
    }

    public int getExitCode() {
        return exitCode;
    }

    public String getStdout() {
        return stdout;
    }

    public String getStderr() {
        return stderr;
    }

    public boolean isTimedOut() {
        return timedOut;
    }
}
