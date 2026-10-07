package edu.iitm.cs5013.panoramic.execution;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * Handles controlled execution of external CLI processes using Java ProcessBuilder.
 * Manages arguments, working directories, timeout enforcement, and non-blocking stream capture.
 */
public class ProcessExecutor {

    public ProcessExecutionResult execute(
            List<String> command,
            Path workingDirectory,
            Map<String, String> extraEnv,
            long timeoutSeconds) throws ProcessExecutionException {

        if (command == null || command.isEmpty()) {
            throw new ProcessExecutionException("Command cannot be empty", -1, "", "", false);
        }

        ProcessBuilder pb = new ProcessBuilder(command);
        if (workingDirectory != null) {
            File dir = workingDirectory.toFile();
            if (dir.exists() && dir.isDirectory()) {
                pb.directory(dir);
            }
        }
        if (extraEnv != null && !extraEnv.isEmpty()) {
            pb.environment().putAll(extraEnv);
        }

        long startTime = System.currentTimeMillis();
        ExecutorService streamPool = Executors.newFixedThreadPool(2);
        Process process = null;

        try {
            process = pb.start();
            final InputStream inStream = process.getInputStream();
            final InputStream errStream = process.getErrorStream();

            Future<String> stdoutFuture = streamPool.submit(() -> readStream(inStream));
            Future<String> stderrFuture = streamPool.submit(() -> readStream(errStream));

            boolean completed = process.waitFor(timeoutSeconds > 0 ? timeoutSeconds : 60, TimeUnit.SECONDS);
            long durationMs = System.currentTimeMillis() - startTime;

            if (!completed) {
                process.destroyForcibly();
                String stdout = getStreamQuietly(stdoutFuture);
                String stderr = getStreamQuietly(stderrFuture);
                throw new ProcessExecutionException(
                        "Process execution timed out after " + timeoutSeconds + " seconds",
                        -1, stdout, stderr, true);
            }

            int exitCode = process.exitValue();
            String stdout = getStreamQuietly(stdoutFuture);
            String stderr = getStreamQuietly(stderrFuture);

            if (exitCode != 0) {
                throw new ProcessExecutionException(
                        "Process exited with non-zero exit code: " + exitCode + ". Stderr: " + stderr.trim(),
                        exitCode, stdout, stderr, false);
            }

            return new ProcessExecutionResult(exitCode, stdout, stderr, durationMs, false);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            if (process != null) {
                process.destroyForcibly();
            }
            throw new ProcessExecutionException("Process was interrupted", e);
        } catch (ProcessExecutionException pe) {
            throw pe;
        } catch (Exception e) {
            if (process != null) {
                process.destroyForcibly();
            }
            throw new ProcessExecutionException("Failed to launch process: " + e.getMessage(), e);
        } finally {
            streamPool.shutdownNow();
        }
    }

    private String readStream(InputStream inputStream) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append(System.lineSeparator());
            }
            return sb.toString();
        } catch (Exception e) {
            return "";
        }
    }

    private String getStreamQuietly(Future<String> future) {
        try {
            return future.get(500, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * Replaces variable tokens such as {IMAGE_PATH}, {OUTPUT_PATH}, etc. in a command template.
     */
    public static List<String> buildCommand(List<String> template, Map<String, String> variables) {
        List<String> rendered = new ArrayList<>();
        for (String part : template) {
            String resolved = part;
            for (Map.Entry<String, String> entry : variables.entrySet()) {
                resolved = resolved.replace("{" + entry.getKey() + "}", entry.getValue());
            }
            rendered.add(resolved);
        }
        return rendered;
    }
}
