package edu.iitm.cs5013.panoramic.execution;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ExecutionBoundaryTest {

    private final ProcessExecutor processExecutor = new ProcessExecutor();

    @Test
    void testCommandTemplateSubstitution() {
        List<String> template = List.of("python", "segment.py", "--input", "{IMAGE_PATH}", "--out", "{OUTPUT_PATH}");
        Map<String, String> vars = Map.of(
                "IMAGE_PATH", "C:/data/pano1.jpg",
                "OUTPUT_PATH", "C:/data/out.json"
        );

        List<String> built = ProcessExecutor.buildCommand(template, vars);
        assertEquals(6, built.size());
        assertEquals("C:/data/pano1.jpg", built.get(3));
        assertEquals("--out", built.get(4));
        assertEquals("C:/data/out.json", built.get(5));
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void testSuccessfulProcessExecutionOnWindows() throws ProcessExecutionException {
        List<String> command = List.of("cmd.exe", "/c", "echo", "panoramic_ok");
        ProcessExecutionResult result = processExecutor.execute(command, null, null, 10);

        assertTrue(result.isSuccess());
        assertEquals(0, result.getExitCode());
        assertTrue(result.getStdout().contains("panoramic_ok"));
        assertFalse(result.isTimedOut());
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void testNonZeroExitCodeThrows() {
        List<String> command = List.of("cmd.exe", "/c", "exit 2");
        ProcessExecutionException ex = assertThrows(ProcessExecutionException.class, () ->
                processExecutor.execute(command, null, null, 10));

        assertEquals(2, ex.getExitCode());
        assertFalse(ex.isTimedOut());
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void testProcessTimeoutEnforcement() {
        // Run PowerShell sleep for 5 seconds with a 1 second timeout
        List<String> command = List.of("powershell.exe", "-Command", "Start-Sleep -Seconds 5");
        ProcessExecutionException ex = assertThrows(ProcessExecutionException.class, () ->
                processExecutor.execute(command, null, null, 1));

        assertTrue(ex.isTimedOut());
    }

    @Test
    void testHttpExecutionInvalidUrlThrows() {
        HttpExecutor httpExecutor = new HttpExecutor();
        assertThrows(HttpExecutionException.class, () ->
                httpExecutor.postJson("", "{}", null, 5));
    }
}
