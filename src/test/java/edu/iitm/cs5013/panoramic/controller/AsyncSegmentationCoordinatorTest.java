package edu.iitm.cs5013.panoramic.controller;

import edu.iitm.cs5013.panoramic.adapter.DeterministicTestAdapter;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class AsyncSegmentationCoordinatorTest {

    private AsyncSegmentationCoordinator coordinator;
    private final Path testImgPath = Path.of("src/test/resources/test_panorama.png");

    @BeforeEach
    void setUp() {
        coordinator = new AsyncSegmentationCoordinator(2);
    }

    @AfterEach
    void tearDown() {
        coordinator.shutdown();
    }

    @Test
    void testConcurrentToolExecutionAndCaching() throws Exception {
        SegmentationTool tool1 = new DeterministicTestAdapter("mock-1", "Mock 1", false);
        SegmentationTool tool2 = new DeterministicTestAdapter("mock-2", "Mock 2", false);

        List<String> events = new ArrayList<>();
        ToolExecutionListener listener = new ToolExecutionListener() {
            @Override
            public void onToolStarted(String toolId) {
                events.add("start:" + toolId);
            }
            @Override
            public void onToolProgress(String toolId, String message, double progress) {}
            @Override
            public void onToolSuccess(String toolId, SegmentationResult result) {
                events.add("success:" + toolId);
            }
            @Override
            public void onToolFailure(String toolId, String errorMessage) {
                events.add("fail:" + toolId);
            }
        };

        CompletableFuture<List<SegmentationResult>> future = coordinator.executeAllAsync(
                testImgPath,
                List.of(tool1, tool2),
                listener
        );

        List<SegmentationResult> results = future.get(10, TimeUnit.SECONDS);

        assertEquals(2, results.size());
        assertTrue(results.get(0).isSuccess());
        assertTrue(results.get(1).isSuccess());
        assertTrue(events.contains("start:mock-1"));
        assertTrue(events.contains("start:mock-2"));
        assertTrue(events.contains("success:mock-1"));
        assertTrue(events.contains("success:mock-2"));

        // Verify result caching
        assertEquals(2, coordinator.getCacheSize());

        // Re-executing should hit cache
        events.clear();
        CompletableFuture<List<SegmentationResult>> cachedFuture = coordinator.executeAllAsync(
                testImgPath,
                List.of(tool1),
                listener
        );
        List<SegmentationResult> cachedResults = cachedFuture.get(5, TimeUnit.SECONDS);
        assertEquals(1, cachedResults.size());
        assertTrue(cachedResults.get(0).isSuccess());
        assertTrue(events.contains("success:mock-1"));
    }

    @Test
    void testFailureStateDoesNotCrashCoordinator() throws Exception {
        SegmentationTool failTool = new DeterministicTestAdapter("fail-tool", "Failing Tool", true);

        CompletableFuture<List<SegmentationResult>> future = coordinator.executeAllAsync(
                testImgPath,
                List.of(failTool),
                null
        );

        List<SegmentationResult> results = future.get(5, TimeUnit.SECONDS);
        assertEquals(1, results.size());
        assertFalse(results.get(0).isSuccess());
        assertNotNull(results.get(0).getErrorMessage());
    }
}
