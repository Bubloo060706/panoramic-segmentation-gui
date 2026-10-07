package edu.iitm.cs5013.panoramic.controller;

import edu.iitm.cs5013.panoramic.model.ExecutionStatus;
import edu.iitm.cs5013.panoramic.model.SegmentationException;
import edu.iitm.cs5013.panoramic.model.SegmentationRequest;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Asynchronous execution coordinator.
 * Dispatches inference requests to selected model adapters on background threads
 * without blocking the JavaFX application thread.
 * Manages running, completion, error states, and result caching.
 */
public class AsyncSegmentationCoordinator {

    private final ExecutorService threadPool;
    private final Map<String, SegmentationResult> resultCache = new ConcurrentHashMap<>();

    public AsyncSegmentationCoordinator(int threadCount) {
        this.threadPool = Executors.newFixedThreadPool(Math.max(1, threadCount));
    }

    public AsyncSegmentationCoordinator() {
        this(Math.max(2, Runtime.getRuntime().availableProcessors() / 2));
    }

    /**
     * Executes multiple segmentation tools concurrently in the background.
     *
     * @param imagePath Path to the panoramic image.
     * @param tools Selected tools to execute.
     * @param listener Callback listener for events.
     * @return CompletableFuture completing when all tools have finished.
     */
    public CompletableFuture<List<SegmentationResult>> executeAllAsync(
            Path imagePath,
            List<SegmentationTool> tools,
            ToolExecutionListener listener) {

        if (tools == null || tools.isEmpty()) {
            return CompletableFuture.completedFuture(Collections.emptyList());
        }

        List<CompletableFuture<SegmentationResult>> futures = new ArrayList<>();

        for (SegmentationTool tool : tools) {
            CompletableFuture<SegmentationResult> future = CompletableFuture.supplyAsync(() -> {
                String toolId = tool.getMetadata().getId();
                String cacheKey = buildCacheKey(toolId, imagePath);

                if (listener != null) {
                    listener.onToolStarted(toolId);
                    listener.onToolProgress(toolId, "Checking cache...", 0.1);
                }

                // Check cache
                SegmentationResult cached = resultCache.get(cacheKey);
                if (cached != null && cached.isSuccess()) {
                    if (listener != null) {
                        listener.onToolProgress(toolId, "Retrieved from cache", 1.0);
                        listener.onToolSuccess(toolId, cached);
                    }
                    return cached;
                }

                if (listener != null) {
                    listener.onToolProgress(toolId, "Running inference...", 0.4);
                }

                try {
                    SegmentationRequest req = new SegmentationRequest(
                            imagePath.toAbsolutePath().toString(),
                            toolId,
                            null,
                            tool.getConfiguration().getParameters(),
                            tool.getConfiguration().getTimeoutSeconds()
                    );

                    SegmentationResult result = tool.segment(req);

                    if (listener != null) {
                        listener.onToolProgress(toolId, "Inference completed", 1.0);
                        listener.onToolSuccess(toolId, result);
                    }

                    if (result.isSuccess()) {
                        resultCache.put(cacheKey, result);
                    }
                    return result;

                } catch (SegmentationException se) {
                    String err = se.getMessage();
                    if (listener != null) {
                        listener.onToolFailure(toolId, err);
                    }
                    return SegmentationResult.failure(toolId, tool.getMetadata().getName(), err, se.getStatus());
                } catch (Throwable t) {
                    String err = "Unexpected error: " + t.getMessage();
                    if (listener != null) {
                        listener.onToolFailure(toolId, err);
                    }
                    return SegmentationResult.failure(toolId, tool.getMetadata().getName(), err, ExecutionStatus.FAILED);
                }
            }, threadPool);

            futures.add(future);
        }

        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(v -> {
                    List<SegmentationResult> results = new ArrayList<>();
                    for (CompletableFuture<SegmentationResult> f : futures) {
                        results.add(f.join());
                    }
                    return results;
                });
    }

    private String buildCacheKey(String toolId, Path imagePath) {
        try {
            long lastMod = Files.getLastModifiedTime(imagePath).toMillis();
            return toolId + ":" + imagePath.toAbsolutePath() + ":" + lastMod;
        } catch (IOException e) {
            return toolId + ":" + imagePath.toAbsolutePath();
        }
    }

    public void clearCache() {
        resultCache.clear();
    }

    public int getCacheSize() {
        return resultCache.size();
    }

    public void shutdown() {
        threadPool.shutdown();
    }
}
