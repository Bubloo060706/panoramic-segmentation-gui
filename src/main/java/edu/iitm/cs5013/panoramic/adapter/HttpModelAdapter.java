package edu.iitm.cs5013.panoramic.adapter;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.execution.HttpExecutionException;
import edu.iitm.cs5013.panoramic.execution.HttpExecutionResult;
import edu.iitm.cs5013.panoramic.execution.HttpExecutor;
import edu.iitm.cs5013.panoramic.model.*;
import edu.iitm.cs5013.panoramic.normalization.ResultNormalizer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;

/**
 * Adapter for remote HTTP/REST inference services.
 */
public class HttpModelAdapter implements SegmentationAdapter<String> {

    private final ToolMetadata metadata;
    private final ToolConfig config;
    private final HttpExecutor httpExecutor;
    private final ResultNormalizer normalizer;
    private final ObjectMapper objectMapper;
    private int cachedInputWidth = -1;
    private int cachedInputHeight = -1;

    public HttpModelAdapter(ToolMetadata metadata, ToolConfig config, HttpExecutor httpExecutor, ResultNormalizer normalizer) {
        this.metadata = Objects.requireNonNull(metadata, "metadata cannot be null");
        this.config = Objects.requireNonNull(config, "config cannot be null");
        this.httpExecutor = httpExecutor != null ? httpExecutor : new HttpExecutor();
        this.normalizer = normalizer != null ? normalizer : new ResultNormalizer();
        this.objectMapper = new ObjectMapper();
    }

    public HttpModelAdapter(ToolConfig config) {
        this(
                new ToolMetadata(
                        config.getId(),
                        config.getName(),
                        config.getDescription(),
                        "1.0.0",
                        Set.of(SegmentationType.SEMANTIC),
                        Set.of("jpg", "jpeg", "png")
                ),
                config,
                new HttpExecutor(),
                new ResultNormalizer()
        );
    }

    @Override
    public ToolMetadata getMetadata() {
        return metadata;
    }

    @Override
    public ToolConfig getConfiguration() {
        return config;
    }

    @Override
    public boolean isAvailable() {
        if (!config.isEnabled() || config.getEndpointUrl().isBlank()) {
            return false;
        }
        return httpExecutor.ping(config.getEndpointUrl(), 2);
    }

    @Override
    public void preprocess(SegmentationRequest request) throws SegmentationException {
        request.validate();
        try {
            BufferedImage img = ImageIO.read(request.getImageNioPath().toFile());
            if (img == null) {
                throw new SegmentationException(config.getId(), "Unable to decode panoramic image: " + request.getImagePath());
            }
            this.cachedInputWidth = img.getWidth();
            this.cachedInputHeight = img.getHeight();
        } catch (IOException e) {
            throw new SegmentationException(config.getId(), "Failed to read image: " + e.getMessage(), e);
        }
    }

    @Override
    public String executeRaw(SegmentationRequest request) throws SegmentationException {
        try {
            Map<String, Object> payloadMap = new HashMap<>();
            payloadMap.put("imagePath", request.getImageNioPath().toAbsolutePath().toString());
            payloadMap.put("parameters", request.getParameters());
            String jsonPayload = objectMapper.writeValueAsString(payloadMap);

            HttpExecutionResult result = httpExecutor.postJson(
                    config.getEndpointUrl(),
                    jsonPayload,
                    Map.of("User-Agent", "PanoramicSegmentationGUI/1.0"),
                    config.getTimeoutSeconds()
            );

            if (!result.isSuccess() || result.getResponseBody().isBlank()) {
                throw new SegmentationException(config.getId(), "HTTP endpoint returned empty response or error code " + result.getStatusCode());
            }
            return result.getResponseBody();
        } catch (HttpExecutionException he) {
            throw translateError(he, request);
        } catch (Exception e) {
            throw new SegmentationException(config.getId(), "Failed to prepare HTTP payload: " + e.getMessage(), e);
        }
    }

    @Override
    public SegmentationResult postprocess(String rawOutput, SegmentationRequest request) throws SegmentationException {
        return normalizer.normalizeFromJson(rawOutput, config, cachedInputWidth, cachedInputHeight);
    }

    @Override
    public SegmentationException translateError(Throwable t, SegmentationRequest request) {
        if (t instanceof SegmentationException se) {
            return se;
        }
        if (t instanceof HttpExecutionException he) {
            return new SegmentationException(config.getId(),
                    "HTTP service error (status " + he.getStatusCode() + "): " + he.getMessage(),
                    ExecutionStatus.FAILED, he);
        }
        return new SegmentationException(config.getId(), "Unexpected HTTP adapter error: " + t.getMessage(), ExecutionStatus.FAILED, t);
    }
}
