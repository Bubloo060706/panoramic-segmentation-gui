package edu.iitm.cs5013.panoramic.execution;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Result returned from an external HTTP inference service.
 */
public final class HttpExecutionResult {
    private final int statusCode;
    private final String responseBody;
    private final Map<String, List<String>> headers;
    private final long durationMs;

    public HttpExecutionResult(int statusCode, String responseBody, Map<String, List<String>> headers, long durationMs) {
        this.statusCode = statusCode;
        this.responseBody = responseBody != null ? responseBody : "";
        this.headers = headers != null ? Collections.unmodifiableMap(headers) : Collections.emptyMap();
        this.durationMs = durationMs;
    }

    public int getStatusCode() {
        return statusCode;
    }

    public String getResponseBody() {
        return responseBody;
    }

    public Map<String, List<String>> getHeaders() {
        return headers;
    }

    public long getDurationMs() {
        return durationMs;
    }

    public boolean isSuccess() {
        return statusCode >= 200 && statusCode < 300;
    }
}
