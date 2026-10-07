package edu.iitm.cs5013.panoramic.execution;

/**
 * Thrown when an HTTP inference request fails or returns an error status code.
 */
public class HttpExecutionException extends Exception {
    private final int statusCode;
    private final String responseBody;

    public HttpExecutionException(String message, int statusCode, String responseBody) {
        super(message);
        this.statusCode = statusCode;
        this.responseBody = responseBody;
    }

    public HttpExecutionException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = -1;
        this.responseBody = "";
    }

    public int getStatusCode() {
        return statusCode;
    }

    public String getResponseBody() {
        return responseBody;
    }
}
