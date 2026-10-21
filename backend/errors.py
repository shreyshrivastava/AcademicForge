"""Public error contracts. Internal exceptions stay in server logs."""


class ServiceError(RuntimeError):
    status_code = 503
    code = "service_unavailable"
    public_message = "The service is temporarily unavailable. Please try again."


class CapacityError(ServiceError):
    status_code = 429
    code = "capacity_exceeded"
    public_message = "The service is busy. Please try again shortly."


class SourceUnavailable(ServiceError):
    code = "sources_unavailable"
    public_message = "Paper sources are temporarily unavailable. Please try again."


class GenerationUnavailable(ServiceError):
    code = "generation_unavailable"
    public_message = "Generation is unavailable. Check model readiness and try again."


class StreamInterrupted(GenerationUnavailable):
    code = "stream_interrupted"
    public_message = "Generation was interrupted. Start a new attempt to replace the partial answer."
