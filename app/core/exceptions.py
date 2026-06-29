class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotLoadedError(ServiceError):
    def __init__(self):
        super().__init__("모델이 로드되지 않았습니다.", status_code=503)


class UploadValidationError(ServiceError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)
