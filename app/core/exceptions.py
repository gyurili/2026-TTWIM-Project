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


class StepLengthError(ServiceError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class DuplicateUserError(ServiceError):
    def __init__(self, message: str = "User ID already exists"):
        super().__init__(message, status_code=400)


class InvalidCredentialsError(ServiceError):
    def __init__(self, message: str = "Incorrect user ID or password"):
        super().__init__(message, status_code=401)


class UnauthorizedError(ServiceError):
    def __init__(self, message: str = "로그인 토큰이 필요합니다."):
        super().__init__(message, status_code=401)


class DtwNotFoundError(ServiceError):
    def __init__(self, message: str = "저장된 DTW 캘리브레이션이 없습니다."):
        super().__init__(message, status_code=404)


class GaitSaveError(ServiceError):
    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class SessionNotFoundError(ServiceError):
    def __init__(self, message: str = "러닝 세션을 찾을 수 없습니다."):
        super().__init__(message, status_code=404)


class SessionForbiddenError(ServiceError):
    def __init__(self, message: str = "본인의 세션이 아닙니다."):
        super().__init__(message, status_code=403)


class FeedbackNotFoundError(ServiceError):
    def __init__(self, message: str = "생성된 피드백이 없습니다."):
        super().__init__(message, status_code=404)
