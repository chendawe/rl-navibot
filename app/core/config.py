from pydantic_settings import BaseSettings
from typing import Any


class Settings(BaseSettings):
    # -------------------
    # 项目基础配置
    # -------------------
    API_PREFIX: str = "/api"  # 接口统一前缀
    PROJECT_NAME: str = "FastAPI Server"
    VERSION: str = "1.0.0"

    # -------------------
    # 运行配置
    # -------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # -------------------
    # 跨域配置
    # -------------------
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ]

    # -------------------
    # 数据库（可选）
    # -------------------
    DATABASE_URL: str = "sqlite:///./test.db"

    # -------------------
    # JWT 安全配置（可选）
    # -------------------
    SECRET_KEY: str = "your-secret-key-keep-it-safe"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7天

    # -------------------
    # 静态文件 / 前端路径
    # -------------------
    FRONTEND_DIR: str = "./frontend"
    STATIC_DIR: str = "./static"

    class Config:
        case_sensitive = True


# 全局唯一配置实例
settings = Settings()