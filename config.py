import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "smartedu-dev-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "smartedu.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STUDENTS_PER_PAGE = 25
