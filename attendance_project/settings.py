from pathlib import Path
import os
import dj_database_url


DATABASES = {
    'default': dj_database_url.config(
        default=os.environ.get('DATABASE_URL', 'postgres://avnadmin:AVNS_MwItjpFgLS4GPOMZU8f@pg-2318a3f0-iconictk2-0c57.j.aivencloud.com:12207/test?sslmode=require')
    )
}
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
# Quick-start development settings - unsuitable for production
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'your-default-secret-key')

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'jazzmin',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'attendance_app',
    'recognize_app',
    'rest_framework_simplejwt',
    'corsheaders',
    'django_filters',

]
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
}
from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=30),  
    'REFRESH_TOKEN_LIFETIME': timedelta(days=30), 
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': 'your-secret-key',  # Thay bằng key bí mật của bạn
    'AUTH_HEADER_TYPES': ('Bearer',),
}
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOWED_ORIGINS = [
    "https://f58c-183-81-127-52.ngrok-free.app",  # địa chỉ frontend React
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'attendance_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'attendance_project.wsgi.application'

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql',
#         'NAME': 'datapbl5',  # Tên database
#         'USER': 'root',  # User mặc định của MySQL trên XAMPP
#         'PASSWORD': '',  # XAMPP mặc định không có mật khẩu
#         'HOST': '127.0.0.1',
#         'PORT': '3306',
#         'OPTIONS': {
#             'charset': 'utf8mb4',
#             'init_command': "SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;",
#         }
#     }
# }

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

AUTH_USER_MODEL = 'attendance_app.CustomUser'

LANGUAGE_CODE = 'vi'  # Chuyển sang tiếng Việt
TIME_ZONE = 'Asia/Ho_Chi_Minh'  # Chuyển sang múi giờ Việt Nam

USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'

STATICFILES_DIRS = [BASE_DIR / 'static']

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')